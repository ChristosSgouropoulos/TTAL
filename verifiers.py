import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import sys
import os
import tempfile
import uuid
from transformers import ClapModel, ClapProcessor, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from audiobox_aesthetics.infer import AesPredictor

# --- BEATs imports ---
BEATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unilm', 'beats')
if BEATS_DIR not in sys.path:
    sys.path.insert(0, BEATS_DIR)
from BEATs import BEATs, BEATsConfig


# AudioSet label ontology (527 classes)
# Download from: https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv
AUDIOSET_LABELS_URL = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"


def load_audioset_labels(csv_path=None):
    """Load AudioSet class index → display_name mapping.
    
    Falls back to downloading the CSV if no local path is provided.
    Returns:
        dict: {int_index: str_display_name}
    """
    if csv_path is not None and os.path.exists(csv_path):
        import csv
        labels = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header: index, mid, display_name
            for row in reader:
                labels[int(row[0])] = row[2]
        return labels

    # Fallback: hardcoded subset of common AudioSet labels
    # In production, download the full CSV from AUDIOSET_LABELS_URL
    try:
        import urllib.request, csv, io
        response = urllib.request.urlopen(AUDIOSET_LABELS_URL)
        text = response.read().decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        next(reader)  # skip header
        labels = {}
        for row in reader:
            labels[int(row[0])] = row[2]
        return labels
    except Exception:
        # Minimal fallback so the module still loads
        print("[WARNING] Could not load AudioSet labels. BEATs keyword matching will be disabled.")
        return {}


class CLAPRewardModel:
    """CLAP-based reward: measures text-audio alignment via cosine similarity."""

    def __init__(self, device="cuda"):
        self.clap = ClapModel.from_pretrained("laion/larger_clap_general").to(device).eval()
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        self.device = device
        self._cached_text_features = None
        self._cached_prompt = None
        self._resample_to_48k = T.Resample(44100, 48000)

    @torch.no_grad()
    def __call__(self, waveforms, text_prompt):
        """Score decoded waveforms against text prompt.
        Args:
            waveforms: (N, channels, samples) tensor at 48kHz
            text_prompt: str or list of str
        Returns:
            scores: (N,) tensor of cosine similarities in [-1, 1]
        """
        # Cache text features if prompt hasn't changed
        if text_prompt != self._cached_prompt:
            text_inputs = self.processor(text=text_prompt, return_tensors="pt", padding=True)
            text_features = self.clap.get_text_features(
                **{k: v.to(self.device) for k, v in text_inputs.items()}
            )
            # In transformers >= 5.5, this returns an object instead of a tensor
            if not isinstance(text_features, torch.Tensor):
                text_features = getattr(text_features, "pooler_output", text_features[0])
            
            # L2-normalize to match CLAP's contrastive training convention
            self._cached_text_features = F.normalize(text_features, p=2, dim=-1)
            self._cached_prompt = text_prompt

        text_features = self._cached_text_features

        # Process each audio sample
        scores = []
        for i in range(waveforms.shape[0]):
            audio_mono = waveforms[i].mean(dim=0).cpu()  # mono
            audio_48k = self._resample_to_48k(audio_mono).numpy()
            audio_inputs = self.processor(audio=audio_48k, sampling_rate=48000, return_tensors="pt")
                
            audio_features = self.clap.get_audio_features(
                **{k: v.to(self.device) for k, v in audio_inputs.items()}
            )
            # Extract tensor if an object is returned
            if not isinstance(audio_features, torch.Tensor):
                audio_features = getattr(audio_features, "pooler_output", audio_features[0])
                
            # L2-normalize audio embeddings before similarity
            audio_features = F.normalize(audio_features, p=2, dim=-1)
            score = F.cosine_similarity(text_features, audio_features)
            scores.append(score)

        return torch.cat(scores)


class AudioboxAESRewardModel:
    """Meta Audiobox Aesthetics reward: measures production quality + content enjoyment.
    
    Uses AesPredictor from audiobox_aesthetics.infer which handles:
      - Loading the pretrained model from HuggingFace (facebook/audiobox-aesthetics)
      - Resampling to 16kHz
      - Windowed inference (10s windows with 10s hop)
      - Returns CE (Content Enjoyment), CU, PC, PQ scores
    """

    def __init__(self, device="cuda"):
        # AesPredictor loads from HF when checkpoint_pth=None
        self.predictor = AesPredictor(checkpoint_pth=None, data_col="path")
        self.device = device
        self._tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".aes_tmp")
        os.makedirs(self._tmp_dir, exist_ok=True)

    @torch.no_grad()
    def __call__(self, waveforms, sample_rate=44100):
        """Score decoded waveforms for aesthetic quality.
        Args:
            waveforms: (N, channels, samples) tensor
        Returns:
            scores: (N,) tensor — average of PQ + CE axes
        """
        import torchaudio
        scores = []
        batch = []

        import uuid
        batch_id = uuid.uuid4().hex
        
        # AesPredictor expects list of dicts with "path" key pointing to audio files
        # We save waveforms to temp files, run inference, then clean up
        tmp_paths = []
        for i in range(waveforms.shape[0]):
            tmp_path = os.path.join(self._tmp_dir, f"aes_tmp_{batch_id}_{i}.wav")
            audio = waveforms[i].cpu()
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            torchaudio.save(tmp_path, audio, sample_rate)
            batch.append({"path": tmp_path})
            tmp_paths.append(tmp_path)

        # Run inference
        results = self.predictor.forward(batch)

        # Extract PQ + CE and average
        for result in results:
            pq = result.get("PQ", 0.0)
            ce = result.get("CE", 0.0)
            scores.append((pq + ce) / 2.0)

        # Clean up temp files
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass

        return torch.tensor(scores, dtype=torch.float32)


class BEATsRewardModel:
    """BEATs-based reward: measures semantic correctness via AudioSet classification."""

    def __init__(self, checkpoint_path=None, device="cuda", audioset_labels_csv=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

        # Load AudioSet label ontology for keyword matching
        self.label_names = load_audioset_labels(audioset_labels_csv)

    def _prompt_to_label_indices(self, text_prompt):
        """Find AudioSet label indices whose names overlap with prompt keywords."""
        if not self.label_names:
            return []
        prompt_words = set(text_prompt.lower().split())
        # Also try bigrams for multi-word labels like "dog bark"
        prompt_text = text_prompt.lower()
        print(prompt_text)
        print(prompt_words)
        indices = []
        for idx, name in self.label_names.items():
            name_lower = name.lower()
            # Match if any prompt word appears in the label name, or vice versa
            if (any(word in name_lower for word in prompt_words)
                    or any(label_word in prompt_text for label_word in name_lower.split())):
                indices.append(idx)
        return indices

    @torch.no_grad()
    def __call__(self, waveforms, text_prompt, sample_rate=44100, target_sr=16000):
        """Score how well audio matches expected sound events from prompt.
        Args:
            waveforms: (N, channels, samples) tensor
            text_prompt: str
        Returns:
            scores: (N,) tensor in [0, 1] — classification confidence for prompt-relevant classes
        """
        import torchaudio.transforms as T
        resample = T.Resample(sample_rate, target_sr).to(self.device)

        target_indices = self._prompt_to_label_indices(text_prompt)
        print(target_indices)
        scores = []

        for i in range(waveforms.shape[0]):
            # Resample to 16kHz mono
            audio_16k = resample(waveforms[i].mean(dim=0, keepdim=True).to(self.device))
            padding_mask = torch.zeros(1, audio_16k.shape[-1], dtype=torch.bool, device=self.device)

            # extract_features returns (features, padding_mask) — features are embeddings, not logits.
            # For classification, we use the model's `predict` if available,
            # otherwise use the raw feature similarity as a proxy.
            features, _ = self.model.extract_features(audio_16k, padding_mask=padding_mask)
            print(features.shape)
            
            # When the BEATs model is fine-tuned, its `extract_features` method internally applies the 
            # classification head (`self.predictor`) and directly returns the sigmoid probabilities `lprobs`.
            if hasattr(self.model, 'predictor') and self.model.predictor is not None and features.shape[-1] == self.model.predictor.out_features:
                probs = features  # already (1, num_classes) probabilities!
                if target_indices:
                    score = probs[0, target_indices].max().item()
                else:
                    score = probs.max().item()
            else:
                # BEATs extract_features returns frame-level features (1, T, D).
                # Use mean-pooled features as the representation.
                pooled = features.mean(dim=1)  # (1, D)
                # No classifier head — use feature norm as a rough quality proxy
                score = pooled.norm(dim=-1).item() / 100.0  # normalize to ~[0,1]

            scores.append(score)

        return torch.tensor(scores, dtype=torch.float32)


class QwenAudioWrapper:
    """Wrapper around Qwen2.5-Omni that extracts Yes/No logits for a given audio+question.
 
    Handles all preprocessing: resampling to 16kHz, saving to temp file for the
    chat template, and cleaning up afterwards.
    """
 
    def __init__(self, model_id="Qwen/Qwen2.5-Omni-7B", device="cuda"):
        self.device = device
        self.model_id = model_id
 
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        # Pre-compute token IDs for "Yes" and "No" and verify they are single tokens
        yes_ids = self.processor.tokenizer("Yes", add_special_tokens=False).input_ids
        no_ids  = self.processor.tokenizer("No",  add_special_tokens=False).input_ids
        assert len(yes_ids) == 1, f"'Yes' tokenizes to {yes_ids}, expected a single token"
        assert len(no_ids)  == 1, f"'No' tokenizes to {no_ids}, expected a single token"
        self.yes_token_id = yes_ids[0]
        self.no_token_id  = no_ids[0]
 
        # Resampler: TangoFlux outputs 44100 Hz, Qwen expects 16000 Hz
        self._resample_44k = T.Resample(44100, 16000)
 
        # Temp directory for audio files required by Qwen's chat template
        self._tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".qwen_tmp")
        os.makedirs(self._tmp_dir, exist_ok=True)
 
    def _resample_to_16k(self, waveform_1d, source_sr=44100):
        """Resample a 1-D waveform tensor from source_sr to 16 kHz."""
        if source_sr == 16000:
            return waveform_1d
        elif source_sr == 44100:
            return self._resample_44k(waveform_1d)
        else:
            resampler = T.Resample(source_sr, 16000)
            return resampler(waveform_1d)
 
    @torch.no_grad()
    def __call__(self, waveform, question, source_sr=44100):
        """Run Qwen2.5-Omni and return Yes/No logits.
 
        Args:
            waveform:  (C, T) or (T,) tensor — a single audio clip
            question:  str — the yes/no question to ask about the audio
            source_sr: int — sample rate of the input waveform
 
        Returns:
            dict with keys 'logits_yes' and 'logits_no' (float values)
        """
        # --- 1. Ensure mono 1-D tensor ---
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)  # stereo -> mono
 
        # --- 2. Resample to 16 kHz (Qwen's expected rate) ---
        waveform_16k = self._resample_to_16k(waveform.cpu(), source_sr)
 
        # --- 3. Save to a temp WAV file so the chat template gets a real path ---
        tmp_path = os.path.join(self._tmp_dir, f"qwen_{uuid.uuid4().hex}.wav")
        torchaudio.save(tmp_path, waveform_16k.unsqueeze(0), 16000)
 
        try:
            # --- 4. Build the chat template with the real audio path ---
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Your role is to listen attentively to the given audio and decide "
                                "whether the provided text accurately and completely describes what "
                                "is heard. Make your judgment strictly based on the sounds in the "
                                "audio — do not guess, imagine, or add information that is not "
                                "clearly audible. If something is missing, unclear, or uncertain, "
                                "do not assume it exists. Your task: given an audio clip and a text "
                                "description, carefully compare the text with what you actually hear. "
                                "Identify the main sound events (such as speech, background noise, "
                                "music, or environmental sounds) and decide whether the text correctly "
                                "reflects them. Always respond objectively and concisely, using "
                                '"yes" or "no" to indicate whether the text matches the audio content.'
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": tmp_path},
                        {"type": "text", "text": question},
                    ],
                },
            ]
 
            # ✅ apply_chat_template via the Omni processor
            text_prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
 
            # --- 5. Tokenize text + extract audio features ---
            # ✅ padding=True is required by Qwen2.5-Omni processor
            inputs = self.processor(
                text=text_prompt,
                audio=[waveform_16k.numpy()],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
 
            # --- 6. Forward pass (teacher-forcing style) to get logits ---
            # ✅ use_audio_in_video=False keeps the model in audio-only mode
            outputs = self.model(
                **inputs
            )
 
            # --- 7. The logits at the last input position predict the first new token ---
            next_token_logits = outputs.logits[0, -1, :]
 
            yes_logit = next_token_logits[self.yes_token_id].item()
            no_logit  = next_token_logits[self.no_token_id].item()
 
            return {
                "logits_yes": yes_logit,
                "logits_no":  no_logit,
            }
 
        finally:
            # --- 8. Always clean up the temp file ---
            try:
                os.remove(tmp_path)
            except OSError:
                pass
 
 
class AQAScoreRewardModel:
    """AQAScore-style reward: estimates P("yes" | audio, question(prompt)).
 
    Self-contained verifier — automatically creates a QwenAudioWrapper when
    no external audio_llm is provided. Works identically to CLAP / AES / BEATs.
    """
 
    def __init__(
        self,
        audio_llm=None,
        model_id="Qwen/Qwen2.5-Omni-7B",
        device="cuda",
    ):
        # If no external audio LLM provided, create QwenAudioWrapper automatically
        self.audio_llm = audio_llm if audio_llm is not None else QwenAudioWrapper(
            model_id=model_id, device=device
        )
        self.device = device
 
    # ---------------------------------------------------
    # Prompt construction
    # ---------------------------------------------------
    def _build_question(self, text_prompt):
        """Build a single yes/no question from the full text prompt."""
        return (
            f'Does this audio contain the sound events described by the text: '
            f'"{text_prompt}"? Answer yes or no.'
        )
 
    # ---------------------------------------------------
    # Score one waveform against one question
    # ---------------------------------------------------
    def _score_with_audio_llm(self, waveform, question):
        """Get P(yes) from the audio LLM for a single waveform + question."""
        outputs = self.audio_llm(waveform, question)
 
        yes_logit = outputs["logits_yes"]
        no_logit  = outputs["logits_no"]
 
        # Softmax over the two logits to get a probability
        score = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[0]
        return score.item()
 
    # ---------------------------------------------------
    # Main call — follows the same (waveforms, text_prompt) interface
    # ---------------------------------------------------
    @torch.no_grad()
    def __call__(self, waveforms, text_prompt):
        """Score a batch of waveforms against a text prompt.
 
        Args:
            waveforms:   (N, C, T) or (N, T) tensor
            text_prompt: str
 
        Returns:
            scores: (N,) tensor of P(yes) values in [0, 1]
        """
        question = self._build_question(text_prompt)  # ✅ build once, reuse for all samples
        scores = []
 
        for i in range(waveforms.shape[0]):
            waveform = waveforms[i].to(self.device)
            score = self._score_with_audio_llm(waveform, question)
            scores.append(score)
 
        return torch.tensor(scores, dtype=torch.float32)


class EnsembleAudioReward:
    """Flexible reward ensemble — loads only the verifiers specified in verifier_list."""

    def __init__(
        self,
        device="cuda",
        verifier_list=None,
        clap_weight=0.4,
        aes_weight=0.3,
        beats_weight=0.3,
        aqa_weight=0.3,
    ):
        if verifier_list is None:
            verifier_list = ["clap", "aes", "beats"]
        self.verifier_list = [v.lower() for v in verifier_list]

        # Only load requested verifiers
        self.clap = CLAPRewardModel(device=device) if "clap" in self.verifier_list else None
        self.aes = AudioboxAESRewardModel(device=device) if "aes" in self.verifier_list else None
        self.beats = BEATsRewardModel(device=device) if "beats" in self.verifier_list else None
        self.aqa = AQAScoreRewardModel(device=device) if "aqa" in self.verifier_list else None

        # Store raw weights, re-normalize to sum to 1 among active verifiers
        raw = {}
        if self.clap is not None:
            raw["clap"] = clap_weight
        if self.aes is not None:
            raw["aes"] = aes_weight
        if self.beats is not None:
            raw["beats"] = beats_weight
        if self.aqa is not None:
            raw["aqa"] = aqa_weight
        total = sum(raw.values()) or 1.0
        self.clap_weight = raw.get("clap", 0) / total
        self.aes_weight = raw.get("aes", 0) / total
        self.beats_weight = raw.get("beats", 0) / total
        self.aqa_weight = raw.get("aqa", 0) / total

        print(f"  Active verifiers: {self.verifier_list}")
        print(f"  Normalized weights: CLAP={self.clap_weight:.2f}, "
              f"AES={self.aes_weight:.2f}, BEATs={self.beats_weight:.2f}, "
              f"AQA={self.aqa_weight:.2f}")

    def __call__(self, waveforms, text_prompt):
        """Score waveforms with all active verifiers and return weighted combination.

        Args:
            waveforms: (N, channels, samples) decoded audio
            text_prompt: str
        Returns:
            combined_scores: (N,) tensor in [0, 1]
        """
        N = waveforms.shape[0]
        combined = torch.zeros(N)

        if self.clap is not None:
            clap_norm = (self.clap(waveforms, text_prompt).cpu() + 1.0) / 2.0
            combined += self.clap_weight * clap_norm
        if self.aes is not None:
            aes_norm = self.aes(waveforms).cpu() / 5.0
            combined += self.aes_weight * aes_norm
        if self.beats is not None:
            beats_norm = self.beats(waveforms, text_prompt).cpu()
            combined += self.beats_weight * beats_norm
        if self.aqa is not None:
            # AQAScore already returns P(yes) in [0, 1] — no normalization needed
            aqa_scores = self.aqa(waveforms, text_prompt).cpu()
            combined += self.aqa_weight * aqa_scores

        return combined

    def score_with_details(self, waveforms, text_prompt):
        """Returns all individual + combined scores for logging/analysis."""
        details = {}
        N = waveforms.shape[0]
        combined = torch.zeros(N)

        if self.clap is not None:
            clap_scores = self.clap(waveforms, text_prompt).cpu()
            details["clap"] = clap_scores
            combined += self.clap_weight * (clap_scores + 1.0) / 2.0
        if self.aes is not None:
            aes_scores = self.aes(waveforms).cpu()
            details["aes"] = aes_scores
            combined += self.aes_weight * aes_scores / 5.0
        if self.beats is not None:
            beats_scores = self.beats(waveforms, text_prompt).cpu()
            details["beats"] = beats_scores
            combined += self.beats_weight * beats_scores
        if self.aqa is not None:
            aqa_scores = self.aqa(waveforms, text_prompt).cpu()
            details["aqa"] = aqa_scores
            combined += self.aqa_weight * aqa_scores

        details["combined"] = combined
        return details