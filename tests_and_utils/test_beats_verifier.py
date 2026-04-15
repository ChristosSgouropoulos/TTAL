import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

from transformers import ClapModel, ClapProcessor
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



class BEATsRewardModel:
    """BEATs-based reward: measures semantic correctness via AudioSet classification."""

    def __init__(self, checkpoint_path=None, device="cuda", audioset_labels_csv=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "BEATs_iter3_plus_AS2M.pt"
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
if __name__=='__main__':
    beats_reward = BEATsRewardModel(checkpoint_path="/home/theodoros_giannakopoulos_demokri/TTAL/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt", device="cuda:0", audioset_labels_csv=None)
    text_prompt ="Dog Barking while raining and Bells ringing"
    wav = torch.rand(1,1,44100*10)
    print(beats_reward(wav, text_prompt))

    
    