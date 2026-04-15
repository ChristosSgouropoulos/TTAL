import os
import re
import json
import torch
import torchaudio
from tqdm import tqdm
from verifiers import AQAScoreRewardModel

# For the old model
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import torchaudio.transforms as T
import uuid

# --- CONFIGURATION ---
AUDIO_DIR = "/home/theodoros_giannakopoulos_demokri/TTAL/stable-audio-metrics/reference_886"
JSON_PATH = "/mnt/storage/audiocaps/test_audiocaps.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_captions():
    print(f"Loading captions from {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    # Create mapping: filename -> caption
    mapping = {}
    for item in data:
        filename = os.path.basename(item["location"])
        mapping[filename] = item["captions"]
        
    return mapping

# --- STANDALONE WRAPPER FOR OLD QWEN 2 AUDIO ---
# Because verifiers.py is now strictly hardcoded to use Omni's "Thinker" architecture,
# we need to natively instantiate the old architecture strictly within this benchmark
# to satisfy the user's requirement to cleanly compare both without modifying verifiers.py.
class OldQwen2AudioWrapper:
    def __init__(self, model_id="Qwen/Qwen2-Audio-7B-Instruct", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        yes_ids = self.processor.tokenizer("Yes", add_special_tokens=False).input_ids
        no_ids = self.processor.tokenizer("No", add_special_tokens=False).input_ids
        self.yes_token_id = yes_ids[0]
        self.no_token_id = no_ids[0]
        self._resample_44k = T.Resample(44100, 16000).to(device)
        self._tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".qwen_tmp")
        os.makedirs(self._tmp_dir, exist_ok=True)

    def _resample(self, waveform, sr):
        if sr == 16000:
            return waveform
        if sr == 44100:
            return self._resample_44k(waveform.to(self.device)).cpu()
        return T.Resample(sr, 16000)(waveform)

    @torch.no_grad()
    def __call__(self, waveform, question, sr=22050):
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        
        waveform_16k = self._resample(waveform, sr)
        tmp_path = os.path.join(self._tmp_dir, f"old_qwen_{uuid.uuid4().hex}.wav")
        torchaudio.save(tmp_path, waveform_16k.unsqueeze(0), 16000)
        
        try:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": "Your role is to listen attentively to the given audio and decide whether the provided text accurately and completely describes what is heard. Make your judgment strictly based on the sounds in the audio — do not guess, imagine, or add information that is not clearly audible. If something is missing, unclear, or uncertain, do not assume it exists. Your task: given an audio clip and a text description, carefully compare the text with what you actually hear. Identify the main sound events (such as speech, background noise, music, or environmental sounds) and decide whether the text correctly reflects them. Always respond objectively and concisely, using “yes” or “no” to indicate whether the text matches the audio content."}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": tmp_path},
                    {"type": "text", "text": question},
                ]}
            ]
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            
            inputs = self.processor(
                text=text_prompt,
                audios=waveform_16k.cpu().numpy(),
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            yes_logit = next_token_logits[self.yes_token_id].item()
            no_logit = next_token_logits[self.no_token_id].item()
            score = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[0].item()
            return score
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def main():
    captions_map = load_captions()
    
    # Collect available audios
    valid_files = []
    print(f"Scanning {AUDIO_DIR}...")
    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith(".wav") and filename in captions_map:
            valid_files.append(filename)
            
    print(f"Found {len(valid_files)} matching audio files for evaluation.")
    
    results_omni = []
    results_v2 = []
    
    # -----------------------------------------------------
    # Test Qwen 2.5 Omni (using existing verifiers.py)
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("Evaluating Qwen2.5-Omni-7B")
    print("="*50)
    
    omni_model = AQAScoreRewardModel(model_id="Qwen/Qwen2.5-Omni-7B", device=DEVICE)
    
    for filename in tqdm(valid_files):
        path = os.path.join(AUDIO_DIR, filename)
        caption = captions_map[filename]
        
        # Decompose the prompt
        delimiters = [' and ', ' while ', ' with ', ',', ' followed by ', ' then ']
        regex_pattern = '|'.join(map(re.escape, delimiters))
        splits = [e.strip() for e in re.split(regex_pattern, caption) if len(e.strip()) > 3]
        if not splits:
            splits = [caption]
            
        waveform, sr = torchaudio.load(path)
        # the AQAScoreRewardModel expects a batch dimension
        waveform_batched = waveform.unsqueeze(0)
        
        # 1. Evaluate Full Prompt
        score_full = omni_model(waveform_batched, caption)[0].item()
        
        # 2. Evaluate Decomposed Prompt
        split_scores = []
        for split_event in splits:
            split_scores.append(omni_model(waveform_batched, split_event)[0].item())
        score_split = sum(split_scores) / len(split_scores) if split_scores else score_full
        
        results_omni.append({
            "file": filename, 
            "caption": caption, 
            "score_full": score_full,
            "score_split": score_split,
            "splits": splits
        })
        
    avg_omni_full = sum(x["score_full"] for x in results_omni) / len(results_omni)
    avg_omni_split = sum(x["score_split"] for x in results_omni) / len(results_omni)
    
    # Write detailed output
    with open("results_omni_split.json", "w") as f:
        json.dump({"average_score_full": avg_omni_full, "average_score_split": avg_omni_split, "details": results_omni}, f, indent=4)
        
    print(f"\n✅ Qwen2.5-Omni Average P(yes) | FULL: {avg_omni_full:.4f}  | SPLIT: {avg_omni_split:.4f}")
    
    # Release VRAM
    del omni_model
    torch.cuda.empty_cache()
    
    # -----------------------------------------------------
    # Test OLD Qwen2-Audio (using standalone class)
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("Evaluating Qwen2-Audio-7B-Instruct (Baseline)")
    print("="*50)
    
    v2_model = OldQwen2AudioWrapper(model_id="Qwen/Qwen2-Audio-7B-Instruct", device=DEVICE)
    
    for filename in tqdm(valid_files):
        path = os.path.join(AUDIO_DIR, filename)
        caption = captions_map[filename]
        
        splits = [e.strip() for e in re.split(regex_pattern, caption) if len(e.strip()) > 3]
        if not splits:
            splits = [caption]
            
        waveform, sr = torchaudio.load(path)
        
        # 1. Evaluate Full Prompt
        question_full = f'Does this audio contain the sound events described by the text: "{caption}"? Answer yes or no.'
        score_full = v2_model(waveform, question_full, sr=sr)
        
        # 2. Evaluate Decomposed Prompt
        split_scores = []
        for split_event in splits:
            question_split = f'Does this audio contain the sound events described by the text: "{split_event}"? Answer yes or no.'
            split_scores.append(v2_model(waveform, question_split, sr=sr))
        score_split = sum(split_scores) / len(split_scores) if split_scores else score_full

        results_v2.append({
            "file": filename, 
            "caption": caption, 
            "score_full": score_full,
            "score_split": score_split,
            "splits": splits
        })
        
    avg_v2_full = sum(x["score_full"] for x in results_v2) / len(results_v2)
    avg_v2_split = sum(x["score_split"] for x in results_v2) / len(results_v2)
    
    with open("results_v2_split.json", "w") as f:
        json.dump({"average_score_full": avg_v2_full, "average_score_split": avg_v2_split, "details": results_v2}, f, indent=4)
        
    print(f"\n✅ Qwen2-Audio (Old) Average P(yes) | FULL: {avg_v2_full:.4f}  | SPLIT: {avg_v2_split:.4f}")
    
    # -----------------------------------------------------
    # Final Comparison
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("Final Benchmark Results (Ground Truth Pair Scores):")
    print(f"Qwen2-Audio-7B (Old)   | FULL: {avg_v2_full:.4f}  | SPLIT: {avg_v2_split:.4f}")
    print(f"Qwen2.5-Omni-7B (New)  | FULL: {avg_omni_full:.4f}  | SPLIT: {avg_omni_split:.4f}")
    print("="*50)
    print("Detailed scores saved to 'results_omni_split.json' and 'results_v2_split.json'.")

if __name__ == "__main__":
    main()
