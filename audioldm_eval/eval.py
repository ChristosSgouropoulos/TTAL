import torch
from audioldm_eval import EvaluationHelperParallel
import torch.multiprocessing as mp

generation_result_path = "/mnt/storage/outputs/inference_scaling_zero_order_ode_linear_6_candidates_4_search_radius_095_clap/audio"
target_audio_path = "/mnt/storage/audiocaps/reference_886_stereo"

if __name__ == '__main__':    
    evaluator = EvaluationHelperParallel(16000, 2, backbone="cnn14") # 2 denotes number of GPUs
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path,
        limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
    )
