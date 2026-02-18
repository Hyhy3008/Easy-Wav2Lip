import torch
import cv2
import numpy as np
import os
from easy_functions import load_model
from batch_face import RetinaFace
# Đã xóa import enhance

class Wav2LipModelWrapper:
    def __init__(self, checkpoint_path, device='cuda', face_det_batch_size=16, 
                 debug=False):
        self.device = device
        self.debug = debug
        
        print(f"[Wrapper] Loading Wav2Lip model from {checkpoint_path}...")
        self.model = load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("[Wrapper] Loading Face Detector (RetinaFace)...")
        gpu_id = 0 if device == 'cuda' else -1
        self.face_detector = RetinaFace(gpu_id=gpu_id)

        self._warmup()

    def _warmup(self):
        print("[Wrapper] Warming up GPU context...")
        try:
            if self.device == 'cuda':
                torch.cuda.synchronize()
                print("[Wrapper] Warmup complete.")
        except Exception as e:
            print(f"[Wrapper] Warmup skipped: {e}")

    @torch.no_grad()
    def infer_batch(self, img_batch, mel_batch):
        """
        Chạy Wav2Lip inference.
        Input: 
            img_batch: Tensor (B, C, H, W) normalized [-1, 1]
            mel_batch: Tensor (B, 1, 80, 16)
        Output: Numpy array (B, H, W, C) uint8
        """
        # Wav2Lip Forward
        pred = self.model(img_batch, mel_batch)
        
        # Post-process
        pred = pred.cpu().numpy()
        # Transpose (B, C, H, W) -> (B, H, W, C)
        if pred.shape[1] == 3:
            pred = np.transpose(pred, (0, 2, 3, 1))
            
        # Denormalize về 0-255
        pred = (pred + 1) * 127.5
        pred = pred.astype(np.uint8)
        
        return pred
