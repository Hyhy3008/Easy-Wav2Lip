import torch
import cv2
import numpy as np
import os
from easy_functions import load_model
from batch_face import RetinaFace
from enhance import load_sr, upscale

class Wav2LipModelWrapper:
    def __init__(self, checkpoint_path, device='cuda', face_det_batch_size=16, 
                 enhance=False, segmentation_path=None, debug=False):
        self.device = device
        self.enhance = enhance
        self.debug = debug
        self.face_det_batch_size = face_det_batch_size
        
        print(f"[Wrapper] Loading Wav2Lip model from {checkpoint_path}...")
        
        # SỬA LỖI: Hàm load_model gốc chỉ nhận 1 tham số path.
        # Ta load model về sau đó tự di chuyển sang device (GPU/CPU).
        self.model = load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("[Wrapper] Loading Face Detector (RetinaFace)...")
        gpu_id = 0 if device == 'cuda' else -1
        self.face_detector = RetinaFace(gpu_id=gpu_id)
        
        # Load Enhance Model (GFPGAN/RestoreFormer)
        self.enhancer = None
        self.segmenter = None
        if self.enhance:
            print("[Wrapper] Loading Enhancement Model...")
            try:
                self.enhancer, self.segmenter = load_sr(segmentation_path)
            except Exception as e:
                print(f"[Wrapper] Warning: Could not load Enhancement model. Error: {e}")
                self.enhance = False

        self._warmup()

    def _warmup(self):
        print("[Wrapper] Warming up GPU context...")
        try:
            if self.device == 'cuda':
                # Dummy pass to init CUDA kernels
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

    def enhance_face(self, face_image):
        """
        Nét hóa khuôn mặt (nếu bật Enhance).
        Input: Ảnh numpy (H, W, C)
        Output: Ảnh đã enhance
        """
        if not self.enhance or self.enhancer is None:
            return face_image
        
        try:
            # Hàm upscale từ enhance.py xử lý numpy array
            return upscale(face_image, self.enhancer, self.segmenter)
        except Exception as e:
            print(f"[Wrapper] Enhance error: {e}")
            return face_image
