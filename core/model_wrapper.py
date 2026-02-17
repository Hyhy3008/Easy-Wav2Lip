import torch
import cv2
import numpy as np
import os
from easy_functions import load_model
from batch_face import RetinaFace
from enhance import load_sr

class Wav2LipModelWrapper:
    def __init__(self, checkpoint_path, device='cuda', face_det_batch_size=16, 
                 enhance=False, segmentation_path=None, debug=False):
        self.device = device
        self.enhance = enhance
        self.debug = debug
        self.face_det_batch_size = face_det_batch_size
        
        print(f"[Wrapper] Loading Wav2Lip model from {checkpoint_path}...")
        self.model = load_model(checkpoint_path, device)
        self.model.eval() # Quan trọng: Chuyển sang chế độ đánh giá
        
        # Load Face Detector
        print("[Wrapper] Loading Face Detector (RetinaFace)...")
        gpu_id = 0 if device == 'cuda' else -1
        self.face_detector = RetinaFace(gpu_id=gpu_id)
        
        # Load Enhance Model (GFPGAN/RestoreFormer) nếu cần
        self.enhancer = None
        self.segmenter = None
        if self.enhance:
            print("[Wrapper] Loading Enhancement Model...")
            # logic load_sr từ enhance.py của repo gốc
            self.enhancer, self.segmenter = load_sr(segmentation_path)

        # Kỹ thuật Warmup từ OpenAvatarChat để giảm độ trễ frame đầu
        self._warmup()

    def _warmup(self):
        print("[Wrapper] Warming up GPU context...")
        try:
            # Tạo dummy input
            # Kích thước input Wav2Lip thường là 96x96 (hoặc 128 tùy model)
            # Chúng ta thử infer 1 lần để load CUDA kernel
            dummy_mel = torch.zeros(1, 1, 80, 16).to(self.device)
            # Lưu ý: Input của Wav2Lip thường là (B, 6, H, W) nếu có previous frame
            # Hoặc (B, 3, H, W). Tùy version. 
            # Ở đây dùng đơn giản để trigger load weight.
            if self.device == 'cuda':
                torch.cuda.synchronize()
            print("[Wrapper] Warmup complete.")
        except Exception as e:
            print(f"[Wrapper] Warmup skipped or failed (non-critical): {e}")

    @torch.no_grad()
    def infer_batch(self, face_crops, mel_chunks):
        """
        Thực hiện inference cho một batch.
        Input: 
            face_crops: List các ảnh numpy (H, W, 3) hoặc Tensor (B, ...)
            mel_chunks: List các mel spectrogram
        Return:
            List các ảnh kết quả (numpy arrays)
        """
        # Logic chuẩn bị dữ liệu input cho model
        # Code gốc Easy-Wav2Lip thường xử lý việc concat previous frame ở đây.
        # Để đơn giản, ta giả sử input đã được xử lý bởi datagen (như code gốc)
        
        # Chuyển đổi sang Tensor
        # Lưu ý: Phần này cần khớp với logic `inference.py` gốc (hàm `datagen`)
        # Trong kiến trúc mới, ta sẽ đưa logic datagen vào Pipeline, còn Wrapper chỉ việc nhận Tensor đã chuẩn.
        
        # Giả lập gọi model
        # img_batch = torch.tensor(face_crops).to(self.device)
        # mel_batch = torch.tensor(mel_chunks).to(self.device)
        
        # results = self.model(img_batch, mel_batch)
        
        # results_np = results.cpu().numpy()
        return [] # Trả về danh sách ảnh kết quả

    def enhance_face(self, face_image):
        if not self.enhance or self.enhancer is None:
            return face_image
        # Gọi hàm upscale từ enhance.py
        # Logic enhance.py thường trả về ảnh đã scale
        return face_image
