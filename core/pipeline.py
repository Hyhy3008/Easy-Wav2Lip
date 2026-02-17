import queue
import threading
import time
import cv2
import torch
import numpy as np
import os
import sys
from tqdm import tqdm

# Thêm thư mục gốc vào sys.path để import module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio
from core.model_wrapper import Wav2LipModelWrapper

class Wav2LipPipeline:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Khởi tạo Model Wrapper
        self.model_wrapper = Wav2LipModelWrapper(
            checkpoint_path=args.checkpoint_path,
            device=self.device,
            face_det_batch_size=args.face_det_batch_size,
            enhance=args.enhance,
            segmentation_path=args.segmentation_path
        )
        
        # Các Queue (Hàng đợi) giao tiếp giữa các luồng
        self.frame_queue = queue.Queue(maxsize=500)
        self.infer_queue = queue.Queue(maxsize=100)
        self.write_queue = queue.Queue(maxsize=500)
        
        self.stop_event = threading.Event()
        
        # Metadata video
        self.video_fps = 25.0
        self.video_size = (640, 480)
        self.total_frames = 0

    def run(self):
        print(f"[Pipeline] Bắt đầu (Enhance: {self.args.enhance})...")
        start_time = time.time()
        
        # 1. Chuẩn bị Audio (Mel-spectrogram)
        if not os.path.isfile(self.args.audio):
            raise FileNotFoundError(f"Audio file not found: {self.args.audio}")
            
        print("[Pipeline] Xử lý âm thanh...")
        wav = audio.load_wav(self.args.audio, 16000)
        mel = audio.melspectrogram(wav)
        
        # Sửa lỗi: Đảm bảo mel có shape (80, T) -> squeeze bỏ chiều batch nếu có
        if len(mel.shape) == 3:
            mel = mel.squeeze(0)
        
        # Tính toán mel chunks tương ứng với FPS video
        mel_idx_multiplier = 80. / self.args.fps 
        mel_chunks = []
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > mel.shape[1]:
                # Padding cuối cùng nếu thiếu
                chunk = np.zeros((80, 16))
                remaining = mel.shape[1] - start_idx
                if remaining > 0:
                    chunk[:, :remaining] = mel[:, start_idx:]
                mel_chunks.append(chunk)
                break
            mel_chunks.append(mel[:, start_idx : start_idx + 16])
            i += 1

        # 2. Khởi động các Worker Threads
        threads = []
        
        # Thread 1: Đọc video
        t_read = threading.Thread(target=self._worker_video_reader, args=(len(mel_chunks),))
        threads.append(t_read)
        
        # Thread 2: Chuẩn bị dữ liệu (Face Detect + Crop)
        t_prep = threading.Thread(target=self._worker_data_prep, args=(mel_chunks,))
        threads.append(t_prep)
        
        # Thread 3: Inference (Wav2Lip + Enhance)
        t_infer = threading.Thread(target=self._worker_inference)
        threads.append(t_infer)
        
        # Thread 4: Ghi video
        t_write = threading.Thread(target=self._worker_video_writer)
        threads.append(t_write)
        
        # Bắt đầu tất cả các luồng
        for t in threads:
            t.start()
            
        # Chờ hoàn thành
        for t in threads:
            t.join()
            
        end_time = time.time()
        print(f"[Pipeline] Hoàn tất! Thời gian: {end_time - start_time:.2f}s")

    def _worker_video_reader(self, total_len):
        """Đọc frame từ video và đẩy vào frame_queue"""
        video_stream = cv2.VideoCapture(self.args.face)
        self.total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = video_stream.get(cv2.CAP_PROP_FPS)
        self.video_size = (int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                           int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        frame_idx = 0
        while not self.stop_event.is_set():
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
            
            self.frame_queue.put((frame_idx, frame))
            frame_idx += 1
            
        self.frame_queue.put(None) # Tín hiệu kết thúc
        video_stream.release()

    def _worker_data_prep(self, mel_chunks):
        """Xử lý Face Detection và chuẩn bị dữ liệu cho Model"""
        from batch_face import RetinaFace
        
        gpu_id = 0 if self.device == 'cuda' else -1
        face_detector = RetinaFace(gpu_id=gpu_id)
        
        while not self.stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=1)
                if item is None:
                    self.infer_queue.put(None) # Kết thúc
                    break
                
                idx, frame = item
                
                # Resize nếu cần
                if self.args.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//self.args.resize_factor, frame.shape[0]//self.args.resize_factor))

                # Detect Face
                faces = face_detector.detect(frame)
                if len(faces) == 0:
                    continue # Bỏ qua frame không có mặt
                
                face_box = faces[0]
                x1, y1, x2, y2 = [int(v) for v in face_box[:4]]
                
                # Padding
                try:
                    pad = self.args.pads
                    y1 = max(0, y1 - pad[0])
                    y2 = min(frame.shape[0], y2 + pad[1])
                    x1 = max(0, x1 - pad[2])
                    x2 = min(frame.shape[1], x2 + pad[3])
                except:
                    pass

                # Crop & Resize face về 96x96 (input Wav2Lip)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                face_crop_96 = cv2.resize(face_crop, (96, 96))
                
                # Lấy mel chunk tương ứng với frame index
                if idx >= len(mel_chunks):
                    continue
                mel_chunk = mel_chunks[idx]
                
                # Đẩy sang queue inference
                self.infer_queue.put({
                    'idx': idx,
                    'face_96': face_crop_96,
                    'mel': mel_chunk,
                    'original_frame': frame,
                    'box': [x1, y1, x2, y2]
                })

            except queue.Empty:
                continue

    def _worker_inference(self):
        """Chạy Model Wav2Lip và Enhance"""
        batch_data = []
        batch_size = self.args.wav2lip_batch_size
        
        while not self.stop_event.is_set():
            try:
                item = self.infer_queue.get(timeout=1)
                
                if item is None:
                    # Xử lý nốt batch còn sót
                    if batch_data:
                        self._process_batch(batch_data)
                    self.write_queue.put(None) # Kết thúc Writer
                    break
                
                batch_data.append(item)
                
                if len(batch_data) >= batch_size:
                    self._process_batch(batch_data)
                    batch_data = []
                    
            except queue.Empty:
                continue

    def _process_batch(self, batch_data):
        """Xử lý batch: Inference -> Enhance -> Paste Back"""
        # 1. Chuẩn bị Tensor
        faces_96 = [d['face_96'] for d in batch_data]
        mels = [d['mel'] for d in batch_data]
        
        # Chuẩn bị Image Tensor
        img_batch = np.array(faces_96)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2)) # B, H, W, C -> B, C, H, W
        img_tensor = torch.FloatTensor(img_batch).to(self.device)
        img_tensor = (img_tensor / 255.0) * 2 - 1 # Normalize [-1, 1]
        
        # Chuẩn bị Mel Tensor
        mel_batch = np.array(mels)
        # Shape hiện tại: (B, 80, 16). Model cần (B, 1, 80, 16)
        mel_tensor = torch.FloatTensor(mel_batch).unsqueeze(1).to(self.device)
        
        # 2. Wav2Lip Inference
        gen_faces = self.model_wrapper.infer_batch(img_tensor, mel_tensor)
        
        # 3. Xử lý hậu kỳ (Post-process) từng ảnh
        for i, gen_face in enumerate(gen_faces):
            original_frame = batch_data[i]['original_frame']
            x1, y1, x2, y2 = batch_data[i]['box']
            
            # Resize mặt sinh ra về kích thước box gốc
            box_h, box_w = y2 - y1, x2 - x1
            
            # Kiểm tra kích thước hợp lệ
            if box_h <= 0 or box_w <= 0:
                continue
                
            gen_face_resized = cv2.resize(gen_face, (box_w, box_h))
            
            # Enhance (nếu bật)
            if self.args.enhance:
                gen_face_resized = self.model_wrapper.enhance_face(gen_face_resized)
            
            # Paste Back vào ảnh gốc
            if y1 >= 0 and y2 <= original_frame.shape[0] and x1 >= 0 and x2 <= original_frame.shape[1]:
                original_frame[y1:y2, x1:x2] = gen_face_resized
            
            # Gửi sang Writer
            self.write_queue.put({
                'idx': batch_data[i]['idx'],
                'frame': original_frame
            })

    def _worker_video_writer(self):
        """Ghi video kết quả"""
        out_path = self.args.outfile
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = self.video_size
        out = cv2.VideoWriter(out_path, fourcc, self.video_fps, (width, height))
        
        frame_buffer = {}
        next_idx = 0
        pbar = tqdm(total=self.total_frames, desc="Writing Video")
        
        while not self.stop_event.is_set():
            try:
                item = self.write_queue.get(timeout=1)
                if item is None:
                    # Ghi nốt buffer còn sót
                    while next_idx in frame_buffer:
                        out.write(frame_buffer[next_idx])
                        del frame_buffer[next_idx]
                        next_idx += 1
                    break
                
                idx = item['idx']
                frame = item['frame']
                
                # Lưu vào buffer để sắp xếp lại thứ tự
                frame_buffer[idx] = frame
                
                # Ghi liên tục nếu có frame đúng thứ tự
                while next_idx in frame_buffer:
                    out.write(frame_buffer[next_idx])
                    del frame_buffer[next_idx]
                    next_idx += 1
                    pbar.update(1)

            except queue.Empty:
                continue
        
        out.release()
        pbar.close()
        print(f"[Pipeline] Video saved to: {out_path}")
