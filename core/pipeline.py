import queue
import threading
import time
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
from audio import load_mel
from easy_functions import face_detect

# Import Wrapper vừa tạo
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
        
        # Định nghĩa các Queue (Hàng đợi)
        # Queue 1: Lưu trữ frame gốc đọc từ video (Đọc -> Queue1)
        self.raw_frame_queue = queue.Queue(maxsize=200)
        
        # Queue 2: Lưu trữ dữ liệu đã xử lý (Face crop + Mel) chờ inference (Queue1 -> Detect -> Queue2)
        self.process_queue = queue.Queue(maxsize=100)
        
        # Queue 3: Lưu trữ kết quả video chờ ghi (Queue2 -> Inference -> Queue3)
        self.write_queue = queue.Queue(maxsize=200)
        
        self.stop_event = threading.Event()
        
        # Biến toàn cục để tính toán progress bar
        self.total_frames = 0
        self.processed_frames = 0

    def run(self):
        """Hàm chính để khởi động pipeline"""
        print("[Pipeline] Starting pipeline...")
        start_time = time.time()
        
        # 1. Chuẩn bị dữ liệu âm thanh (Pre-calculate MEL)
        # Đây là bước CPU heavy nhưng làm 1 lần duy nhất
        print("[Pipeline] Processing audio...")
        mel = load_mel(self.args.audio) # Hàm này có trong audio.py của repo gốc
        # Chia mel thành các chunk tương ứng với FPS
        mel_chunks = self._prepare_mel_chunks(mel)
        
        # 2. Khởi động các Worker Threads
        threads = []
        
        # Thread 1: Đọc Video
        t_reader = threading.Thread(target=self._worker_video_reader, args=(mel_chunks,))
        threads.append(t_reader)
        
        # Thread 2: Detect Face & Prep Data (CPU heavy)
        t_prep = threading.Thread(target=self._worker_data_preparation)
        threads.append(t_prep)
        
        # Thread 3: Inference (GPU heavy)
        t_infer = threading.Thread(target=self._worker_inference)
        threads.append(t_infer)
        
        # Thread 4: Ghi Video
        t_writer = threading.Thread(target=self._worker_video_writer, args=(self.args.outfile,))
        threads.append(t_writer)
        
        # Bắt đầu tất cả các thread
        for t in threads:
            t.start()
            
        # Chờ hoàn thành
        for t in threads:
            t.join()
            
        end_time = time.time()
        print(f"[Pipeline] Done! Total time: {end_time - start_time:.2f}s")

    def _worker_video_reader(self, mel_chunks):
        """Đọc frame từ video file và đẩy vào queue"""
        cap = cv2.VideoCapture(self.args.face)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Lưu metadata để writer dùng
        self.video_meta = {'fps': fps, 'width': width, 'height': height}
        
        frame_idx = 0
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Đẩy (frame_idx, frame_image) vào queue
            self.raw_frame_queue.put((frame_idx, frame))
            frame_idx += 1
            
        # Đẩy tín hiệu kết thúc (Poison Pill)
        self.raw_frame_queue.put(None)
        cap.release()

    def _worker_data_preparation(self):
        """
        Lấy frame từ raw_queue -> Detect Face -> Cắt -> Đẩy sang process_queue
        Logic này lấy từ hàm `face_detect` và `datagen` của inference.py gốc
        """
        while not self.stop_event.is_set():
            try:
                item = self.raw_frame_queue.get(timeout=1)
                if item is None:
                    # Kết thúc, đẩy tín hiệu sang bước tiếp theo
                    self.process_queue.put(None)
                    break
                
                idx, frame = item
                
                # --- LOGIC DETECT FACE (Sử dụng lại từ repo gốc) ---
                # Lưu ý: face_detect trong repo gốc trả về bounding box
                # Bạn cần implement logic cắt ảnh ở đây.
                # Ví dụ đơn giản:
                # faces = self.model_wrapper.face_detector.detect(frame)
                # if len(faces) > 0:
                #    box = faces[0] # Lấy mặt đầu tiên
                #    face_crop = crop_logic(frame, box)
                # else:
                #    face_crop = np.zeros((96, 96, 3)) # Fallback
                
                # Tạm thời dùng frame gốc làm demo (Cần sửa logic crop)
                face_crop = cv2.resize(frame, (96, 96)) 
                
                # Đẩy sang queue inference
                self.process_queue.put({
                    'idx': idx,
                    'crop': face_crop,
                    'original': frame,
                    # 'mel': mel_chunk tương ứng
                })
                
            except queue.Empty:
                continue

    def _worker_inference(self):
        """
        Gom batch từ process_queue -> Chạy AI -> Đẩy sang write_queue
        Đây là nơi áp dụng kỹ thuật Batching từ OpenAvatarChat
        """
        batch_size = self.args.wav2lip_batch_size
        
        while not self.stop_event.is_set():
            try:
                # Gom batch
                batch_data = []
                
                # Cố gắng lấy đủ batch hoặc chờ timeout
                while len(batch_data) < batch_size:
                    try:
                        item = self.process_queue.get(timeout=0.1) # Timeout ngắn để check stop event
                        if item is None:
                            # Hết dữ liệu
                            if batch_data:
                                self._process_batch(batch_data)
                            self.write_queue.put(None) # Báo hiệu hết cho writer
                            return
                        batch_data.append(item)
                    except queue.Empty:
                        break # Đã hết dữ liệu trong queue, xử lý nốt batch hiện tại
                
                if batch_data:
                    self._process_batch(batch_data)
                    
            except Exception as e:
                print(f"[Inference Worker] Error: {e}")

    def _process_batch(self, batch_data):
        """Chạy inference cho một batch"""
        # Chuẩn bị input tensor
        # crops = [d['crop'] for d in batch_data]
        # mels = [d['mel'] for d in batch_data]
        
        # Gọi Wrapper
        # results = self.model_wrapper.infer_batch(crops, mels)
        
        # Giả lập kết quả (Cần thay bằng kết quả thật)
        results = [d['crop'] for d in batch_data] 
        
        # Đẩy kết quả sang writer
        for i, res in enumerate(results):
            original_frame = batch_data[i]['original']
            idx = batch_data[i]['idx']
            
            # Logic paste face back vào original frame (Cần implement)
            # final_frame = paste_back(original_frame, res)
            final_frame = original_frame 
            
            self.write_queue.put((idx, final_frame))

    def _worker_video_writer(self, output_path):
        """Lấy frame từ write_queue -> Ghi ra file"""
        # Cấu hình VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        frame_buffer = {} # Buffer để sắp xếp lại frame nếu bị lệch thứ tự
        next_idx_to_write = 0
        
        while not self.stop_event.is_set():
            try:
                item = self.write_queue.get(timeout=1)
                if item is None:
                    break
                
                idx, frame = item
                
                # Khởi tạo VideoWriter khi có frame đầu tiên
                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, self.video_meta['fps'], (w, h))
                
                # Sắp xếp lại frame (vì multi-thread có thể làm frame 5 về trước frame 4)
                frame_buffer[idx] = frame
                
                # Ghi liên tục nếu có frame tiếp theo
                while next_idx_to_write in frame_buffer:
                    out.write(frame_buffer[next_idx_to_write])
                    del frame_buffer[next_idx_to_write]
                    next_idx_to_write += 1
                    self.processed_frames += 1
                    
            except queue.Empty:
                continue
        
        if out:
            out.release()
        print(f"[Writer] Finished writing to {output_path}")

    def _prepare_mel_chunks(self, mel):
        # Logic chia nhỏ mel spectrogram tương ứng với FPS
        # Copy logic từ hàm `datagen` trong `inference.py` gốc
        return [] # Trả về list mel chunks
