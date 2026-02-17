import queue
import threading
import time
import cv2
import torch
import numpy as np
import os
import sys
from tqdm import tqdm

# Thêm thư mục gốc vào sys.path để import được các module của Easy-Wav2Lip
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các hàm cần thiết từ code gốc
import audio
from easy_functions import load_model, face_detect, get_smoothened_boxes
from enhance import load_sr, upscale

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
        # Queue 1: Frame gốc đọc từ video
        self.frame_queue = queue.Queue(maxsize=500)
        # Queue 2: Dữ liệu đã chuẩn bị sẵn (Face crop, Mel, Metadata) để Inference
        self.infer_queue = queue.Queue(maxsize=100)
        # Queue 3: Kết quả video để ghi ra file
        self.write_queue = queue.Queue(maxsize=500)
        
        self.stop_event = threading.Event()
        
        # Lưu trữ thông tin video
        self.video_fps = 25.0
        self.video_size = (640, 480)
        self.total_frames = 0
        
        # Cache cho Face Detection (Tối ưu quan trọng)
        self.face_cache = {} 
        self.face_det_thread_results = {}

    def run(self):
        print("[Pipeline] Bắt đầu khởi chạy pipeline tối ưu hóa...")
        start_time = time.time()
        
        # 1. Chuẩn bị Audio (Mel-spectrogram)
        print("[Pipeline] Đang xử lý âm thanh...")
        if not os.path.isfile(self.args.audio):
            raise FileNotFoundError(f"Audio file not found: {self.args.audio}")
            
        wav = audio.load_wav(self.args.audio, 16000)
        mel = audio.melspectrogram(wav)
        # chia nhỏ mel thành các chunks tương ứng với fps
        mel_idx_multiplier = 80. / self.args.fps 
        mel_chunks = []
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - 16:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + 16])
            i += 1
            
        print(f"[Pipeline] Số lượng Mel chunks: {len(mel_chunks)}")

        # 2. Khởi động các Worker Threads
        threads = []
        
        # Thread 1: Đọc Video
        t_read = threading.Thread(target=self._worker_video_reader, args=(len(mel_chunks),))
        threads.append(t_read)
        
        # Thread 2: Xử lý Face Detection & Prep (CPU heavy)
        # Số lượng thread prep có thể tăng nếu CPU mạnh
        t_prep = threading.Thread(target=self._worker_data_prep, args=(mel_chunks,))
        threads.append(t_prep)
        
        # Thread 3: Inference (GPU heavy)
        t_infer = threading.Thread(target=self._worker_inference)
        threads.append(t_infer)
        
        # Thread 4: Ghi Video
        t_write = threading.Thread(target=self._worker_video_writer)
        threads.append(t_write)
        
        # Bắt đầu
        for t in threads:
            t.start()
            
        # Chờ hoàn thành
        for t in threads:
            t.join()
            
        end_time = time.time()
        print(f"[Pipeline] Hoàn tất! Thời gian: {end_time - start_time:.2f}s")

    def _worker_video_reader(self, total_len):
        """Đọc frame từ video và đẩy vào queue"""
        print("[Reader] Bắt đầu đọc video...")
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
            
        # Đẩy tín hiệu kết thúc
        self.frame_queue.put(None)
        video_stream.release()
        print(f"[Reader] Đọc xong {frame_idx} frames.")

    def _worker_data_prep(self, mel_chunks):
        """
        Lấy frame -> Detect Face (nếu chưa cache) -> Chuẩn bị dữ liệu cho model.
        Logic này sao chép từ hàm 'datagen' trong inference.py gốc
        """
        print("[Prep] Bắt đầu xử lý dữ liệu (Face Detect & Crop)...")
        
        # Load face detector riêng cho thread này để tránh xung đột
        from batch_face import RetinaFace
        gpu_id = 0 if self.device == 'cuda' else -1
        face_detector = RetinaFace(gpu_id=gpu_id)
        
        window = []
        while not self.stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=1)
                if item is None:
                    # Kết thúc, đẩy tín hiệu cho inference
                    self.infer_queue.put(None)
                    break
                
                idx, frame = item
                
                # --- Logic lấy window 5 frames (Từ inference.py gốc) ---
                # Wav2Lip cần 5 frame: [t-2, t-1, t, t+1, t+2] để sinh ra frame t
                # Chúng ta đơn giản hóa: dùng chính frame đó lặp lại hoặc buffer
                # Code gốc dùng image_datagen, ở đây ta làm real-time.
                
                # 1. Detect Face (Nếu chưa có)
                # Tối ưu: Chỉ detect 1 lần, các frame sau dùng lại box nếu video ít chuyển động
                # Nhưng để chuẩn nhất, ta detect theo batch nhỏ hoặc từng frame.
                # Ở đây ta detect từng frame để đảm bảo đúng logic gốc.
                
                # Lưu ý: Code gốc rescale frame về 1/resize_factor
                if self.args.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//self.args.resize_factor, frame.shape[0]//self.args.resize_factor))

                # Detect
                # face_detect function từ easy_functions hoặc tự viết lại retinaface
                # Ở đây gọi trực tiếp RetinaFace để tối ưu batch size nhỏ
                faces = face_detector.detect(frame)
                
                if len(faces) == 0:
                    # Nếu không detect ra mặt, bỏ qua hoặc dùng fallback
                    continue
                
                # Lấy mặt to nhất
                face_box = faces[0] # (x1, y1, x2, y2, score)
                # Logic padding và crop từ inference.py gốc
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

                # Crop và resize về 96x96 (size input của Wav2Lip)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                face_crop = cv2.resize(face_crop, (96, 96))
                
                # Chuẩn bị Mel chunk
                mel_idx = idx
                if mel_idx >= len(mel_chunks):
                    continue # Bỏ qua nếu video dài hơn audio
                mel_chunk = mel_chunks[mel_idx]
                
                # Đẩy vào queue inference
                # Chúng ta đẩy dạng dict để dễ xử lý
                self.infer_queue.put({
                    'idx': idx,
                    'face': face_crop,
                    'mel': mel_chunk,
                    'original_frame': frame, # Cần để paste lại
                    'box': [x1, y1, x2, y2]
                })

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Prep] Lỗi xử lý frame {idx}: {e}")

    def _worker_inference(self):
        """
        Gom batch và chạy Model.
        """
        print("[Inference] Bắt đầu GPU Inference...")
        batch_size = self.args.wav2lip_batch_size
        
        batch_data = []
        
        while not self.stop_event.is_set():
            try:
                # Lấy dữ liệu từ queue
                item = self.infer_queue.get(timeout=1)
                
                if item is None:
                    # Xử lý nốt batch còn sót
                    if batch_data:
                        self._process_batch(batch_data)
                    # Kết thúc writer
                    self.write_queue.put(None)
                    break
                
                batch_data.append(item)
                
                # Nếu đủ batch size thì xử lý
                if len(batch_data) >= batch_size:
                    self._process_batch(batch_data)
                    batch_data = []
                    
            except queue.Empty:
                continue
        
    def _process_batch(self, batch_data):
        """Chạy model cho một batch"""
        # Chuẩn bị input Tensor
        # faces: list of numpy arrays -> convert to tensor
        faces = [d['face'] for d in batch_data]
        mels = [d['mel'] for d in batch_data]
        
        # Chuyển sang Tensor (tham khảo code gốc: permute, normalize)
        img_batch = np.array(faces)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2)) # B, H, W, C -> B, C, H, W
        
        mel_batch = np.array(mels)
        
        # Chuyển sang Tensor torch
        img_tensor = torch.FloatTensor(img_batch).to(self.device)
        mel_tensor = torch.FloatTensor(mel_batch).to(self.device)
        
        # Normalize image [-1, 1]
        img_tensor = (img_tensor / 255.0) * 2 - 1
        # Mel normalize (tùy model, thường giữ nguyên hoặc scale)
        
        # ===== CHẠY MODEL =====
        with torch.no_grad():
            # Model Wav2Lip gốc thường nhận (img, mel)
            pred = self.model_wrapper.model(img_tensor, mel_tensor)
            
        # Post-process: Chuyển kết quả về numpy
        pred = pred.cpu().numpy()
        
        # pred shape: (B, C, H, W) hoặc (B, H, W, C) tùy version
        # Code gốc thường output (B, 3, 96, 96) -> transpose về (B, 96, 96, 3)
        if pred.shape[1] == 3:
            pred = np.transpose(pred, (0, 2, 3, 1))
            
        # Scale về 0-255
        pred = (pred + 1) * 127.5
        pred = pred.astype(np.uint8)

        # Gửi kết quả sang Writer
        for i, p in enumerate(pred):
            item = batch_data[i]
            # Tạo bản sao item để tránh conflict
            out_item = item.copy()
            out_item['gen_face'] = p
            self.write_queue.put(out_item)

    def _worker_video_writer(self):
        """Ghi video kết quả"""
        print("[Writer] Bắt đầu ghi video...")
        
        # Cấu hình output
        out_path = self.args.outfile
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # VideoWriter setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Lấy size từ frame gốc hoặc config
        width, height = self.video_size
        out = cv2.VideoWriter(out_path, fourcc, self.video_fps, (width, height))
        
        # Buffer để sắp xếp lại frame (do multi-thread có thể trả về không thứ tự)
        frame_buffer = {}
        next_idx = 0
        
        pbar = tqdm(total=self.total_frames, desc="Writing Video")
        
        while not self.stop_event.is_set():
            try:
                item = self.write_queue.get(timeout=1)
                if item is None:
                    # Ghi nốt các frame còn trong buffer nếu hết queue
                    while next_idx in frame_buffer:
                        out.write(frame_buffer[next_idx])
                        del frame_buffer[next_idx]
                        next_idx += 1
                    break
                
                idx = item['idx']
                original_frame = item['original_frame']
                gen_face = item['gen_face']
                box = item['box'] # x1, y1, x2, y2
                
                # === LOGIC PASTE BACK ===
                # Resize face gen về size box
                x1, y1, x2, y2 = box
                box_h, box_w = y2 - y1, x2 - x1
                
                gen_face_resized = cv2.resize(gen_face, (box_w, box_h))
                
                # Paste vào original frame
                # Lưu ý: frame gốc đã resize ở bước prep (nếu resize_factor > 1)
                # Cần resize lại về gốc nếu muốn output full HD? 
                # Để đơn giản, ta giữ nguyên size đã xử lý. 
                # Nếu muốn output full size, cần upscale box.
                
                # Paste
                # Cần check bounds để tránh lỗi
                if y1 >= 0 and y2 <= original_frame.shape[0] and x1 >= 0 and x2 <= original_frame.shape[1]:
                     original_frame[y1:y2, x1:x2] = gen_face_resized
                
                # Lưu vào buffer
                frame_buffer[idx] = original_frame
                
                # Ghi liên tục nếu có frame tiếp theo
                while next_idx in frame_buffer:
                    out.write(frame_buffer[next_idx])
                    del frame_buffer[next_idx]
                    next_idx += 1
                    pbar.update(1)

            except queue.Empty:
                continue
        
        out.release()
        pbar.close()
        print(f"[Writer] Đã ghi file tại: {out_path}")
