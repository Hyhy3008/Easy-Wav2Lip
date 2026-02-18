import queue
import threading
import time
import cv2
import torch
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio
from core.model_wrapper import Wav2LipModelWrapper

class Wav2LipPipeline:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # TỐI ƯU: Tăng batch size mặc định để chạy nhanh hơn
        # Nếu GPU yếu, có thể giảm xuống 16
        self.batch_size = getattr(args, 'wav2lip_batch_size', 32)
        if self.batch_size > 128: self.batch_size = 128
        
        self.model_wrapper = Wav2LipModelWrapper(
            checkpoint_path=args.checkpoint_path,
            device=self.device,
            face_det_batch_size=args.face_det_batch_size
            # Đã xóa tham số enhance
        )
        
        self.frame_queue = queue.Queue(maxsize=1000) # Tăng queue size
        self.infer_queue = queue.Queue(maxsize=200)
        self.write_queue = queue.Queue(maxsize=1000)
        
        self.stop_event = threading.Event()
        
        self.video_fps = 25.0
        self.video_size = (640, 480)
        self.total_frames = 0

    def run(self):
        print(f"[Pipeline] MAX SPEED MODE (Batch: {self.batch_size})")
        start_time = time.time()
        
        if not os.path.isfile(self.args.audio):
            raise FileNotFoundError(f"Audio file not found: {self.args.audio}")
            
        print("[Pipeline] Processing audio...")
        wav = audio.load_wav(self.args.audio, 16000)
        mel = audio.melspectrogram(wav)
        
        if len(mel.shape) == 3:
            mel = mel.squeeze(0)
        
        mel_idx_multiplier = 80. / self.args.fps 
        mel_chunks = []
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > mel.shape[1]:
                chunk = np.zeros((80, 16))
                remaining = mel.shape[1] - start_idx
                if remaining > 0:
                    chunk[:, :remaining] = mel[:, start_idx:]
                mel_chunks.append(chunk)
                break
            mel_chunks.append(mel[:, start_idx : start_idx + 16])
            i += 1

        total_frames_needed = len(mel_chunks)

        threads = []
        t_read = threading.Thread(target=self._worker_video_reader, args=(total_frames_needed,))
        t_prep = threading.Thread(target=self._worker_data_prep, args=(mel_chunks,))
        t_infer = threading.Thread(target=self._worker_inference)
        t_write = threading.Thread(target=self._worker_video_writer)
        
        for t in [t_read, t_prep, t_infer, t_write]:
            t.start()
            
        for t in [t_read, t_prep, t_infer, t_write]:
            t.join()
            
        end_time = time.time()
        print(f"[Pipeline] Done! Time: {end_time - start_time:.2f}s")

    def _worker_video_reader(self, total_frames_needed):
        video_stream = cv2.VideoCapture(self.args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_fps = self.args.fps if self.args.fps else fps
        self.video_size = (width, height)
        
        frames_list = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
            frames_list.append(frame)
        video_stream.release()
        
        num_video_frames = len(frames_list)
        if num_video_frames == 0:
            self.stop_event.set()
            return

        self.total_frames = total_frames_needed
        
        for i in range(total_frames_needed):
            if self.stop_event.is_set(): break
            frame = frames_list[i % num_video_frames]
            self.frame_queue.put((i, frame))
            
        self.frame_queue.put(None)

    def _worker_data_prep(self, mel_chunks):
        from batch_face import RetinaFace
        gpu_id = 0 if self.device == 'cuda' else -1
        face_detector = RetinaFace(gpu_id=gpu_id)
        
        batch_frames = []
        batch_indices = []
        det_batch_size = self.args.face_det_batch_size

        while not self.stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=1)
                
                if item is None:
                    if batch_frames:
                        self._process_detection_batch(batch_indices, batch_frames, face_detector, mel_chunks)
                    self.infer_queue.put(None)
                    break
                
                idx, frame = item
                
                if self.args.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//self.args.resize_factor, frame.shape[0]//self.args.resize_factor))
                
                batch_indices.append(idx)
                batch_frames.append(frame)
                
                if len(batch_frames) >= det_batch_size:
                    self._process_detection_batch(batch_indices, batch_frames, face_detector, mel_chunks)
                    batch_frames = []
                    batch_indices = []

            except queue.Empty:
                continue

    def _process_detection_batch(self, batch_indices, batch_frames, face_detector, mel_chunks):
        try:
            all_faces = face_detector.detect(batch_frames)
        except Exception as e:
            return

        for i, faces in enumerate(all_faces):
            idx = batch_indices[i]
            frame = batch_frames[i]
            
            if len(faces) == 0: continue
                
            face_box = faces[0]
            x1, y1, x2, y2 = [int(v) for v in face_box[:4]]
            
            try:
                pad = self.args.pads
                y1 = max(0, y1 - pad[0])
                y2 = min(frame.shape[0], y2 + pad[1])
                x1 = max(0, x1 - pad[2])
                x2 = min(frame.shape[1], x2 + pad[3])
            except:
                pass

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0: continue
            
            face_crop_96 = cv2.resize(face_crop, (96, 96))
            
            if idx >= len(mel_chunks): continue
            mel_chunk = mel_chunks[idx]
            
            self.infer_queue.put({
                'idx': idx,
                'face_96': face_crop_96,
                'mel': mel_chunk,
                'original_frame': frame,
                'box': [x1, y1, x2, y2]
            })

    def _worker_inference(self):
        batch_data = []
        
        while not self.stop_event.is_set():
            try:
                item = self.infer_queue.get(timeout=1)
                
                if item is None:
                    if batch_data:
                        self._process_batch(batch_data)
                    self.write_queue.put(None)
                    break
                
                batch_data.append(item)
                
                if len(batch_data) >= self.batch_size:
                    self._process_batch(batch_data)
                    batch_data = []
                    
            except queue.Empty:
                continue

    def _process_batch(self, batch_data):
        faces_96 = [d['face_96'] for d in batch_data]
        mels = [d['mel'] for d in batch_data]
        
        img_batch = np.array(faces_96)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))
        img_tensor = torch.FloatTensor(img_batch).to(self.device)
        img_tensor = (img_tensor / 255.0) * 2 - 1
        
        mel_batch = np.array(mels)
        mel_tensor = torch.FloatTensor(mel_batch).unsqueeze(1).to(self.device)
        
        gen_faces = self.model_wrapper.infer_batch(img_tensor, mel_tensor)
        
        # TỐI ƯU: Xử lý NumPy thuần túy, không có bước Enhance chậm chạp
        for i, gen_face in enumerate(gen_faces):
            original_frame = batch_data[i]['original_frame']
            x1, y1, x2, y2 = batch_data[i]['box']
            
            box_h, box_w = y2 - y1, x2 - x1
            if box_h <= 0 or box_w <= 0: continue
            
            # Resize fast
            gen_face_resized = cv2.resize(gen_face, (box_w, box_h))
            
            # Paste fast
            if y1 >= 0 and y2 <= original_frame.shape[0] and x1 >= 0 and x2 <= original_frame.shape[1]:
                original_frame[y1:y2, x1:x2] = gen_face_resized
            
            self.write_queue.put({
                'idx': batch_data[i]['idx'],
                'frame': original_frame
            })

    def _worker_video_writer(self):
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
                    while next_idx in frame_buffer:
                        out.write(frame_buffer[next_idx])
                        del frame_buffer[next_idx]
                        next_idx += 1
                    break
                
                idx = item['idx']
                frame = item['frame']
                
                frame_buffer[idx] = frame
                
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
