"""
Wav2LipEngine - Always-On Architecture
=======================================
Model được nạp 1 lần duy nhất vào VRAM.
Gọi process() nhiều lần mà không cần reload.
"""

# ============================================================================
# IMPORTS
# ============================================================================
print("\rLoading torch       ", end="")
import torch

print("\rLoading numpy       ", end="")
import numpy as np

print("\rLoading cv2         ", end="")
import cv2

print("\rLoading os          ", end="")
import os

print("\rLoading subprocess  ", end="")
import subprocess

print("\rLoading pickle      ", end="")
import pickle

print("\rLoading math        ", end="")
import math

print("\rLoading tqdm        ", end="")
from tqdm import tqdm

print("\rLoading audio       ", end="")
import audio

print("\rLoading RetinaFace  ", end="")
from batch_face import RetinaFace

print("\rLoading warnings    ", end="")
import warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)

print("\rLoading GFPGAN      ", end="")
from gfpgan import GFPGANer

print("\rLoading load_model  ", end="")
from easy_functions import load_model

print("\rImports loaded!     ")

# ============================================================================
# LOAD PREDICTORS (Dlib) - Chỉ load 1 lần khi import module
# ============================================================================
print("Loading face predictors...")
with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
    predictor = pickle.load(f)

with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
    mouth_detector = pickle.load(f)

print("Predictors loaded!")

# ============================================================================
# CONSTANTS
# ============================================================================
MEL_STEP_SIZE = 16
IMG_SIZE = 96


# ============================================================================
# HELPER FUNCTIONS (Giữ nguyên các hàm đã tối ưu)
# ============================================================================

def match_color(target, source):
    """
    Ép màu ảnh source theo màu ảnh target.
    Dùng không gian màu Lab để giữ độ tự nhiên.
    """
    if target.shape != source.shape:
        source = cv2.resize(source, (target.shape[1], target.shape[0]))
    
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)

    for i in range(3):
        target_mean = target_lab[:, :, i].mean()
        target_std = target_lab[:, :, i].std()
        source_mean = source_lab[:, :, i].mean()
        source_std = source_lab[:, :, i].std()

        if source_std > 1e-5:
            source_lab[:, :, i] = (source_lab[:, :, i] - source_mean) * (target_std / source_std) + target_mean
        else:
            source_lab[:, :, i] = target_mean

    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
    
    return result


def create_mask(img, original_img, mask_dilation=150, mask_feathering=151, cached_mask=None):
    """
    Tạo mask vùng miệng bằng Numpy (No-PIL).
    Trả về (blended_image, mask) để có thể cache mask.
    """
    if cached_mask is not None:
        mask_to_use = cv2.resize(cached_mask, (img.shape[1], img.shape[0]))
    else:
        faces = mouth_detector(img)
        if len(faces) == 0:
            return img, None
        
        face = faces[0]
        shape = predictor(img, face)
        
        mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
        x, y, w, h = cv2.boundingRect(mouth_points)
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)
        
        kernel_size = int(max(w, h) * mask_dilation / 100)
        if kernel_size < 1:
            kernel_size = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel)
        
        if mask_feathering > 0:
            blur = int(max(w, h) * mask_feathering / 100)
            if blur % 2 == 0:
                blur += 1
            if blur < 1:
                blur = 1
            mask_to_use = cv2.GaussianBlur(dilated_mask, (blur, blur), 0)
        else:
            mask_to_use = dilated_mask
    
    # Blend using Numpy (No-PIL)
    mask_3ch = cv2.cvtColor(mask_to_use, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out = (img.astype(np.float32) * mask_3ch + original_img.astype(np.float32) * (1 - mask_3ch))
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    return out, mask_to_use


def create_tracked_mask(img, original_img, mask_dilation=150, mask_feathering=151, last_mask=None):
    """
    Tạo mask với tracking miệng theo từng frame.
    """
    faces = mouth_detector(img)
    
    if len(faces) == 0:
        if last_mask is not None:
            mask_to_use = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
        else:
            return img, None
    else:
        face = faces[0]
        shape = predictor(img, face)
        
        mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
        x, y, w, h = cv2.boundingRect(mouth_points)
        
        kernel_size = int(max(w, h) * mask_dilation / 100)
        if kernel_size < 1:
            kernel_size = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)
        dilated_mask = cv2.dilate(mask, kernel)
        
        blur = int(max(w, h) * mask_feathering / 100)
        if blur % 2 == 0:
            blur += 1
        if blur < 1:
            blur = 1
        
        mask_to_use = cv2.GaussianBlur(dilated_mask, (blur, blur), 0)
    
    mask_3ch = cv2.cvtColor(mask_to_use, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out = (img.astype(np.float32) * mask_3ch + original_img.astype(np.float32) * (1 - mask_3ch))
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    return out, mask_to_use


def get_smoothened_boxes(boxes, T=5):
    """Làm mượt bounding boxes."""
    smoothed = []
    for i in range(len(boxes)):
        start = max(0, i - T // 2)
        end = min(len(boxes), i + T // 2 + 1)
        window = boxes[start:end]
        mean_box = np.mean(window, axis=0)
        smoothed.append(np.round(mean_box).astype(np.int32))
    return np.array(smoothed)


# ============================================================================
# WAV2LIP ENGINE CLASS
# ============================================================================

class Wav2LipEngine:
    """
    Always-On Wav2Lip Engine.
    Load models 1 lần, gọi process() nhiều lần.
    """
    
    def __init__(self, gpu_id=0, checkpoint_path="checkpoints/wav2lip.pth", load_sr_model=True):
        """
        Khởi tạo Engine và nạp models vào VRAM.
        
        Args:
            gpu_id: ID của GPU (0, 1, 2...)
            checkpoint_path: Đường dẫn tới model Wav2Lip
            load_sr_model: Có nạp GFPGAN không
        """
        # Xác định device
        if torch.cuda.is_available():
            self.device = f'cuda:{gpu_id}'
            self.gpu_id = gpu_id
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            self.gpu_id = -1
        else:
            self.device = 'cpu'
            self.gpu_id = -1
            print("⚠️ Warning: No GPU detected! Inference will be VERY SLOW!")
        
        print(f"🔌 Initializing Wav2LipEngine on {self.device}...")
        
        # Load Wav2Lip model
        print(f"   Loading Wav2Lip model...")
        self.model = load_model(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"   ✅ Wav2Lip model loaded!")
        
        # Load Face Detector (RetinaFace)
        print(f"   Loading RetinaFace detector...")
        self.detector = RetinaFace(
            gpu_id=self.gpu_id,
            model_path="checkpoints/mobilenet.pth",
            network="mobilenet"
        )
        print(f"   ✅ RetinaFace loaded!")
        
        # Load Super Resolution model (GFPGAN)
        self.sr_model = None
        if load_sr_model:
            print(f"   Loading GFPGAN model...")
            self.sr_model = self._load_sr()
            print(f"   ✅ GFPGAN loaded!")
        
        print(f"✅ Wav2LipEngine Ready on {self.device}!")
    
    def _load_sr(self):
        """Load GFPGAN model."""
        sr_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return GFPGANer(
            model_path="checkpoints/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=torch.device(sr_device)
        )
    
    def _upscale(self, image):
        """Upscale image using GFPGAN."""
        if self.sr_model is None:
            return image
        try:
            _, restored_faces, _ = self.sr_model.enhance(
                image,
                has_aligned=True,
                only_center_face=False,
                paste_back=False
            )
            return restored_faces[0]
        except Exception as e:
            print(f"⚠️ Upscale error: {e}")
            return image
    
    def _face_rect_generator(self, images, batch_size=16):
        """Generator để detect faces theo batch."""
        num_batches = math.ceil(len(images) / batch_size)
        prev_ret = None
        
        for i in range(num_batches):
            batch = images[i * batch_size : (i + 1) * batch_size]
            all_faces = self.detector(batch)
            
            for faces in all_faces:
                if faces:
                    box, landmarks, score = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret
    
    def _face_detect(self, images, cache_file=None, pads=(0, 10, 0, 0), 
                     batch_size=16, smooth=True):
        """
        Detect faces và cache kết quả.
        
        Args:
            images: List các frame
            cache_file: Đường dẫn file cache (.pkl)
            pads: Padding (top, bottom, left, right)
            batch_size: Batch size cho face detection
            smooth: Có làm mượt bounding boxes không
        
        Returns:
            List of [face_crop, (y1, y2, x1, x2)]
        """
        # Check cache
        if cache_file and os.path.exists(cache_file):
            print(f"📦 Loading face cache: {cache_file}")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            print(f"   ✅ Loaded {len(cached)} cached entries")
            return cached
        
        print(f"🔍 Detecting faces for {len(images)} frames...")
        results = []
        pady1, pady2, padx1, padx2 = pads
        
        for image, rect in tqdm(
            zip(images, self._face_rect_generator(images, batch_size)),
            total=len(images),
            desc="Detecting faces",
            ncols=100
        ):
            if rect is None:
                raise ValueError("Face not detected! Ensure video contains a face.")
            
            y1 = int(max(0, rect[1] - pady1))
            y2 = int(min(image.shape[0], rect[3] + pady2))
            x1 = int(max(0, rect[0] - padx1))
            x2 = int(min(image.shape[1], rect[2] + padx2))
            results.append([x1, y1, x2, y2])
        
        boxes = np.array(results, dtype=np.int32)
        
        # Smooth boxes
        if smooth:
            boxes = get_smoothened_boxes(boxes, T=5)
        
        # Create final results with face crops
        final_results = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_crop = images[i][y1:y2, x1:x2].copy()
            final_results.append([face_crop, (y1, y2, x1, x2)])
        
        # Save cache
        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump(final_results, f)
            print(f"   💾 Saved cache: {len(final_results)} entries -> {cache_file}")
        
        return final_results
    
    def _load_video(self, video_path, resize_height=None, crop=None):
        """
        Load video frames.
        
        Args:
            video_path: Đường dẫn video hoặc ảnh
            resize_height: Chiều cao resize (None = giữ nguyên)
            crop: Tuple (y1, y2, x1, x2) để crop
        
        Returns:
            (frames, fps)
        """
        # Check if image
        if video_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            frame = cv2.imread(video_path)
            if frame is None:
                raise ValueError(f"Cannot read image: {video_path}")
            return [frame], 25.0
        
        # Load video
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Resize
            if resize_height and resize_height > 0:
                aspect = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (int(resize_height * aspect), resize_height))
            
            # Crop
            if crop:
                y1, y2, x1, x2 = crop
                if x2 == -1:
                    x2 = frame.shape[1]
                if y2 == -1:
                    y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
            
            frames.append(frame)
        
        video.release()
        print(f"📹 Loaded {len(frames)} frames at {fps:.2f} FPS")
        return frames, fps
    
    def _load_audio(self, audio_path, fps):
        """
        Load audio và tạo mel spectrogram chunks.
        
        Returns:
            List of mel chunks
        """
        # Convert to wav if needed
        if not audio_path.endswith('.wav'):
            print("🔊 Converting audio to WAV...")
            wav_path = "temp/temp_audio.wav"
            subprocess.check_call([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", audio_path, wav_path
            ])
            audio_path = wav_path
        
        print("🔊 Analyzing audio...")
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError("Mel spectrogram contains NaN!")
        
        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps
        i = 0
        
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + MEL_STEP_SIZE > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + MEL_STEP_SIZE])
            i += 1
        
        print(f"   ✅ Created {len(mel_chunks)} mel chunks")
        return mel_chunks
    
    def _prepare_batch(self, faces, mels):
        """
        Chuẩn bị batch cho inference.
        
        Returns:
            (img_batch, mel_batch) as torch tensors on device
        """
        img_batch = np.asarray(faces)
        mel_batch = np.asarray(mels)
        
        # Mask bottom half of faces
        img_masked = img_batch.copy()
        img_masked[:, IMG_SIZE // 2:] = 0
        
        # Concatenate masked and original
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        
        # Reshape mel
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        
        # Convert to torch tensors
        img_tensor = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
        mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
        
        return img_tensor, mel_tensor
    
    def process(self, video_path, audio_path, output_path, settings=None):
        """
        Xử lý lip-sync cho video.
        
        Args:
            video_path: Đường dẫn video input
            audio_path: Đường dẫn audio input
            output_path: Đường dẫn video output
            settings: Dictionary chứa các tham số:
                - quality: "Fast" | "Improved" | "Enhanced"
                - sharpen_amount: float (1.0 - 3.0)
                - enable_color_match: bool
                - mask_dilation: int (50 - 200)
                - mask_feathering: int (50 - 200)
                - mouth_tracking: bool
                - batch_size: int
                - resize_height: int (0 = no resize)
                - crop: tuple (y1, y2, x1, x2)
                - pads: tuple (top, bottom, left, right)
                - cache_file: str (path to cache .pkl)
                - smooth_boxes: bool
                - debug_mask: bool
                - preview_only: bool (chỉ xử lý 1 frame)
        
        Returns:
            output_path nếu thành công
        """
        # Default settings
        s = {
            'quality': 'Enhanced',
            'sharpen_amount': 1.5,
            'enable_color_match': True,
            'mask_dilation': 150,
            'mask_feathering': 151,
            'mouth_tracking': False,
            'batch_size': 1,
            'resize_height': 0,
            'crop': (0, -1, 0, -1),
            'pads': (0, 10, 0, 0),
            'cache_file': None,
            'smooth_boxes': True,
            'debug_mask': False,
            'preview_only': False,
        }
        
        # Update with user settings
        if settings:
            s.update(settings)
        
        print("=" * 60)
        print("🎬 WAV2LIP PROCESSING")
        print("=" * 60)
        print(f"   Quality: {s['quality']}")
        print(f"   Sharpen: {s['sharpen_amount']}")
        print(f"   Color Match: {s['enable_color_match']}")
        print(f"   Mask Dilation: {s['mask_dilation']}")
        print(f"   Mask Feathering: {s['mask_feathering']}")
        print("=" * 60)
        
        # ================================================================
        # STEP 1: LOAD VIDEO
        # ================================================================
        resize_h = s['resize_height'] if s['resize_height'] > 0 else None
        full_frames, fps = self._load_video(video_path, resize_h, s['crop'])
        
        if s['preview_only']:
            full_frames = [full_frames[0]]
        
        # ================================================================
        # STEP 2: LOAD AUDIO
        # ================================================================
        mel_chunks = self._load_audio(audio_path, fps)
        
        if s['preview_only']:
            mel_chunks = [mel_chunks[0]]
        
        # ================================================================
        # STEP 3: SYNC VIDEO & AUDIO LENGTH
        # ================================================================
        print(f"📊 Sync Info:")
        print(f"   Video: {len(full_frames)} frames")
        print(f"   Audio: {len(mel_chunks)} mel chunks")
        
        if len(mel_chunks) > len(full_frames):
            print("   ⚠️ Audio longer than video - Looping video...")
            original_len = len(full_frames)
            looped_frames = []
            while len(looped_frames) < len(mel_chunks):
                looped_frames.extend(full_frames)
            full_frames = looped_frames[:len(mel_chunks)]
            print(f"   ✅ Looped: {original_len} -> {len(full_frames)} frames")
        else:
            full_frames = full_frames[:len(mel_chunks)]
            print(f"   ✅ Trimmed to: {len(full_frames)} frames")
        
        # ================================================================
        # STEP 4: FACE DETECTION (với Cache)
        # ================================================================
        face_det_results = self._face_detect(
            full_frames,
            cache_file=s['cache_file'],
            pads=s['pads'],
            batch_size=s['batch_size'],
            smooth=s['smooth_boxes']
        )
        
        # ================================================================
        # SYNC FIX: Đảm bảo frames và cache khớp nhau
        # ================================================================
        num_synced = min(len(full_frames), len(face_det_results), len(mel_chunks))
        full_frames = full_frames[:num_synced]
        face_det_results = face_det_results[:num_synced]
        mel_chunks = mel_chunks[:num_synced]
        
        print(f"   🔄 Synced to: {num_synced} frames")
        
        # ================================================================
        # STEP 5: SETUP OUTPUT VIDEO
        # ================================================================
        frame_h, frame_w = full_frames[0].shape[:2]
        temp_video = "temp/result_temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video, fourcc, fps, (frame_w, frame_h))
        
        # ================================================================
        # STEP 6: INFERENCE LOOP
        # ================================================================
        print(f"🚀 Processing {num_synced} frames...")
        
        # Cache cho mask (nếu không tracking)
        cached_mask = None
        last_tracked_mask = None
        
        # Process theo batch
        batch_size = s['batch_size']
        
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        for idx, mel in enumerate(tqdm(mel_chunks, desc="Processing", ncols=100)):
            frame = full_frames[idx].copy()
            face, coords = face_det_results[idx]
            
            # Resize face to model input size
            face_resized = cv2.resize(face.copy(), (IMG_SIZE, IMG_SIZE))
            
            img_batch.append(face_resized)
            mel_batch.append(mel)
            frame_batch.append(frame)
            coords_batch.append(coords)
            
            # Process khi đủ batch
            if len(img_batch) >= batch_size or idx == len(mel_chunks) - 1:
                # Prepare batch
                img_tensor, mel_tensor = self._prepare_batch(img_batch, mel_batch)
                
                # Inference
                with torch.no_grad():
                    pred = self.model(mel_tensor, img_tensor)
                
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                
                # Process each prediction
                for p, f, c in zip(pred, frame_batch, coords_batch):
                    y1, y2, x1, x2 = c
                    target_h = y2 - y1
                    target_w = x2 - x1
                    
                    # Debug mask mode
                    if s['debug_mask']:
                        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    
                    # Original face crop
                    cf = f[y1:y2, x1:x2].copy()
                    
                    # ============================================
                    # QUALITY MODES
                    # ============================================
                    
                    if s['quality'] == "Enhanced":
                        # STEP 1: GFPGAN Upscale
                        p = self._upscale(p.astype(np.uint8))
                        
                        # STEP 2: Resize với LANCZOS4
                        p = cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                        
                        # STEP 3: Kernel Sharpening
                        k = s['sharpen_amount']
                        sharpen_kernel = np.array([
                            [0, -1, 0],
                            [-1, k + 3.5, -1],
                            [0, -1, 0]
                        ]) / (k + 0.5)
                        p = cv2.filter2D(p, -1, sharpen_kernel)
                        
                        # STEP 4: Color Match
                        if s['enable_color_match']:
                            p = match_color(cf, p)
                        
                        # STEP 5: Mask Blending
                        if s['mouth_tracking']:
                            p, last_tracked_mask = create_tracked_mask(
                                p, cf, 
                                s['mask_dilation'], 
                                s['mask_feathering'],
                                last_tracked_mask
                            )
                        else:
                            p, cached_mask = create_mask(
                                p, cf,
                                s['mask_dilation'],
                                s['mask_feathering'],
                                cached_mask
                            )
                    
                    elif s['quality'] == "Improved":
                        # Chỉ resize + mask
                        p = cv2.resize(p.astype(np.uint8), (target_w, target_h))
                        
                        if s['mouth_tracking']:
                            p, last_tracked_mask = create_tracked_mask(
                                p, cf,
                                s['mask_dilation'],
                                s['mask_feathering'],
                                last_tracked_mask
                            )
                        else:
                            p, cached_mask = create_mask(
                                p, cf,
                                s['mask_dilation'],
                                s['mask_feathering'],
                                cached_mask
                            )
                    
                    else:  # Fast
                        p = cv2.resize(p.astype(np.uint8), (target_w, target_h))
                    
                    # Đảm bảo kích thước đúng
                    if p.shape[0] != target_h or p.shape[1] != target_w:
                        p = cv2.resize(p, (target_w, target_h))
                    
                    # Paste vào frame
                    f[y1:y2, x1:x2] = p
                    
                    # Write frame
                    if not s['preview_only']:
                        out.write(f)
                
                # Clear batches
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        # Release video writer
        out.release()
        
        # ================================================================
        # STEP 7: MERGE AUDIO
        # ================================================================
        if s['preview_only']:
            # Chỉ save 1 frame preview
            cv2.imwrite(output_path.replace('.mp4', '.jpg'), f)
            print(f"✅ Preview saved: {output_path.replace('.mp4', '.jpg')}")
            return output_path.replace('.mp4', '.jpg')
        
        print("🔊 Merging audio...")
        
        # Ensure temp video exists
        if not os.path.exists(temp_video):
            raise FileNotFoundError(f"Temp video not found: {temp_video}")
        
        # Convert audio to wav if needed
        audio_wav = audio_path
        if not audio_path.endswith('.wav'):
            audio_wav = "temp/temp_audio.wav"
            if not os.path.exists(audio_wav):
                subprocess.check_call([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", audio_path, audio_wav
                ])
        
        # Merge with FFmpeg
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp_video,
            "-i", audio_wav,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Cleanup temp files
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        print(f"✅ Done! Output: {output_path}")
        print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return output_path
    
    def create_cache(self, video_path, cache_file, target_frames=0, 
                     resize_height=0, pads=(0, 10, 0, 0)):
        """
        Tạo cache face detection cho video.
        
        Args:
            video_path: Đường dẫn video
            cache_file: Đường dẫn output cache (.pkl)
            target_frames: Số frames cần tạo (0 = tất cả)
            resize_height: Chiều cao resize
            pads: Padding cho face detection
        
        Returns:
            Số entries đã tạo
        """
        print("=" * 60)
        print("📦 CACHE CREATION MODE")
        print("=" * 60)
        
        # Load video
        resize_h = resize_height if resize_height > 0 else None
        frames, fps = self._load_video(video_path, resize_h)
        
        print(f"   Original: {len(frames)} frames")
        print(f"   Target: {target_frames if target_frames > 0 else 'All'}")
        
        # Loop nếu cần
        if target_frames > 0 and target_frames > len(frames):
            print(f"   Looping to reach {target_frames} frames...")
            looped = []
            while len(looped) < target_frames:
                looped.extend(frames)
            frames = looped[:target_frames]
        elif target_frames > 0:
            frames = frames[:target_frames]
        
        # Delete old cache
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"   Deleted old cache")
        
        # Detect faces
        results = self._face_detect(frames, cache_file=cache_file, pads=pads)
        
        print("=" * 60)
        print(f"✅ Cache created: {cache_file}")
        print(f"   Total entries: {len(results)}")
        print("=" * 60)
        
        return len(results)


# ============================================================================
# STANDALONE USAGE (Backward compatibility)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wav2Lip Inference")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--face", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--outfile", type=str, default="results/result.mp4")
    parser.add_argument("--quality", type=str, default="Enhanced")
    parser.add_argument("--sharpen_amount", type=float, default=1.5)
    parser.add_argument("--enable_color_match", action="store_true")
    parser.add_argument("--mask_dilation", type=int, default=150)
    parser.add_argument("--mask_feathering", type=int, default=151)
    parser.add_argument("--mouth_tracking", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_file", type=str, default=None)
    parser.add_argument("--only_detect", action="store_true")
    parser.add_argument("--target_frames", type=int, default=0)
    parser.add_argument("--pads", nargs="+", type=int, default=[0, 10, 0, 0])
    parser.add_argument("--out_height", type=int, default=0)
    parser.add_argument("--preview_settings", action="store_true")
    
    args = parser.parse_args()
    
    # Create engine
    engine = Wav2LipEngine(
        gpu_id=0,
        checkpoint_path=args.checkpoint_path,
        load_sr_model=(args.quality == "Enhanced")
    )
    
    if args.only_detect:
        # Cache creation mode
        engine.create_cache(
            video_path=args.face,
            cache_file=args.cache_file or "master_cache.pkl",
            target_frames=args.target_frames,
            resize_height=args.out_height,
            pads=tuple(args.pads)
        )
    else:
        # Normal processing
        settings = {
            'quality': args.quality,
            'sharpen_amount': args.sharpen_amount,
            'enable_color_match': args.enable_color_match,
            'mask_dilation': args.mask_dilation,
            'mask_feathering': args.mask_feathering,
            'mouth_tracking': args.mouth_tracking,
            'batch_size': args.batch_size,
            'cache_file': args.cache_file,
            'resize_height': args.out_height,
            'pads': tuple(args.pads),
            'preview_only': args.preview_settings,
        }
        
        engine.process(
            video_path=args.face,
            audio_path=args.audio,
            output_path=args.outfile,
            settings=settings
        )
