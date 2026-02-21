"""
Wav2LipEngine - Always-On Architecture (Parallel GPU Safe)
===========================================================
Version: V18 Parallel Safe
- Unique temp paths per process (no conflict)
- Multi-GPU parallel processing support
- Model loaded once, reused multiple times
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

print("\rLoading uuid        ", end="")
import uuid

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
# LOAD PREDICTORS (Dlib) - Load 1 lần khi import module
# ============================================================================
print("Loading face predictors...")

predictor = None
mouth_detector = None

try:
    with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
        predictor = pickle.load(f)
    with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
        mouth_detector = pickle.load(f)
    print("✅ Predictors loaded!")
except Exception as e:
    print(f"⚠️ Warning: Could not load predictors: {e}")
    print("   Masking features may not work properly.")


# ============================================================================
# CONSTANTS
# ============================================================================
MEL_STEP_SIZE = 16
IMG_SIZE = 96


# ============================================================================
# HELPER FUNCTIONS - Export cho Cell 2 sử dụng
# ============================================================================

def get_video_info(video_path):
    """
    Lấy thông tin video: fps, total_frames, width, height
    
    Args:
        video_path: Đường dẫn video
    
    Returns:
        tuple: (fps, total_frames, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, width, height


def get_audio_duration(audio_path):
    """
    Lấy độ dài audio bằng ffprobe (giây)
    
    Args:
        audio_path: Đường dẫn file audio
    
    Returns:
        float: Độ dài tính bằng giây
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except:
        return 0.0


def match_color(target, source):
    """
    Ép màu ảnh source theo màu ảnh target.
    Dùng không gian màu Lab để giữ độ tự nhiên.
    
    Args:
        target: Ảnh gốc (màu chuẩn)
        source: Ảnh AI (cần điều chỉnh màu)
    
    Returns:
        Ảnh source đã được điều chỉnh màu
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
    
    Args:
        img: Ảnh AI (đã xử lý)
        original_img: Ảnh gốc
        mask_dilation: Độ mở rộng mask (pixels)
        mask_feathering: Độ mờ viền mask
        cached_mask: Mask đã cache (nếu có)
    
    Returns:
        tuple: (blended_image, mask)
    """
    global predictor, mouth_detector
    
    if cached_mask is not None:
        mask_to_use = cv2.resize(cached_mask, (img.shape[1], img.shape[0]))
    else:
        if mouth_detector is None or predictor is None:
            return img, None
        
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
        kernel_size = max(1, kernel_size)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel)
        
        if mask_feathering > 0:
            blur = int(max(w, h) * mask_feathering / 100)
            blur = blur if blur % 2 == 1 else blur + 1
            blur = max(1, blur)
            mask_to_use = cv2.GaussianBlur(dilated_mask, (blur, blur), 0)
        else:
            mask_to_use = dilated_mask
    
    mask_3ch = cv2.cvtColor(mask_to_use, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out = (img.astype(np.float32) * mask_3ch + original_img.astype(np.float32) * (1 - mask_3ch))
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    return out, mask_to_use


def create_tracked_mask(img, original_img, mask_dilation=150, mask_feathering=151, last_mask=None):
    """
    Tạo mask với tracking miệng theo từng frame.
    
    Args:
        img: Ảnh AI
        original_img: Ảnh gốc
        mask_dilation: Độ mở rộng mask
        mask_feathering: Độ mờ viền
        last_mask: Mask frame trước (fallback)
    
    Returns:
        tuple: (blended_image, new_mask)
    """
    global predictor, mouth_detector
    
    if mouth_detector is None or predictor is None:
        return img, last_mask
    
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
        kernel_size = max(1, kernel_size)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)
        dilated_mask = cv2.dilate(mask, kernel)
        
        blur = int(max(w, h) * mask_feathering / 100)
        blur = blur if blur % 2 == 1 else blur + 1
        blur = max(1, blur)
        
        mask_to_use = cv2.GaussianBlur(dilated_mask, (blur, blur), 0)
    
    mask_3ch = cv2.cvtColor(mask_to_use, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out = (img.astype(np.float32) * mask_3ch + original_img.astype(np.float32) * (1 - mask_3ch))
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    return out, mask_to_use


def get_smoothened_boxes(boxes, T=5):
    """
    Làm mượt bounding boxes qua các frame.
    
    Args:
        boxes: Array của bounding boxes
        T: Window size cho smoothing
    
    Returns:
        Array đã smooth
    """
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
    Always-On Wav2Lip Engine - Parallel GPU Safe.
    Load models 1 lần, gọi process() nhiều lần.
    Mỗi process() sử dụng temp path riêng biệt để tránh conflict.
    """
    
    def __init__(self, gpu_id=0, checkpoint_path="checkpoints/wav2lip.pth", load_sr_model=True):
        """
        Khởi tạo Engine và nạp models vào VRAM.
        
        Args:
            gpu_id: ID của GPU (0, 1, 2...)
            checkpoint_path: Đường dẫn tới model Wav2Lip
            load_sr_model: Có nạp GFPGAN không
        """
        # Lưu GPU ID để tạo unique temp paths
        self.engine_id = gpu_id
        
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
        print(f"   Loading Wav2Lip model from {checkpoint_path}...")
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
        
        print(f"✅ Wav2LipEngine {gpu_id} Ready on {self.device}!")
    
    def _load_sr(self):
        """Load GFPGAN model lên đúng GPU."""
        sr_device = f'cuda:{self.gpu_id}' if torch.cuda.is_available() and self.gpu_id >= 0 else 'cpu'
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
            if restored_faces and len(restored_faces) > 0:
                return restored_faces[0]
            return image
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
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                print(f"   ✅ Loaded {len(cached)} cached entries")
                return cached
            except Exception as e:
                print(f"   ⚠️ Cache load failed: {e}, detecting faces...")
        
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
            try:
                cache_dir = os.path.dirname(cache_file)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(final_results, f)
                print(f"   💾 Saved cache: {len(final_results)} entries -> {cache_file}")
            except Exception as e:
                print(f"   ⚠️ Cache save failed: {e}")
        
        return final_results
    
    def _load_video(self, video_path, resize_height=None, crop=None):
        """
        Load video frames.
        
        Args:
            video_path: Đường dẫn video hoặc ảnh
            resize_height: Chiều cao resize (None/0 = giữ nguyên)
            crop: Tuple (y1, y2, x1, x2) để crop
        
        Returns:
            (frames, fps)
        """
        # Check if image
        if video_path.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            frame = cv2.imread(video_path)
            if frame is None:
                raise ValueError(f"Cannot read image: {video_path}")
            return [frame], 25.0
        
        # Load video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Resize
            if resize_height and resize_height > 0:
                aspect = frame.shape[1] / frame.shape[0]
                new_width = int(resize_height * aspect)
                frame = cv2.resize(frame, (new_width, resize_height))
            
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
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from video: {video_path}")
        
        print(f"📹 Loaded {len(frames)} frames at {fps:.2f} FPS")
        return frames, fps
    
    def _load_audio(self, audio_path, fps):
        """
        Load audio và tạo mel spectrogram chunks.
        
        Args:
            audio_path: Đường dẫn audio
            fps: FPS của video
        
        Returns:
            List of mel chunks
        """
        # ✅ UNIQUE TEMP PATH PER ENGINE - Tránh conflict khi parallel
        unique_id = uuid.uuid4().hex[:8]
        temp_wav = f"temp/audio_{self.engine_id}_{unique_id}.wav"
        os.makedirs("temp", exist_ok=True)
        
        need_cleanup = False
        
        # Convert to wav if needed
        if not audio_path.endswith('.wav'):
            print("🔊 Converting audio to WAV...")
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", audio_path, "-ac", "1", "-ar", "16000", temp_wav
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio_path = temp_wav
            need_cleanup = True
        
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
        
        # Cleanup temp wav
        if need_cleanup and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass
        
        print(f"   ✅ Created {len(mel_chunks)} mel chunks")
        return mel_chunks
    
    def _prepare_batch(self, faces, mels):
        """
        Chuẩn bị batch cho inference.
        
        Returns:
            (img_tensor, mel_tensor) on device
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
        ✅ PARALLEL SAFE: Mỗi process dùng temp path riêng
        
        Args:
            video_path: Đường dẫn video input
            audio_path: Đường dẫn audio input
            output_path: Đường dẫn video output
            settings: Dictionary chứa các tham số:
                - quality: "Fast" | "Improved" | "Enhanced"
                - sharpen_amount: float (0.0 - 3.0)
                - enable_color_match: bool
                - mask_dilation: int (50 - 300)
                - mask_feathering: int (0 - 150)
                - mouth_tracking: bool
                - batch_size: int
                - resize_height: int (0 = no resize)
                - crop: tuple (y1, y2, x1, x2)
                - pads: tuple (top, bottom, left, right)
                - cache_file: str (path to cache .pkl)
                - smooth_boxes: bool
                - debug_mask: bool
                - preview_only: bool
        
        Returns:
            output_path nếu thành công, None nếu thất bại
        """
        # Default settings
        s = {
            'quality': 'Enhanced',
            'sharpen_amount': 0,
            'enable_color_match': False,
            'mask_dilation': 150,
            'mask_feathering': 75,
            'mouth_tracking': False,
            'batch_size': 16,
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
        print(f"🎬 WAV2LIP PROCESSING [GPU {self.engine_id}]")
        print("=" * 60)
        print(f"   Quality: {s['quality']}")
        print(f"   Sharpen: {s['sharpen_amount']}")
        print(f"   Color Match: {s['enable_color_match']}")
        print(f"   Mask: dilation={s['mask_dilation']}, feather={s['mask_feathering']}")
        print(f"   Mouth Tracking: {s['mouth_tracking']}")
        print(f"   Batch Size: {s['batch_size']}")
        print("=" * 60)
        
        try:
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
            # SYNC FIX: Đảm bảo frames, cache, mel khớp nhau
            # ================================================================
            num_synced = min(len(full_frames), len(face_det_results), len(mel_chunks))
            full_frames = full_frames[:num_synced]
            face_det_results = face_det_results[:num_synced]
            mel_chunks = mel_chunks[:num_synced]
            
            print(f"   🔄 Synced to: {num_synced} frames")
            
            if num_synced == 0:
                raise ValueError("No frames to process after sync!")
            
            # ================================================================
            # STEP 5: SETUP OUTPUT VIDEO - ✅ UNIQUE TEMP PATH
            # ================================================================
            frame_h, frame_w = full_frames[0].shape[:2]
            
            # ✅ UNIQUE TEMP FILE PER PROCESS - Tránh conflict khi parallel
            unique_id = uuid.uuid4().hex[:8]
            output_dir = os.path.dirname(output_path)
            if output_dir:
                temp_dir = output_dir
            else:
                temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_video = os.path.join(temp_dir, f"temp_{self.engine_id}_{unique_id}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_video, fourcc, fps, (frame_w, frame_h))
            
            # ================================================================
            # STEP 6: INFERENCE LOOP
            # ================================================================
            print(f"🚀 Processing {num_synced} frames...")
            
            # Cache cho mask
            cached_mask = None
            last_tracked_mask = None
            
            # Batch processing
            batch_size = min(s['batch_size'], 128)  # Cap batch size
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
                
                # Process khi đủ batch hoặc cuối cùng
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
                        
                        # Skip invalid regions
                        if target_h <= 0 or target_w <= 0:
                            out.write(f)
                            continue
                        
                        # Debug mask mode
                        if s['debug_mask']:
                            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                        
                        # Original face crop
                        cf = f[y1:y2, x1:x2].copy()
                        
                        # ====================================================
                        # QUALITY MODES
                        # ====================================================
                        
                        if s['quality'] == "Enhanced" and self.sr_model is not None:
                            # STEP 1: GFPGAN Upscale
                            p = self._upscale(p.astype(np.uint8))
                            
                            # STEP 2: Resize với LANCZOS4
                            p = cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                            
                            # STEP 3: Kernel Sharpening
                            if s['sharpen_amount'] > 0:
                                k = s['sharpen_amount']
                                sharpen_kernel = np.array([
                                    [0, -1, 0],
                                    [-1, k + 3.5, -1],
                                    [0, -1, 0]
                                ]) / (k + 0.5)
                                p = cv2.filter2D(p, -1, sharpen_kernel)
                                p = np.clip(p, 0, 255).astype(np.uint8)
                            
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
                            # Resize + mask only
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
                        
                        # Ensure correct size
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
            # STEP 7: PREVIEW MODE
            # ================================================================
            if s['preview_only']:
                preview_path = output_path.replace('.mp4', '_preview.jpg')
                cv2.imwrite(preview_path, f)
                print(f"✅ Preview saved: {preview_path}")
                # Cleanup temp
                if os.path.exists(temp_video):
                    try:
                        os.remove(temp_video)
                    except:
                        pass
                return preview_path
            
            # ================================================================
            # STEP 8: MERGE AUDIO
            # ================================================================
            print("🔊 Merging audio...")
            
            if not os.path.exists(temp_video):
                raise FileNotFoundError(f"Temp video not found: {temp_video}")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Merge with FFmpeg
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", temp_video,
                "-i", audio_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path
            ]
            
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Cleanup temp
            if os.path.exists(temp_video):
                try:
                    os.remove(temp_video)
                except:
                    pass
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"✅ Done! Output: {output_path} ({file_size:.1f} MB)")
                return output_path
            else:
                raise FileNotFoundError(f"Output file not created: {output_path}")
        
        except Exception as e:
            print(f"❌ Error in process(): {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_cache(self, video_path, cache_file, target_frames=0, 
                     resize_height=0, pads=(0, 10, 0, 0)):
        """
        Tạo cache face detection cho video.
        
        Args:
            video_path: Đường dẫn video
            cache_file: Đường dẫn output cache (.pkl)
            target_frames: Số frames cần tạo (0 = tất cả)
            resize_height: Chiều cao resize (0 = giữ nguyên)
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
# STANDALONE USAGE (CLI Compatibility)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wav2Lip Inference - Always-On Engine")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to Wav2Lip model")
    parser.add_argument("--face", type=str, required=True, help="Video or image file")
    parser.add_argument("--audio", type=str, required=True, help="Audio file")
    parser.add_argument("--outfile", type=str, default="results/result.mp4", help="Output path")
    parser.add_argument("--quality", type=str, default="Enhanced", choices=["Fast", "Improved", "Enhanced"])
    parser.add_argument("--sharpen_amount", type=float, default=0)
    parser.add_argument("--enable_color_match", action="store_true")
    parser.add_argument("--mask_dilation", type=int, default=150)
    parser.add_argument("--mask_feathering", type=int, default=75)
    parser.add_argument("--mouth_tracking", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cache_file", type=str, default=None)
    parser.add_argument("--only_detect", action="store_true", help="Only create cache")
    parser.add_argument("--target_frames", type=int, default=0)
    parser.add_argument("--pads", nargs="+", type=int, default=[0, 10, 0, 0])
    parser.add_argument("--out_height", type=int, default=0)
    parser.add_argument("--nosmooth", action="store_true")
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
            'smooth_boxes': not args.nosmooth,
            'preview_only': args.preview_settings,
        }
        
        result = engine.process(
            video_path=args.face,
            audio_path=args.audio,
            output_path=args.outfile,
            settings=settings
        )
        
        if result:
            print(f"✅ Success: {result}")
        else:
            print("❌ Processing failed!")
