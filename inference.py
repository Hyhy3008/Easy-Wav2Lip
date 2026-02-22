# ============================================================================
# Cell 1: INFERENCE.PY - CODEFORMER VERSION
# ============================================================================
# Thay thế GFPGAN bằng CodeFormer cho chất lượng tốt hơn
# ============================================================================

import os

WORK_DIR = '/kaggle/working/Easy-Wav2Lip'
os.makedirs(WORK_DIR, exist_ok=True)

INFERENCE_CODE = '''"""
Wav2LipEngine - CodeFormer Version
===================================
- CodeFormer thay thế GFPGAN
- Fidelity control (0.0-1.0)
- Chất lượng tốt hơn, giữ face identity
"""

import torch
import numpy as np
import cv2
import os
import subprocess
import pickle
import math
import uuid
from tqdm import tqdm
import warnings

# Import local modules
import audio
from batch_face import RetinaFace
from easy_functions import load_model

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONSTANTS
# ============================================================================
MEL_STEP_SIZE = 16
IMG_SIZE = 96

# ============================================================================
# LOAD PREDICTORS
# ============================================================================
predictor = None
mouth_detector = None

try:
    with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
        predictor = pickle.load(f)
    with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
        mouth_detector = pickle.load(f)
    print("✅ Predictors loaded!")
except Exception as e:
    print(f"⚠️ Predictors not loaded: {e}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_smoothened_boxes(boxes, T=5):
    """Smooth bounding boxes"""
    smoothed = []
    for i in range(len(boxes)):
        start = max(0, i - T // 2)
        end = min(len(boxes), i + T // 2 + 1)
        window = boxes[start:end]
        mean_box = np.mean(window, axis=0)
        smoothed.append(np.round(mean_box).astype(np.int32))
    return np.array(smoothed)


def match_color(target, source):
    """Match color from target to source"""
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
    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)


def create_mask(img, original_img, mask_dilation=150, mask_feathering=151, cached_mask=None):
    """Create mouth mask"""
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


# ============================================================================
# CODEFORMER ENHANCER CLASS
# ============================================================================

class CodeFormerEnhancer:
    """
    CodeFormer Face Enhancement
    Tối ưu cho tốc độ cao
    """
    
    def __init__(self, device='cuda:0', fidelity_weight=0.7):
        """
        Args:
            device: GPU device
            fidelity_weight: 0.0 = quality, 1.0 = fidelity (recommend 0.5-0.7)
        """
        self.device = device
        self.fidelity_weight = fidelity_weight
        self.model = None
        self.face_helper = None
        
        self._load_model()
    
    def _load_model(self):
        """Load CodeFormer model"""
        try:
            from codeformer import CodeFormer
            from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
            
            # Load model
            self.model = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=['32', '64', '128', '256']
            ).to(self.device)
            
            # Load weights
            ckpt_path = 'checkpoints/codeformer.pth'
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['params_ema'])
                self.model.eval()
                print(f"✅ CodeFormer loaded (fidelity={self.fidelity_weight})")
            else:
                print(f"⚠️ CodeFormer weights not found: {ckpt_path}")
                self.model = None
                
        except ImportError:
            print("⚠️ CodeFormer not installed, trying alternative...")
            self._load_model_alternative()
    
    def _load_model_alternative(self):
        """Alternative loading method using basicsr"""
        try:
            from basicsr.archs.codeformer_arch import CodeFormer
            
            self.model = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=['32', '64', '128', '256']
            ).to(self.device)
            
            ckpt_path = 'checkpoints/codeformer.pth'
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['params_ema'])
                self.model.eval()
                print(f"✅ CodeFormer (basicsr) loaded")
            else:
                self.model = None
                
        except Exception as e:
            print(f"⚠️ CodeFormer not available: {e}")
            self.model = None
    
    @torch.no_grad()
    def enhance(self, img, has_aligned=True):
        """
        Enhance face image
        
        Args:
            img: BGR image (numpy array)
            has_aligned: True if face is already aligned/cropped
        
        Returns:
            Enhanced BGR image
        """
        if self.model is None:
            return img
        
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [-1, 1]
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Resize to 512x512 (CodeFormer input size)
            original_size = (img.shape[1], img.shape[0])
            img_tensor = torch.nn.functional.interpolate(
                img_tensor, size=(512, 512), mode='bilinear', align_corners=False
            )
            
            # Inference
            output = self.model(img_tensor, w=self.fidelity_weight, adain=True)[0]
            
            # Resize back
            output = torch.nn.functional.interpolate(
                output.unsqueeze(0), size=(original_size[1], original_size[0]), 
                mode='bilinear', align_corners=False
            ).squeeze(0)
            
            # Convert back to numpy BGR
            output = (output * 0.5 + 0.5).clamp(0, 1)
            output = (output.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            return output
            
        except Exception as e:
            print(f"⚠️ CodeFormer enhance error: {e}")
            return img


# ============================================================================
# WAV2LIP ENGINE CLASS - CODEFORMER VERSION
# ============================================================================

class Wav2LipEngine:
    """
    Wav2Lip Engine với CodeFormer Enhancement
    """
    
    def __init__(self, gpu_id=0, checkpoint_path="checkpoints/wav2lip.pth", 
                 load_sr_model=True, fidelity_weight=0.7):
        """
        Args:
            gpu_id: GPU ID
            checkpoint_path: Wav2Lip model path
            load_sr_model: Load CodeFormer or not
            fidelity_weight: CodeFormer fidelity (0.0-1.0)
        """
        self.engine_id = gpu_id
        self.fidelity_weight = fidelity_weight
        
        # Device setup
        if torch.cuda.is_available():
            self.device = f'cuda:{gpu_id}'
            self.gpu_id = gpu_id
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            self.gpu_id = -1
        else:
            self.device = 'cpu'
            self.gpu_id = -1
            print("⚠️ No GPU detected!")
        
        print(f"🔌 Engine {gpu_id} initializing on {self.device}...")
        
        # Load Wav2Lip model
        print(f"   Loading Wav2Lip...")
        self.model = load_model(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"   ✅ Wav2Lip loaded")
        
        # Load Face Detector
        print(f"   Loading RetinaFace...")
        self.detector = RetinaFace(
            gpu_id=self.gpu_id,
            model_path="checkpoints/mobilenet.pth",
            network="mobilenet"
        )
        print(f"   ✅ RetinaFace loaded")
        
        # Load CodeFormer (instead of GFPGAN)
        self.sr_model = None
        if load_sr_model:
            print(f"   Loading CodeFormer (fidelity={fidelity_weight})...")
            self.sr_model = self._load_sr()
            if self.sr_model is not None:
                print(f"   ✅ CodeFormer loaded!")
            else:
                print(f"   ⚠️ CodeFormer not available, using basic resize")
        
        print(f"✅ Engine {gpu_id} ready!")
    
    def _load_sr(self):
        """Load CodeFormer model"""
        try:
            return CodeFormerEnhancer(
                device=self.device,
                fidelity_weight=self.fidelity_weight
            )
        except Exception as e:
            print(f"⚠️ CodeFormer load error: {e}")
            return None
    
    def _upscale(self, image):
        """Upscale image using CodeFormer"""
        if self.sr_model is None:
            return image
        try:
            return self.sr_model.enhance(image, has_aligned=True)
        except Exception as e:
            print(f"⚠️ Upscale error: {e}")
            return image
    
    def _face_rect_generator(self, images, batch_size=16):
        """Generate face rectangles"""
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
        """Detect faces"""
        if cache_file and os.path.exists(cache_file):
            print(f"📦 Loading cache: {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                print(f"   ✅ Loaded {len(cached)} entries")
                return cached
            except Exception as e:
                print(f"   ⚠️ Cache failed: {e}")
        
        print(f"🔍 Detecting faces for {len(images)} frames...")
        results = []
        pady1, pady2, padx1, padx2 = pads
        
        for image, rect in tqdm(
            zip(images, self._face_rect_generator(images, batch_size)),
            total=len(images),
            desc="Face detection"
        ):
            if rect is None:
                raise ValueError("Face not detected!")
            
            y1 = int(max(0, rect[1] - pady1))
            y2 = int(min(image.shape[0], rect[3] + pady2))
            x1 = int(max(0, rect[0] - padx1))
            x2 = int(min(image.shape[1], rect[2] + padx2))
            results.append([x1, y1, x2, y2])
        
        boxes = np.array(results, dtype=np.int32)
        
        if smooth:
            boxes = get_smoothened_boxes(boxes, T=5)
        
        final_results = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_crop = images[i][y1:y2, x1:x2].copy()
            final_results.append([face_crop, (y1, y2, x1, x2)])
        
        if cache_file:
            try:
                os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(final_results, f)
                print(f"   💾 Saved cache: {cache_file}")
            except Exception as e:
                print(f"   ⚠️ Cache save failed: {e}")
        
        return final_results
    
    def _load_video(self, video_path, resize_height=None, crop=None):
        """Load video frames"""
        if video_path.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            frame = cv2.imread(video_path)
            if frame is None:
                raise ValueError(f"Cannot read image: {video_path}")
            return [frame], 25.0
        
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = video.get(cv2.CAP_PROP_FPS) or 25.0
        
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            if resize_height and resize_height > 0:
                aspect = frame.shape[1] / frame.shape[0]
                new_width = int(resize_height * aspect)
                frame = cv2.resize(frame, (new_width, resize_height))
            
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
            raise ValueError(f"No frames loaded: {video_path}")
        
        print(f"📹 Loaded {len(frames)} frames @ {fps:.2f} FPS")
        return frames, fps
    
    def _load_audio(self, audio_path, fps):
        """Load audio and create mel chunks"""
        unique_id = uuid.uuid4().hex[:8]
        temp_wav = f"temp/audio_{self.engine_id}_{unique_id}.wav"
        os.makedirs("temp", exist_ok=True)
        
        need_cleanup = False
        
        if not audio_path.endswith('.wav'):
            print("🔊 Converting audio...")
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", audio_path, "-ac", "1", "-ar", "16000", temp_wav
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio_path = temp_wav
            need_cleanup = True
        
        print("🔊 Processing audio...")
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError("Mel contains NaN!")
        
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
        
        if need_cleanup and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass
        
        print(f"   ✅ {len(mel_chunks)} mel chunks")
        return mel_chunks
    
    def _prepare_batch(self, faces, mels):
        """Prepare batch for inference"""
        img_batch = np.asarray(faces)
        mel_batch = np.asarray(mels)
        
        img_masked = img_batch.copy()
        img_masked[:, IMG_SIZE // 2:] = 0
        
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        
        img_tensor = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
        mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
        
        return img_tensor, mel_tensor
    
    def process(self, video_path, audio_path, output_path, settings=None):
        """Main processing function"""
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
        
        if settings:
            s.update(settings)
        
        print("=" * 60)
        print(f"🎬 WAV2LIP PROCESSING [GPU {self.engine_id}]")
        print(f"   Quality: {s['quality']} (CodeFormer)")
        print("=" * 60)
        
        try:
            # Load video
            resize_h = s['resize_height'] if s['resize_height'] > 0 else None
            full_frames, fps = self._load_video(video_path, resize_h, s['crop'])
            
            if s['preview_only']:
                full_frames = [full_frames[0]]
            
            # Load audio
            mel_chunks = self._load_audio(audio_path, fps)
            
            if s['preview_only']:
                mel_chunks = [mel_chunks[0]]
            
            # Sync
            print(f"📊 Video: {len(full_frames)} | Audio: {len(mel_chunks)}")
            
            if len(mel_chunks) > len(full_frames):
                print("   Looping video...")
                looped = []
                while len(looped) < len(mel_chunks):
                    looped.extend(full_frames)
                full_frames = looped[:len(mel_chunks)]
            else:
                full_frames = full_frames[:len(mel_chunks)]
            
            # Face detection
            face_det_results = self._face_detect(
                full_frames,
                cache_file=s['cache_file'],
                pads=s['pads'],
                batch_size=s['batch_size'],
                smooth=s['smooth_boxes']
            )
            
            # Sync fix
            num_synced = min(len(full_frames), len(face_det_results), len(mel_chunks))
            full_frames = full_frames[:num_synced]
            face_det_results = face_det_results[:num_synced]
            mel_chunks = mel_chunks[:num_synced]
            
            print(f"   Synced: {num_synced} frames")
            
            if num_synced == 0:
                raise ValueError("No frames to process!")
            
            # Setup output
            frame_h, frame_w = full_frames[0].shape[:2]
            
            unique_id = uuid.uuid4().hex[:8]
            output_dir = os.path.dirname(output_path) or "temp"
            os.makedirs(output_dir, exist_ok=True)
            
            temp_video = os.path.join(output_dir, f"temp_{self.engine_id}_{unique_id}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_video, fourcc, fps, (frame_w, frame_h))
            
            # Processing loop
            print(f"🚀 Processing {num_synced} frames...")
            
            cached_mask = None
            batch_size = min(s['batch_size'], 128)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
            
            for idx, mel in enumerate(tqdm(mel_chunks, desc="Processing")):
                frame = full_frames[idx].copy()
                face, coords = face_det_results[idx]
                
                face_resized = cv2.resize(face.copy(), (IMG_SIZE, IMG_SIZE))
                
                img_batch.append(face_resized)
                mel_batch.append(mel)
                frame_batch.append(frame)
                coords_batch.append(coords)
                
                if len(img_batch) >= batch_size or idx == len(mel_chunks) - 1:
                    img_tensor, mel_tensor = self._prepare_batch(img_batch, mel_batch)
                    
                    with torch.no_grad():
                        pred = self.model(mel_tensor, img_tensor)
                    
                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                    
                    for p, f, c in zip(pred, frame_batch, coords_batch):
                        y1, y2, x1, x2 = c
                        target_h = y2 - y1
                        target_w = x2 - x1
                        
                        if target_h <= 0 or target_w <= 0:
                            out.write(f)
                            continue
                        
                        if s['debug_mask']:
                            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                        
                        cf = f[y1:y2, x1:x2].copy()
                        
                        # Quality processing with CodeFormer
                        if s['quality'] == "Enhanced" and self.sr_model is not None:
                            # CodeFormer enhancement
                            p = self._upscale(p.astype(np.uint8))
                            p = cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                            
                            # Sharpen
                            if s['sharpen_amount'] > 0:
                                k = s['sharpen_amount']
                                sharpen_kernel = np.array([
                                    [0, -1, 0],
                                    [-1, k + 3.5, -1],
                                    [0, -1, 0]
                                ]) / (k + 0.5)
                                p = cv2.filter2D(p, -1, sharpen_kernel)
                                p = np.clip(p, 0, 255).astype(np.uint8)
                            
                            # Color match
                            if s['enable_color_match']:
                                p = match_color(cf, p)
                            
                            # Mask
                            p, cached_mask = create_mask(
                                p, cf, s['mask_dilation'], s['mask_feathering'], cached_mask
                            )
                        
                        elif s['quality'] == "Improved":
                            p = cv2.resize(p.astype(np.uint8), (target_w, target_h))
                            p, cached_mask = create_mask(
                                p, cf, s['mask_dilation'], s['mask_feathering'], cached_mask
                            )
                        
                        else:
                            p = cv2.resize(p.astype(np.uint8), (target_w, target_h))
                        
                        if p.shape[0] != target_h or p.shape[1] != target_w:
                            p = cv2.resize(p, (target_w, target_h))
                        
                        f[y1:y2, x1:x2] = p
                        
                        if not s['preview_only']:
                            out.write(f)
                    
                    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
            
            out.release()
            
            # Preview mode
            if s['preview_only']:
                preview_path = output_path.replace('.mp4', '_preview.jpg')
                cv2.imwrite(preview_path, f)
                print(f"✅ Preview: {preview_path}")
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                return preview_path
            
            # Merge audio
            print("🔊 Merging audio...")
            
            if not os.path.exists(temp_video):
                raise FileNotFoundError(f"Temp video not found: {temp_video}")
            
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
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
            
            if os.path.exists(temp_video):
                try:
                    os.remove(temp_video)
                except:
                    pass
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"✅ Done! {output_path} ({file_size:.1f} MB)")
                return output_path
            else:
                raise FileNotFoundError(f"Output not created: {output_path}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_cache(self, video_path, cache_file, target_frames=0, 
                     resize_height=0, pads=(0, 10, 0, 0)):
        """Create face detection cache"""
        print("=" * 60)
        print("📦 CACHE CREATION")
        print("=" * 60)
        
        resize_h = resize_height if resize_height > 0 else None
        frames, fps = self._load_video(video_path, resize_h)
        
        print(f"   Original: {len(frames)} frames")
        
        if target_frames > 0 and target_frames > len(frames):
            print(f"   Looping to {target_frames}...")
            looped = []
            while len(looped) < target_frames:
                looped.extend(frames)
            frames = looped[:target_frames]
        elif target_frames > 0:
            frames = frames[:target_frames]
        
        if os.path.exists(cache_file):
            os.remove(cache_file)
        
        results = self._face_detect(frames, cache_file=cache_file, pads=pads)
        
        print("=" * 60)
        print(f"✅ Cache: {cache_file}")
        print(f"   Entries: {len(results)}")
        print("=" * 60)
        
        return len(results)


# Export
__all__ = ['Wav2LipEngine', 'CodeFormerEnhancer']
'''

# Ghi file
output_path = f'{WORK_DIR}/inference.py'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(INFERENCE_CODE)

print("=" * 60)
print("✅ INFERENCE.PY (CodeFormer) CREATED!")
print("=" * 60)
print(f"Path: {output_path}")
print(f"Size: {os.path.getsize(output_path) / 1024:.1f} KB")

# Verify
with open(output_path, 'r') as f:
    content = f.read()
    if 'CodeFormerEnhancer' in content and 'Wav2LipEngine' in content:
        print("✅ CodeFormer integration verified!")
    else:
        print("❌ Verification failed!")

print("=" * 60)
