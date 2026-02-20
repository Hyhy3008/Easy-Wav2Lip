print("\rloading torch       ", end="")
import torch

print("\rloading numpy       ", end="")
import numpy as np

print("\rloading argparse    ", end="")
import argparse

print("\rloading configparser", end="")
import configparser

print("\rloading math        ", end="")
import math

print("\rloading os          ", end="")
import os

print("\rloading subprocess  ", end="")
import subprocess

print("\rloading pickle      ", end="")
import pickle

print("\rloading cv2         ", end="")
import cv2

print("\rloading audio       ", end="")
import audio

print("\rloading RetinaFace ", end="")
from batch_face import RetinaFace

print("\rloading re          ", end="")
import re

print("\rloading partial     ", end="")
from functools import partial

print("\rloading tqdm        ", end="")
from tqdm import tqdm

print("\rloading warnings    ", end="")
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)

print("\rloading GFPGAN      ", end="")
from gfpgan import GFPGANer

print("\rloading load_model  ", end="")
from easy_functions import load_model, g_colab

print("\rimports loaded!     ")

# ========== GLOBAL CONSTANTS ==========
mel_step_size = 16

# ========== LOAD PREDICTOR & MOUTH DETECTOR ==========
with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
    predictor = pickle.load(f)

with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
    mouth_detector = pickle.load(f)

# ========== CHECK COLAB ==========
g_colab = g_colab()

if not g_colab:
    config = configparser.ConfigParser()
    config.read('config.ini')
    preview_window = config.get('OPTIONS', 'preview_window')
else:
    preview_window = "None"


# ==============================================================================
#                           WAV2LIP ENGINE CLASS
# ==============================================================================
class Wav2LipEngine:
    """
    Engine class để giữ models trong GPU VRAM.
    Khởi tạo 1 lần, xử lý nhiều video mà không cần reload models.
    """
    
    def __init__(self, gpu_id=0, checkpoint_path="checkpoints/wav2lip_gan.pth", 
                 load_sr_model=True, sr_model_name="gfpgan"):
        """
        Khởi tạo Engine và nạp sẵn models vào GPU.
        
        Args:
            gpu_id: ID của GPU (0, 1, 2, ...)
            checkpoint_path: Đường dẫn đến Wav2Lip checkpoint
            load_sr_model: Có nạp GFPGAN không (True cho Enhanced quality)
            sr_model_name: Tên model SR ("gfpgan" hoặc "RestoreFormer")
        """
        print(f"\n{'='*60}")
        print(f"🚀 Initializing Wav2LipEngine on GPU {gpu_id}")
        print(f"{'='*60}")
        
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
            print("⚠️ WARNING: No GPU detected! Running on CPU (VERY SLOW)")
        
        print(f"📍 Device: {self.device}")
        
        # ========== NẠP WAV2LIP MODEL ==========
        print(f"📦 Loading Wav2Lip model: {checkpoint_path}")
        self.model = load_model(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ Wav2Lip model loaded!")
        
        # ========== NẠP FACE DETECTOR ==========
        print(f"📦 Loading RetinaFace detector...")
        self.detector = RetinaFace(
            gpu_id=self.gpu_id if self.gpu_id >= 0 else -1,
            model_path="checkpoints/mobilenet.pth",
            network="mobilenet"
        )
        print(f"✅ RetinaFace detector loaded!")
        
        # ========== NẠP GFPGAN (TÙY CHỌN) ==========
        self.sr_model = None
        self.sr_model_name = sr_model_name
        if load_sr_model:
            print(f"📦 Loading {sr_model_name} for super resolution...")
            self.sr_model = self._load_sr_model()
            print(f"✅ {sr_model_name} loaded!")
        
        # ========== VARIABLES FOR MASK ==========
        self.last_mask = None
        self.kernel = None
        self.x = self.y = self.w = self.h = None
        
        print(f"\n{'='*60}")
        print(f"✅ GPU {gpu_id}: ALL MODELS PRE-LOADED INTO VRAM!")
        print(f"{'='*60}\n")
    
    def _load_sr_model(self):
        """Load GFPGAN model"""
        sr_device = self.device if 'cuda' in self.device else 'cpu'
        return GFPGANer(
            model_path="checkpoints/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=torch.device(sr_device)
        )
    
    def _upscale(self, image):
        """Upscale image với GFPGAN"""
        if self.sr_model is None:
            return image
        try:
            _, _, output = self.sr_model.enhance(
                image,
                has_aligned=True,
                only_center_face=False,
                paste_back=False
            )
            return output
        except Exception as e:
            print(f"⚠️ Upscale error: {e}")
            return image
    
    def _reset_mask_state(self):
        """Reset mask state cho video mới"""
        self.last_mask = None
        self.kernel = None
        self.x = self.y = self.w = self.h = None
    
    def _create_mask(self, img, original_img, mask_dilation, mask_feathering):
        """Tạo mask cho mouth region"""
        if self.last_mask is None:
            faces = mouth_detector(img)
            if len(faces) == 0:
                return img, None
            
            face = faces[0]
            shape = predictor(img, face)
            
            mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
            self.x, self.y, self.w, self.h = cv2.boundingRect(mouth_points)
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)
            
            kernel_size = int(max(self.w, self.h) * mask_dilation)
            if kernel_size < 1:
                kernel_size = 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel)
            
            if mask_feathering != 0:
                blur = int(max(self.w, self.h) * mask_feathering)
                if blur % 2 == 0:
                    blur += 1
                if blur < 1:
                    blur = 1
                self.last_mask = cv2.GaussianBlur(dilated_mask, (blur, blur), 0)
            else:
                self.last_mask = dilated_mask
        
        mask_to_use = cv2.resize(self.last_mask, (img.shape[1], img.shape[0]))
        mask_3ch = cv2.cvtColor(mask_to_use, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        out = (img.astype(float) * mask_3ch + original_img.astype(float) * (1 - mask_3ch)).astype(np.uint8)
        
        return out, self.last_mask
    
    def _create_tracked_mask(self, img, original_img, mask_dilation, mask_feathering):
        """Tạo tracked mask cho mỗi frame"""
        faces = mouth_detector(img)
        if len(faces) == 0:
            if self.last_mask is not None:
                mask_to_use = cv2.resize(self.last_mask, (img.shape[1], img.shape[0]))
            else:
                return img, None
        else:
            face = faces[0]
            shape = predictor(img, face)
            
            mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
            self.x, self.y, self.w, self.h = cv2.boundingRect(mouth_points)
            
            kernel_size = int(max(self.w, self.h) * mask_dilation)
            if kernel_size < 1:
                kernel_size = 1
            self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)
            dilated_mask = cv2.dilate(mask, self.kernel)
            
            blur = mask_feathering
            if blur % 2 == 0:
                blur += 1
            blur = int(max(self.w, self.h) * blur)
            if blur % 2 == 0:
                blur += 1
            if blur < 1:
                blur = 1
            
            mask_to_use = cv2.GaussianBlur(dilated_mask, (blur, blur), 0)
            self.last_mask = mask_to_use
        
        mask_3ch = cv2.cvtColor(mask_to_use, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        out = (img.astype(float) * mask_3ch + original_img.astype(float) * (1 - mask_3ch)).astype(np.uint8)
        
        return out, self.last_mask
    
    def _face_rect_generator(self, images, face_det_batch_size=16):
        """Generator để detect faces trong batch"""
        num_batches = math.ceil(len(images) / face_det_batch_size)
        prev_ret = None
        for i in range(num_batches):
            batch = images[i * face_det_batch_size : (i + 1) * face_det_batch_size]
            all_faces = self.detector(batch)
            for faces in all_faces:
                if faces:
                    box, landmarks, score = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret
    
    def _get_smoothened_boxes(self, boxes, T=5):
        """Smooth face boxes"""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
    
    def _face_detect(self, images, pads, nosmooth=False, 
                     face_det_batch_size=16, cache_file=None):
        """
        Detect faces trong tất cả frames.
        Có thể dùng cache nếu cache_file được chỉ định.
        """
        # Check cache
        if cache_file and os.path.exists(cache_file):
            print(f"📂 Loading face cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        
        results = []
        pady1, pady2, padx1, padx2 = pads
        
        tqdm_partial = partial(tqdm, position=0, leave=True)
        for image, rect in tqdm_partial(
            zip(images, self._face_rect_generator(images, face_det_batch_size)),
            total=len(images),
            desc="Detecting faces",
            ncols=100,
        ):
            if rect is None:
                cv2.imwrite("temp/faulty_frame.jpg", image)
                raise ValueError("Face not detected! Ensure video contains a face in all frames.")
            
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])
        
        boxes = np.array(results)
        if not nosmooth:
            boxes = self._get_smoothened_boxes(boxes, T=5)
        
        results = [
            [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]
        
        # Save cache
        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)
            print(f"💾 Face cache saved: {cache_file}")
        
        return results
    
    def _datagen(self, frames, mels, face_det_results, img_size=96, 
                 wav2lip_batch_size=1, static=False, box=None):
        """Generator tạo batches cho Wav2Lip"""
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        num_original_frames = len(frames)
        num_cache_entries = len(face_det_results)
        
        for i, m in enumerate(mels):
            if static:
                idx = 0
            else:
                idx = i % num_original_frames
            
            frame_to_save = frames[idx].copy()
            cache_idx = idx % num_cache_entries
            
            face, coords = face_det_results[cache_idx].copy()
            face = cv2.resize(face, (img_size, img_size))
            
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)
            
            if len(img_batch) >= wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
                
                img_masked = img_batch.copy()
                img_masked[:, img_size // 2:] = 0
                
                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
                mel_batch = np.reshape(
                    mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
                )
                
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            
            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0
            
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )
            
            yield img_batch, mel_batch, frame_batch, coords_batch
    
    def process(self, video_path, audio_path, output_path, settings=None):
        """
        Xử lý lip-sync cho video.
        
        Args:
            video_path: Đường dẫn video input
            audio_path: Đường dẫn audio input
            output_path: Đường dẫn video output
            settings: Dict chứa các cài đặt (xem default bên dưới)
        
        Returns:
            str: Đường dẫn video output nếu thành công
        """
        # Default settings
        default_settings = {
            'quality': 'Enhanced',      # Fast, Improved, Enhanced
            'pads': [0, 10, 0, 0],       # top, bottom, left, right
            'nosmooth': False,
            'mouth_tracking': False,
            'mask_dilation': 150,
            'mask_feathering': 151,
            'wav2lip_batch_size': 1,
            'face_det_batch_size': 16,
            'out_height': 480,
            'fullres': 3,
            'static': False,
            'fps': 25.0,
            'crop': [0, -1, 0, -1],
            'box': [-1, -1, -1, -1],
            'rotate': False,
            'debug_mask': False,
            'preview_settings': False,
            'cache_file': None,         # Đường dẫn file cache face detection
            'img_size': 96,
        }
        
        # Merge settings
        if settings:
            default_settings.update(settings)
        s = default_settings
        
        # Reset mask state
        self._reset_mask_state()
        
        print(f"\n🎬 Processing: {video_path}")
        print(f"🔊 Audio: {audio_path}")
        print(f"📤 Output: {output_path}")
        print(f"⚙️ Quality: {s['quality']}")
        
        # ========== LOAD VIDEO ==========
        if video_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            full_frames = [cv2.imread(video_path)]
            fps = s['fps']
            s['static'] = True
        else:
            video_stream = cv2.VideoCapture(video_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            
            full_frames = []
            while True:
                ret, frame = video_stream.read()
                if not ret:
                    break
                
                if s['fullres'] != 1:
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    frame = cv2.resize(
                        frame, (int(s['out_height'] * aspect_ratio), s['out_height'])
                    )
                
                if s['rotate']:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                y1, y2, x1, x2 = s['crop']
                if x2 == -1:
                    x2 = frame.shape[1]
                if y2 == -1:
                    y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                
                full_frames.append(frame)
            
            video_stream.release()
        
        print(f"📹 Loaded {len(full_frames)} frames at {fps} FPS")
        
        # ========== CONVERT AUDIO ==========
        wav_path = audio_path
        if not audio_path.endswith(".wav"):
            wav_path = "temp/temp.wav"
            print("🔄 Converting audio to WAV...")
            subprocess.check_call([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", audio_path, wav_path
            ])
        
        # ========== ANALYZE AUDIO ==========
        print("🎵 Analyzing audio...")
        wav = audio.load_wav(wav_path, 16000)
        mel = audio.melspectrogram(wav)
        
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError("Mel contains NaN! Try adding small noise to audio.")
        
        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        
        print(f"🎵 Generated {len(mel_chunks)} mel chunks")
        
        # ========== FACE DETECTION ==========
        if s['box'][0] == -1:
            if not s['static']:
                face_det_results = self._face_detect(
                    full_frames, s['pads'], s['nosmooth'],
                    s['face_det_batch_size'], s['cache_file']
                )
            else:
                face_det_results = self._face_detect(
                    [full_frames[0]], s['pads'], s['nosmooth'],
                    s['face_det_batch_size'], s['cache_file']
                )
        else:
            print("📦 Using specified bounding box...")
            y1, y2, x1, x2 = s['box']
            face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]
        
        # ========== PREPARE FRAMES ==========
        full_frames = full_frames[:len(mel_chunks)]
        
        if s['preview_settings']:
            full_frames = [full_frames[0]]
            mel_chunks = [mel_chunks[0]]
        
        print(f"🖼️ Processing {len(full_frames)} frames")
        
        # ========== SETUP OUTPUT VIDEO ==========
        frame_h, frame_w = full_frames[0].shape[:-1]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("temp/result.mp4", fourcc, fps, (frame_w, frame_h))
        
        # ========== PROCESS BATCHES ==========
        gen = self._datagen(
            full_frames.copy(), mel_chunks, face_det_results,
            s['img_size'], s['wav2lip_batch_size'], s['static'], s['box']
        )
        
        batch_size = s['wav2lip_batch_size']
        total_batches = int(np.ceil(float(len(mel_chunks)) / batch_size))
        
        for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=total_batches, desc="Processing Wav2Lip", ncols=100)
        ):
            if i == 0:
                if s['quality'] != "Fast":
                    print(f"🎭 Mask: dilation={s['mask_dilation']}, feathering={s['mask_feathering']}")
                print("🚀 Starting inference...")
            
            # To tensor
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            
            # Inference
            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)
            
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            
            # Process predictions
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                
                if s['debug_mask']:
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                cf = f[y1:y2, x1:x2]
                
                # GFPGAN upscale
                if s['quality'] == "Enhanced" and self.sr_model:
                    p = self._upscale(p)
                
                # Apply mask
                if s['quality'] in ["Enhanced", "Improved"]:
                    if s['mouth_tracking']:
                        p, _ = self._create_tracked_mask(p, cf, s['mask_dilation'], s['mask_feathering'])
                    else:
                        p, _ = self._create_mask(p, cf, s['mask_dilation'], s['mask_feathering'])
                
                f[y1:y2, x1:x2] = p
                
                # Preview window
                if not g_colab and not s['preview_settings']:
                    if preview_window == "Face":
                        cv2.imshow("Face Preview - Q to abort", p)
                    elif preview_window == "Full":
                        cv2.imshow("Full Preview - Q to abort", f)
                    elif preview_window == "Both":
                        cv2.imshow("Face Preview - Q to abort", p)
                        cv2.imshow("Full Preview - Q to abort", f)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        out.release()
                        cv2.destroyAllWindows()
                        return None
                
                if s['preview_settings']:
                    cv2.imwrite("temp/preview.jpg", f)
                    if not g_colab:
                        cv2.imshow("Preview - Q to close", f)
                        cv2.waitKey(-1)
                    return "temp/preview.jpg"
                else:
                    out.write(f)
        
        cv2.destroyAllWindows()
        out.release()
        
        # ========== MERGE AUDIO ==========
        print("🎬 Merging audio with video...")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", "temp/result.mp4",
            "-i", wav_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ Video saved: {output_path}")
            
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"📁 File size: {size_mb:.2f} MB")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed: {e}")
            raise


# ==============================================================================
#                     STANDALONE FACE DETECTION FUNCTION
# ==============================================================================
def standalone_detect(video_path, target_frames=None, output_cache="face_cache.pkl",
                      pads=[0, 10, 0, 0], nosmooth=False, out_height=480,
                      fullres=1, crop=[0, -1, 0, -1], rotate=False,
                      face_det_batch_size=16, loop_video=True):
    """
    Detect mặt và lưu cache độc lập.
    Dùng để pre-cache face detection trước khi xử lý.
    
    Args:
        video_path: Đường dẫn video
        target_frames: Số frames cần detect (nếu None, dùng toàn bộ video)
        output_cache: File .pkl để lưu cache
        pads: Padding [top, bottom, left, right]
        nosmooth: Không smooth boxes
        out_height: Chiều cao output
        fullres: Resize factor
        crop: Crop region [y1, y2, x1, x2]
        rotate: Xoay video 90 độ
        face_det_batch_size: Batch size cho face detection
        loop_video: Nếu True, loop video để đủ target_frames
    
    Returns:
        str: Đường dẫn file cache
    """
    print(f"\n{'='*60}")
    print(f"🔍 STANDALONE FACE DETECTION")
    print(f"{'='*60}")
    print(f"📹 Video: {video_path}")
    print(f"🎯 Target frames: {target_frames or 'All'}")
    print(f"💾 Cache file: {output_cache}")
    
    # Load detector
    gpu_id = 0 if torch.cuda.is_available() else -1
    detector = RetinaFace(
        gpu_id=gpu_id,
        model_path="checkpoints/mobilenet.pth",
        network="mobilenet"
    )
    
    # Load video frames
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video FPS: {fps}, Total frames: {total_video_frames}")
    
    frames = []
    while True:
        ret, frame = video_stream.read()
        if not ret:
            break
        
        if fullres != 1:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (int(out_height * aspect_ratio), out_height))
        
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        y1, y2, x1, x2 = crop
        if x2 == -1:
            x2 = frame.shape[1]
        if y2 == -1:
            y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        
        frames.append(frame)
    
    video_stream.release()
    
    print(f"📹 Loaded {len(frames)} frames from video")
    
    # Loop video nếu cần
    if target_frames and len(frames) < target_frames and loop_video:
        print(f"🔄 Looping video to reach {target_frames} frames...")
        original_frames = frames.copy()
        while len(frames) < target_frames:
            frames.extend(original_frames)
        frames = frames[:target_frames]
        print(f"📹 Extended to {len(frames)} frames")
    
    # Detect faces
    results = []
    pady1, pady2, padx1, padx2 = pads
    
    def face_rect_gen(images):
        num_batches = math.ceil(len(images) / face_det_batch_size)
        prev_ret = None
        for i in range(num_batches):
            batch = images[i * face_det_batch_size : (i + 1) * face_det_batch_size]
            all_faces = detector(batch)
            for faces in all_faces:
                if faces:
                    box, landmarks, score = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret
    
    print(f"🔍 Detecting faces in {len(frames)} frames...")
    
    for image, rect in tqdm(
        zip(frames, face_rect_gen(frames)),
        total=len(frames),
        desc="Face Detection",
        ncols=100,
    ):
        if rect is None:
            cv2.imwrite("temp/faulty_frame.jpg", image)
            raise ValueError("Face not detected!")
        
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])
    
    boxes = np.array(results)
    if not nosmooth:
        for i in range(len(boxes)):
            T = 5
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
    
    final_results = [
        [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
        for image, (x1, y1, x2, y2) in zip(frames, boxes)
    ]
    
    # Save cache
    with open(output_cache, "wb") as f:
        pickle.dump(final_results, f)
    
    cache_size = os.path.getsize(output_cache) / (1024 * 1024)
    print(f"\n✅ Cache saved: {output_cache}")
    print(f"📁 Cache size: {cache_size:.2f} MB")
    print(f"📊 Cached {len(final_results)} face detections")
    
    return output_cache


# ==============================================================================
#                           LEGACY CLI SUPPORT
# ==============================================================================
# Giữ argparse cho backward compatibility với CLI cũ

parser = argparse.ArgumentParser(
    description="Inference code to lip-sync videos using Wav2Lip models"
)

parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="Path to Wav2Lip checkpoint")
parser.add_argument("--segmentation_path", type=str, 
                    default="checkpoints/face_segmentation.pth")
parser.add_argument("--face", type=str, required=True,
                    help="Path to video/image with face")
parser.add_argument("--audio", type=str, required=True,
                    help="Path to audio file")
parser.add_argument("--outfile", type=str, default="results/result_voice.mp4",
                    help="Output video path")
parser.add_argument("--static", type=bool, default=False)
parser.add_argument("--fps", type=float, default=25.0)
parser.add_argument("--pads", nargs="+", type=int, default=[0, 10, 0, 0])
parser.add_argument("--resize_factor", default=1, type=int)
parser.add_argument("--wav2lip_batch_size", type=int, default=1)
parser.add_argument("--face_det_batch_size", type=int, default=16)
parser.add_argument("--out_height", default=480, type=int)
parser.add_argument("--crop", nargs="+", type=int, default=[0, -1, 0, -1])
parser.add_argument("--box", nargs="+", type=int, default=[-1, -1, -1, -1])
parser.add_argument("--rotate", default=False, action="store_true")
parser.add_argument("--nosmooth", type=str, default=False)
parser.add_argument("--no_seg", default=False, action="store_true")
parser.add_argument("--no_sr", default=False, action="store_true")
parser.add_argument("--sr_model", type=str, default="gfpgan")
parser.add_argument("--fullres", default=3, type=int)
parser.add_argument("--debug_mask", type=str, default=False)
parser.add_argument("--preview_settings", type=str, default=False)
parser.add_argument("--mouth_tracking", type=str, default=False)
parser.add_argument("--mask_dilation", default=150, type=float)
parser.add_argument("--mask_feathering", default=151, type=int)
parser.add_argument("--quality", type=str, default="Fast")


def main():
    """Legacy main function cho CLI compatibility"""
    args = parser.parse_args()
    
    # Tạo engine
    engine = Wav2LipEngine(
        gpu_id=0,
        checkpoint_path=args.checkpoint_path,
        load_sr_model=(args.quality == "Enhanced"),
        sr_model_name=args.sr_model
    )
    
    # Chuyển args thành settings dict
    settings = {
        'quality': args.quality,
        'pads': args.pads,
        'nosmooth': str(args.nosmooth) == "True",
        'mouth_tracking': str(args.mouth_tracking) == "True",
        'mask_dilation': args.mask_dilation,
        'mask_feathering': args.mask_feathering,
        'wav2lip_batch_size': args.wav2lip_batch_size,
        'face_det_batch_size': args.face_det_batch_size,
        'out_height': args.out_height,
        'fullres': args.fullres,
        'static': args.static,
        'fps': args.fps,
        'crop': args.crop,
        'box': args.box,
        'rotate': args.rotate,
        'debug_mask': str(args.debug_mask) == "True",
        'preview_settings': str(args.preview_settings) == "True",
        'cache_file': "last_detected_face.pkl",
    }
    
    # Process
    result = engine.process(args.face, args.audio, args.outfile, settings)
    
    if result:
        print(f"\n✅ Done! Output: {result}")
    else:
        print("\n❌ Processing cancelled or failed")


if __name__ == "__main__":
    main()
