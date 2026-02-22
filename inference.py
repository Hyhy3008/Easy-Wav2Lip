"""
Wav2LipEngine - Always-On Architecture (Parallel GPU Safe)
===========================================================
Version: V18 Parallel Safe + CodeFormer Enhancer
- Unique temp paths per process (no conflict)
- Multi-GPU parallel processing support
- Model loaded once, reused multiple times
- Enhanced mode uses CodeFormer (aligned-face enhance) if available
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

# Optional GFPGAN (fallback / optional)
GFPGANer = None
try:
    print("\rLoading GFPGAN      ", end="")
    from gfpgan import GFPGANer as _GFPGANer
    GFPGANer = _GFPGANer
except Exception:
    GFPGANer = None

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
# HELPER FUNCTIONS
# ============================================================================

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, width, height


def get_audio_duration(audio_path):
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
    smoothed = []
    for i in range(len(boxes)):
        start = max(0, i - T // 2)
        end = min(len(boxes), i + T // 2 + 1)
        window = boxes[start:end]
        mean_box = np.mean(window, axis=0)
        smoothed.append(np.round(mean_box).astype(np.int32))
    return np.array(smoothed)


# ============================================================================
# CODEFORMER RESTORER (ALIGNED FACE)
# ============================================================================
class CodeFormerRestorer:
    """
    Minimal CodeFormer wrapper for aligned-face enhancement (no detection/alignment).
    - Input: BGR uint8 face crop (any size)
    - Process: resize -> CodeFormer -> output BGR uint8
    """

    def __init__(self, model_path, device, w=0.5, use_fp16=True, input_size=512):
        self.model_path = model_path
        self.device = device
        self.w = float(w)
        self.use_fp16 = bool(use_fp16)
        self.input_size = int(input_size)

        # Import arch (requires CodeFormer installed so basicsr has codeformer_arch)
        try:
            from basicsr.archs.codeformer_arch import CodeFormer
        except Exception as e:
            raise ImportError(
                "Cannot import basicsr.archs.codeformer_arch.CodeFormer.\n"
                "You must install CodeFormer repo so that codeformer_arch is available.\n"
                "Error: " + str(e)
            )

        # Build net (official settings)
        self.net = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CodeFormer weight not found: {model_path}")

        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "params_ema" in ckpt:
                state = ckpt["params_ema"]
            elif "params" in ckpt:
                state = ckpt["params"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                # sometimes it's directly a state_dict
                state = ckpt
        else:
            state = ckpt

        self.net.load_state_dict(state, strict=True)
        self.net.eval().to(self.device)

    @torch.no_grad()
    def enhance_aligned(self, bgr_img, w=None):
        if w is not None:
            self.w = float(w)

        # resize to input_size for best restoration
        inp = cv2.resize(bgr_img, (self.input_size, self.input_size), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

        x = torch.from_numpy(rgb).float() / 255.0  # HWC 0..1
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)  # 1,3,H,W
        x = (x - 0.5) / 0.5  # to [-1,1]

        use_amp = (self.device.type == "cuda") and self.use_fp16
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = self.net(x, w=self.w, adain=True)

        # out can be tensor or (tensor, ...)
        if isinstance(out, (list, tuple)):
            out = out[0]

        out = out.squeeze(0).float().clamp(-1, 1).cpu()
        out = (out + 1.0) / 2.0  # 0..1
        out = (out.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)  # RGB
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out


# ============================================================================
# WAV2LIP ENGINE CLASS
# ============================================================================
class Wav2LipEngine:
    """
    Always-On Wav2Lip Engine - Parallel GPU Safe.
    Load models 1 lần, gọi process() nhiều lần.
    Mỗi process() sử dụng temp path riêng biệt để tránh conflict.
    """

    def __init__(
        self,
        gpu_id=0,
        checkpoint_path="checkpoints/wav2lip.pth",
        load_sr_model=True,
        sr_backend="codeformer",  # "codeformer" | "gfpgan"
        codeformer_path=None,
        codeformer_w=0.5,
        codeformer_use_fp16=True,
        gfpgan_path="checkpoints/GFPGANv1.4.pth",
    ):
        """
        Args:
            gpu_id: ID GPU
            checkpoint_path: Wav2Lip weights
            load_sr_model: nếu True sẽ load enhancer (CodeFormer hoặc GFPGAN)
            sr_backend: "codeformer" (default) hoặc "gfpgan"
            codeformer_path: path weights codeformer (.pth)
            codeformer_w: fidelity weight (0..1). thấp = đẹp hơn, cao = giống gốc hơn
            codeformer_use_fp16: dùng AMP fp16 trên cuda
            gfpgan_path: path GFPGAN weights
        """
        self.engine_id = gpu_id

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            self.gpu_id = gpu_id
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.gpu_id = -1
        else:
            self.device = torch.device("cpu")
            self.gpu_id = -1
            print("⚠️ Warning: No GPU detected! Inference will be VERY SLOW!")

        self.sr_backend = (sr_backend or "codeformer").lower().strip()
        self.codeformer_w = float(codeformer_w)
        self.codeformer_use_fp16 = bool(codeformer_use_fp16)
        self.gfpgan_path = gfpgan_path

        if codeformer_path is None:
            # common locations
            candidates = [
                os.path.join("checkpoints", "codeformer.pth"),
                os.path.join("checkpoints", "CodeFormer", "codeformer.pth"),
                os.path.join("checkpoints", "CodeFormer", "CodeFormer.pth"),
            ]
            codeformer_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        self.codeformer_path = codeformer_path

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

        # Load enhancer
        self.sr_model = None
        if load_sr_model:
            print(f"   Loading Enhancer backend={self.sr_backend} ...")
            self.sr_model = self._load_sr()
            print(f"   ✅ Enhancer loaded!")

        print(f"✅ Wav2LipEngine {gpu_id} Ready on {self.device}!")

    def _load_sr(self):
        """
        Load enhancer model on correct device.
        - codeformer: CodeFormerRestorer
        - gfpgan: GFPGANer (optional)
        """
        if self.sr_backend == "codeformer":
            return CodeFormerRestorer(
                model_path=self.codeformer_path,
                device=self.device,
                w=self.codeformer_w,
                use_fp16=self.codeformer_use_fp16,
                input_size=512
            )

        if self.sr_backend == "gfpgan":
            if GFPGANer is None:
                raise ImportError("GFPGAN is not available. Please `pip install gfpgan`.")
            sr_device = torch.device(f"cuda:{self.gpu_id}") if torch.cuda.is_available() and self.gpu_id >= 0 else torch.device("cpu")
            if not os.path.exists(self.gfpgan_path):
                raise FileNotFoundError(f"GFPGAN weight not found: {self.gfpgan_path}")
            return GFPGANer(
                model_path=self.gfpgan_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=sr_device
            )

        raise ValueError(f"Unknown sr_backend: {self.sr_backend}")

    def _enhance(self, image_bgr_uint8, codeformer_w=None):
        """
        Enhance face crop using selected backend.
        """
        if self.sr_model is None:
            return image_bgr_uint8

        try:
            if self.sr_backend == "codeformer":
                return self.sr_model.enhance_aligned(image_bgr_uint8, w=codeformer_w)
            else:
                # GFPGAN path
                _, restored_faces, _ = self.sr_model.enhance(
                    image_bgr_uint8,
                    has_aligned=True,
                    only_center_face=False,
                    paste_back=False
                )
                if restored_faces and len(restored_faces) > 0:
                    return restored_faces[0]
                return image_bgr_uint8
        except Exception as e:
            print(f"⚠️ Enhance error ({self.sr_backend}): {e}")
            return image_bgr_uint8

    def _face_rect_generator(self, images, batch_size=16):
        num_batches = math.ceil(len(images) / batch_size)
        prev_ret = None

        for i in range(num_batches):
            batch = images[i * batch_size: (i + 1) * batch_size]
            all_faces = self.detector(batch)

            for faces in all_faces:
                if faces:
                    box, landmarks, score = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret

    def _face_detect(self, images, cache_file=None, pads=(0, 10, 0, 0),
                     batch_size=16, smooth=True):
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

        if smooth:
            boxes = get_smoothened_boxes(boxes, T=5)

        final_results = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_crop = images[i][y1:y2, x1:x2].copy()
            final_results.append([face_crop, (y1, y2, x1, x2)])

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
        if video_path.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            frame = cv2.imread(video_path)
            if frame is None:
                raise ValueError(f"Cannot read image: {video_path}")
            return [frame], 25.0

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
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
            raise ValueError(f"No frames loaded from video: {video_path}")

        print(f"📹 Loaded {len(frames)} frames at {fps:.2f} FPS")
        return frames, fps

    def _load_audio(self, audio_path, fps):
        unique_id = uuid.uuid4().hex[:8]
        temp_wav = f"temp/audio_{self.engine_id}_{unique_id}.wav"
        os.makedirs("temp", exist_ok=True)

        need_cleanup = False

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
            mel_chunks.append(mel[:, start_idx: start_idx + MEL_STEP_SIZE])
            i += 1

        if need_cleanup and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass

        print(f"   ✅ Created {len(mel_chunks)} mel chunks")
        return mel_chunks

    def _prepare_batch(self, faces, mels):
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

            # CodeFormer setting (optional override per run)
            'codeformer_w': None,  # None => use engine default
        }
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
        if self.sr_model is not None:
            if self.sr_backend == "codeformer":
                print(f"   Enhancer: CodeFormer (w={s['codeformer_w'] if s['codeformer_w'] is not None else self.codeformer_w})")
            else:
                print(f"   Enhancer: GFPGAN")
        else:
            print(f"   Enhancer: None")
        print("=" * 60)

        temp_video = None
        out = None

        try:
            # STEP 1: LOAD VIDEO
            resize_h = s['resize_height'] if s['resize_height'] > 0 else None
            full_frames, fps = self._load_video(video_path, resize_h, s['crop'])

            if s['preview_only']:
                full_frames = [full_frames[0]]

            # STEP 2: LOAD AUDIO
            mel_chunks = self._load_audio(audio_path, fps)

            if s['preview_only']:
                mel_chunks = [mel_chunks[0]]

            # STEP 3: SYNC LENGTH
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

            # STEP 4: FACE DETECTION
            face_det_results = self._face_detect(
                full_frames,
                cache_file=s['cache_file'],
                pads=s['pads'],
                batch_size=s['batch_size'],
                smooth=s['smooth_boxes']
            )

            # SYNC FIX
            num_synced = min(len(full_frames), len(face_det_results), len(mel_chunks))
            full_frames = full_frames[:num_synced]
            face_det_results = face_det_results[:num_synced]
            mel_chunks = mel_chunks[:num_synced]

            print(f"   🔄 Synced to: {num_synced} frames")
            if num_synced == 0:
                raise ValueError("No frames to process after sync!")

            # STEP 5: OUTPUT TEMP VIDEO UNIQUE
            frame_h, frame_w = full_frames[0].shape[:2]

            unique_id = uuid.uuid4().hex[:8]
            output_dir = os.path.dirname(output_path)
            temp_dir = output_dir if output_dir else "temp"
            os.makedirs(temp_dir, exist_ok=True)

            temp_video = os.path.join(temp_dir, f"temp_{self.engine_id}_{unique_id}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_video, fourcc, fps, (frame_w, frame_h))

            # STEP 6: INFERENCE LOOP
            print(f"🚀 Processing {num_synced} frames...")

            cached_mask = None
            last_tracked_mask = None

            batch_size = min(int(s['batch_size']), 128)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

            for idx, mel in enumerate(tqdm(mel_chunks, desc="Processing", ncols=100)):
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

                    # IMPORTANT: clip before uint8 to avoid wrap artefacts
                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                    pred = np.clip(pred, 0, 255).astype(np.uint8)

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

                        if s['quality'] == "Enhanced" and self.sr_model is not None:
                            # 1) Enhance (CodeFormer/GFPGAN) on aligned face
                            #    p currently is 96x96 predicted face; enhance works better if run before resizing to target
                            p = self._enhance(p, codeformer_w=s.get('codeformer_w', None))

                            # 2) Resize to target (high quality interpolation)
                            p = cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                            # 3) Sharpen
                            if float(s['sharpen_amount']) > 0:
                                k = float(s['sharpen_amount'])
                                sharpen_kernel = np.array([
                                    [0, -1, 0],
                                    [-1, k + 3.5, -1],
                                    [0, -1, 0]
                                ]) / (k + 0.5)
                                p = cv2.filter2D(p, -1, sharpen_kernel)
                                p = np.clip(p, 0, 255).astype(np.uint8)

                            # 4) Color match
                            if s['enable_color_match']:
                                p = match_color(cf, p)

                            # 5) Mask blend
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
                            p = cv2.resize(p, (target_w, target_h))

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
                        else:
                            p = cv2.resize(p, (target_w, target_h))

                        if p.shape[0] != target_h or p.shape[1] != target_w:
                            p = cv2.resize(p, (target_w, target_h))

                        f[y1:y2, x1:x2] = p

                        if not s['preview_only']:
                            out.write(f)

                    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

            out.release()
            out = None

            # STEP 7: PREVIEW MODE
            if s['preview_only']:
                preview_path = output_path.replace('.mp4', '_preview.jpg')
                cv2.imwrite(preview_path, f)
                print(f"✅ Preview saved: {preview_path}")
                if temp_video and os.path.exists(temp_video):
                    try:
                        os.remove(temp_video)
                    except:
                        pass
                return preview_path

            # STEP 8: MERGE AUDIO
            print("🔊 Merging audio...")

            if not os.path.exists(temp_video):
                raise FileNotFoundError(f"Temp video not found: {temp_video}")

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error",
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

            r = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"FFmpeg merge failed:\n{r.stderr[:2000]}")

            if temp_video and os.path.exists(temp_video):
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

            try:
                if out is not None:
                    out.release()
            except:
                pass

            if temp_video and os.path.exists(temp_video):
                try:
                    os.remove(temp_video)
                except:
                    pass

            return None

    def create_cache(self, video_path, cache_file, target_frames=0,
                     resize_height=0, pads=(0, 10, 0, 0)):
        print("=" * 60)
        print("📦 CACHE CREATION MODE")
        print("=" * 60)

        resize_h = resize_height if resize_height > 0 else None
        frames, fps = self._load_video(video_path, resize_h)

        print(f"   Original: {len(frames)} frames")
        print(f"   Target: {target_frames if target_frames > 0 else 'All'}")

        if target_frames > 0 and target_frames > len(frames):
            print(f"   Looping to reach {target_frames} frames...")
            looped = []
            while len(looped) < target_frames:
                looped.extend(frames)
            frames = looped[:target_frames]
        elif target_frames > 0:
            frames = frames[:target_frames]

        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"   Deleted old cache")

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

    parser = argparse.ArgumentParser(description="Wav2Lip Inference - Always-On Engine (CodeFormer)")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to Wav2Lip model")
    parser.add_argument("--face", type=str, required=True, help="Video or image file")
    parser.add_argument("--audio", type=str, required=True, help="Audio file")
    parser.add_argument("--outfile", type=str, default="results/result.mp4", help="Output path")
    parser.add_argument("--quality", type=str, default="Enhanced", choices=["Fast", "Improved", "Enhanced"])

    parser.add_argument("--sr_backend", type=str, default="codeformer", choices=["codeformer", "gfpgan"])
    parser.add_argument("--codeformer_path", type=str, default=None)
    parser.add_argument("--codeformer_w", type=float, default=0.5)

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

    engine = Wav2LipEngine(
        gpu_id=0,
        checkpoint_path=args.checkpoint_path,
        load_sr_model=(args.quality == "Enhanced"),
        sr_backend=args.sr_backend,
        codeformer_path=args.codeformer_path,
        codeformer_w=args.codeformer_w
    )

    if args.only_detect:
        engine.create_cache(
            video_path=args.face,
            cache_file=args.cache_file or "master_cache.pkl",
            target_frames=args.target_frames,
            resize_height=args.out_height,
            pads=tuple(args.pads)
        )
    else:
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
            'codeformer_w': args.codeformer_w,
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
