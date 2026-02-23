"""
Wav2LipEngine - Always-On Architecture (Parallel GPU Safe)
===========================================================
Version: V18 Parallel Safe + CodeFormer Enhancer + REALTIME STREAM
- Unique temp paths per process (no conflict)
- Multi-GPU parallel processing support
- Model loaded once, reused multiple times
- Enhanced mode uses CodeFormer (aligned-face enhance) if available
- NEW: stream_frames_from_inputs() for realtime frame output
- NEW: separate det_batch_size and wav_batch_size (backward compatible)
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
    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)


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
    - Input: BGR uint8 face crop
    - Process: resize -> CodeFormer -> output BGR uint8
    """

    def __init__(self, model_path, device, w=0.5, use_fp16=True, input_size=512):
        self.model_path = model_path
        self.device = device
        self.w = float(w)
        self.use_fp16 = bool(use_fp16)
        self.input_size = int(input_size)

        try:
            from basicsr.archs.codeformer_arch import CodeFormer
        except Exception as e:
            raise ImportError(
                "Cannot import basicsr.archs.codeformer_arch.CodeFormer.\n"
                "Make sure CodeFormer arch was injected into basicsr.\n"
                "Error: " + str(e)
            )

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
                state = ckpt
        else:
            state = ckpt

        self.net.load_state_dict(state, strict=True)
        self.net.eval().to(self.device)

    @torch.no_grad()
    def enhance_aligned(self, bgr_img, w=None):
        if w is not None:
            self.w = float(w)

        inp = cv2.resize(bgr_img, (self.input_size, self.input_size), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

        x = torch.from_numpy(rgb).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = (x - 0.5) / 0.5

        use_amp = (self.device.type == "cuda") and self.use_fp16
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = self.net(x, w=self.w, adain=True)

        if isinstance(out, (list, tuple)):
            out = out[0]

        out = out.squeeze(0).float().clamp(-1, 1).cpu()
        out = (out + 1.0) / 2.0
        out = (out.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out


# ============================================================================
# WAV2LIP ENGINE CLASS
# ============================================================================
class Wav2LipEngine:
    """
    Always-On Wav2Lip Engine - Parallel GPU Safe.
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
            candidates = [
                os.path.join("checkpoints", "codeformer.pth"),
                os.path.join("checkpoints", "CodeFormer", "codeformer.pth"),
                os.path.join("checkpoints", "CodeFormer", "CodeFormer.pth"),
            ]
            codeformer_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        self.codeformer_path = codeformer_path

        print(f"🔌 Initializing Wav2LipEngine on {self.device}...")

        print(f"   Loading Wav2Lip model from {checkpoint_path}...")
        self.model = load_model(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"   ✅ Wav2Lip model loaded!")

        print(f"   Loading RetinaFace detector...")
        self.detector = RetinaFace(
            gpu_id=self.gpu_id,
            model_path="checkpoints/mobilenet.pth",
            network="mobilenet"
        )
        print(f"   ✅ RetinaFace loaded!")

        self.sr_model = None
        if load_sr_model:
            print(f"   Loading Enhancer backend={self.sr_backend} ...")
            self.sr_model = self._load_sr()
            print(f"   ✅ Enhancer loaded!")

        print(f"✅ Wav2LipEngine {gpu_id} Ready on {self.device}!")

    def _load_sr(self):
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
        if self.sr_model is None:
            return image_bgr_uint8
        try:
            if self.sr_backend == "codeformer":
                return self.sr_model.enhance_aligned(image_bgr_uint8, w=codeformer_w)
            else:
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
        if smooth:
            boxes = get_smoothened_boxes(boxes, T=5)

        final_results = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            face_crop = images[i][y1:y2, x1:x2].copy()
            final_results.append([face_crop, (y1, y2, x1, x2)])

        if cache_file:
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(final_results, f)
            print(f"   💾 Saved cache: {len(final_results)} entries -> {cache_file}")

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
            raise ValueError(f"No frames loaded from video: {video_path}")

        print(f"📹 Loaded {len(frames)} frames at {fps:.2f} FPS")
        return frames, fps

    def _load_audio(self, audio_path, fps):
        unique_id = uuid.uuid4().hex[:8]
        temp_wav = f"temp/audio_{self.engine_id}_{unique_id}.wav"
        os.makedirs("temp", exist_ok=True)

        need_cleanup = False
        if not audio_path.endswith('.wav'):
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", audio_path, "-ac", "1", "-ar", "16000", temp_wav
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio_path = temp_wav
            need_cleanup = True

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

    # ============================================================================
    # REALTIME STREAM API
    # ============================================================================
    def stream_frames_from_inputs(
        self,
        frames,
        mel_chunks,
        coords_list,
        settings=None,
        abs_indices=None,          # list[int] frame_idx tuyệt đối cho mỗi mel (merge multi-GPU)
        frame_callback=None,       # fn(abs_idx:int, out_frame_bgr:np.ndarray)
    ):
        """
        REAL-TIME: xuất frame ngay khi render xong (không ghi mp4).
        - frames: list BGR frames (idle frames preload)
        - mel_chunks: list mel chunks (len = số frame cần)
        - coords_list: list (y1,y2,x1,x2) tương ứng frames (hoặc loop)
        - abs_indices: mapping index tuyệt đối (để merge even/odd GPU)
        - frame_callback: gọi ngay khi xong 1 frame

        Nếu frame_callback=None -> yield (abs_idx, frame)
        """
        s = {
            "quality": "Enhanced",
            "enable_color_match": False,
            "sharpen_amount": 0,
            "mask_dilation": 75,
            "mask_feathering": 75,
            # ⚠️ multi-GPU even/odd: tracking state sẽ lệch; default False cho ổn định
            "mouth_tracking": False,
            "debug_mask": False,
            "codeformer_w": None,
            "wav_batch_size": 256,
        }
        if settings:
            s.update(settings)

        if abs_indices is None:
            abs_indices = list(range(len(mel_chunks)))

        fN = len(frames)
        cN = len(coords_list)
        n = len(mel_chunks)
        if n == 0:
            return

        wav_bs = int(s.get("wav_batch_size", 256))
        wav_bs = max(1, wav_bs)

        cached_mask = None
        last_tracked_mask = None

        img_batch, mel_batch, frame_batch, coords_batch, idx_batch = [], [], [], [], []

        def _emit(ii, fr):
            if frame_callback is not None:
                frame_callback(ii, fr)
                return None
            return (ii, fr)

        for local_i in range(n):
            abs_i = int(abs_indices[local_i])

            frame = frames[abs_i % fN].copy()
            y1, y2, x1, x2 = coords_list[abs_i % cN]

            y1 = max(0, int(y1)); y2 = min(frame.shape[0], int(y2))
            x1 = max(0, int(x1)); x2 = min(frame.shape[1], int(x2))

            if y2 <= y1 or x2 <= x1:
                emitted = _emit(abs_i, frame)
                if emitted is not None:
                    yield emitted
                continue

            face = frame[y1:y2, x1:x2].copy()
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            img_batch.append(face_resized)
            mel_batch.append(mel_chunks[local_i])
            frame_batch.append(frame)
            coords_batch.append((y1, y2, x1, x2))
            idx_batch.append(abs_i)

            if len(img_batch) < wav_bs and local_i != n - 1:
                continue

            img_tensor, mel_tensor = self._prepare_batch(img_batch, mel_batch)
            with torch.no_grad():
                pred = self.model(mel_tensor, img_tensor)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            pred = np.clip(pred, 0, 255).astype(np.uint8)

            for p, out_frame, c, out_idx in zip(pred, frame_batch, coords_batch, idx_batch):
                y1, y2, x1, x2 = c
                th, tw = (y2 - y1), (x2 - x1)
                cf = out_frame[y1:y2, x1:x2].copy()

                if s["quality"] == "Enhanced" and self.sr_model is not None:
                    p = self._enhance(p, codeformer_w=s.get("codeformer_w", None))
                    p = cv2.resize(p, (tw, th), interpolation=cv2.INTER_LANCZOS4)

                    if float(s["sharpen_amount"]) > 0:
                        k = float(s["sharpen_amount"])
                        sharpen_kernel = np.array([
                            [0, -1, 0],
                            [-1, k + 3.5, -1],
                            [0, -1, 0]
                        ]) / (k + 0.5)
                        p = cv2.filter2D(p, -1, sharpen_kernel)
                        p = np.clip(p, 0, 255).astype(np.uint8)

                    if s["enable_color_match"]:
                        p = match_color(cf, p)

                    # multi-GPU stable default: cached mask
                    if s["mouth_tracking"]:
                        p, last_tracked_mask = create_tracked_mask(
                            p, cf, s["mask_dilation"], s["mask_feathering"], last_tracked_mask
                        )
                    else:
                        p, cached_mask = create_mask(
                            p, cf, s["mask_dilation"], s["mask_feathering"], cached_mask
                        )

                elif s["quality"] == "Improved":
                    p = cv2.resize(p, (tw, th))
                else:
                    p = cv2.resize(p, (tw, th))

                out_frame[y1:y2, x1:x2] = p

                emitted = _emit(out_idx, out_frame)
                if emitted is not None:
                    yield emitted

            img_batch, mel_batch, frame_batch, coords_batch, idx_batch = [], [], [], [], []

    # ============================================================================
    # OFFLINE PROCESS (kept for compatibility)
    # ============================================================================
    def process(self, video_path, audio_path, output_path, settings=None):
        s = {
            'quality': 'Enhanced',
            'sharpen_amount': 0,
            'enable_color_match': False,
            'mask_dilation': 150,
            'mask_feathering': 75,
            'mouth_tracking': False,

            # backward compatible:
            'batch_size': 16,
            # NEW:
            'det_batch_size': None,
            'wav_batch_size': None,
            'wav_batch_cap': 256,   # default for dual T4 stability

            'resize_height': 0,
            'crop': (0, -1, 0, -1),
            'pads': (0, 10, 0, 0),
            'cache_file': None,
            'smooth_boxes': True,
            'debug_mask': False,
            'preview_only': False,

            'codeformer_w': None,
        }
        if settings:
            s.update(settings)

        # backward compatibility
        if s['det_batch_size'] is None:
            s['det_batch_size'] = int(s.get('batch_size', 16))
        if s['wav_batch_size'] is None:
            s['wav_batch_size'] = int(s.get('batch_size', 16))

        temp_video = None
        out = None

        try:
            resize_h = s['resize_height'] if s['resize_height'] > 0 else None
            full_frames, fps = self._load_video(video_path, resize_h, s['crop'])
            if s['preview_only']:
                full_frames = [full_frames[0]]

            mel_chunks = self._load_audio(audio_path, fps)
            if s['preview_only']:
                mel_chunks = [mel_chunks[0]]

            if len(mel_chunks) > len(full_frames):
                looped = []
                while len(looped) < len(mel_chunks):
                    looped.extend(full_frames)
                full_frames = looped[:len(mel_chunks)]
            else:
                full_frames = full_frames[:len(mel_chunks)]

            face_det_results = self._face_detect(
                full_frames,
                cache_file=s['cache_file'],
                pads=s['pads'],
                batch_size=int(s['det_batch_size']),
                smooth=s['smooth_boxes']
            )

            num_synced = min(len(full_frames), len(face_det_results), len(mel_chunks))
            full_frames = full_frames[:num_synced]
            face_det_results = face_det_results[:num_synced]
            mel_chunks = mel_chunks[:num_synced]
            if num_synced == 0:
                raise ValueError("No frames to process after sync!")

            frame_h, frame_w = full_frames[0].shape[:2]

            unique_id = uuid.uuid4().hex[:8]
            output_dir = os.path.dirname(output_path)
            temp_dir = output_dir if output_dir else "temp"
            os.makedirs(temp_dir, exist_ok=True)

            temp_video = os.path.join(temp_dir, f"temp_{self.engine_id}_{unique_id}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_video, fourcc, fps, (frame_w, frame_h))

            cached_mask = None
            last_tracked_mask = None

            infer_bs = int(s['wav_batch_size'])
            infer_bs = max(1, infer_bs)

            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
            for idx, mel in enumerate(tqdm(mel_chunks, desc="Processing", ncols=100)):
                frame = full_frames[idx].copy()
                face, coords = face_det_results[idx]
                face_resized = cv2.resize(face.copy(), (IMG_SIZE, IMG_SIZE))

                img_batch.append(face_resized)
                mel_batch.append(mel)
                frame_batch.append(frame)
                coords_batch.append(coords)

                if len(img_batch) < infer_bs and idx != len(mel_chunks) - 1:
                    continue

                img_tensor, mel_tensor = self._prepare_batch(img_batch, mel_batch)
                with torch.no_grad():
                    pred = self.model(mel_tensor, img_tensor)

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                pred = np.clip(pred, 0, 255).astype(np.uint8)

                for p, f, c in zip(pred, frame_batch, coords_batch):
                    y1, y2, x1, x2 = c
                    th, tw = (y2 - y1), (x2 - x1)
                    if th <= 0 or tw <= 0:
                        out.write(f)
                        continue

                    if s['debug_mask']:
                        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

                    cf = f[y1:y2, x1:x2].copy()

                    if s['quality'] == "Enhanced" and self.sr_model is not None:
                        p = self._enhance(p, codeformer_w=s.get('codeformer_w', None))
                        p = cv2.resize(p, (tw, th), interpolation=cv2.INTER_LANCZOS4)

                        if float(s['sharpen_amount']) > 0:
                            k = float(s['sharpen_amount'])
                            sharpen_kernel = np.array([
                                [0, -1, 0],
                                [-1, k + 3.5, -1],
                                [0, -1, 0]
                            ]) / (k + 0.5)
                            p = cv2.filter2D(p, -1, sharpen_kernel)
                            p = np.clip(p, 0, 255).astype(np.uint8)

                        if s['enable_color_match']:
                            p = match_color(cf, p)

                        if s['mouth_tracking']:
                            p, last_tracked_mask = create_tracked_mask(
                                p, cf, s['mask_dilation'], s['mask_feathering'], last_tracked_mask
                            )
                        else:
                            p, cached_mask = create_mask(
                                p, cf, s['mask_dilation'], s['mask_feathering'], cached_mask
                            )
                    elif s['quality'] == "Improved":
                        p = cv2.resize(p, (tw, th))
                    else:
                        p = cv2.resize(p, (tw, th))

                    f[y1:y2, x1:x2] = p
                    if not s['preview_only']:
                        out.write(f)

                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

            out.release()
            out = None

            if s['preview_only']:
                preview_path = output_path.replace('.mp4', '_preview.jpg')
                cv2.imwrite(preview_path, f)
                if temp_video and os.path.exists(temp_video):
                    try: os.remove(temp_video)
                    except: pass
                return preview_path

            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", temp_video,
                "-i", audio_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path
            ]
            r = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"FFmpeg merge failed:\n{r.stderr[:1500]}")

            if temp_video and os.path.exists(temp_video):
                try: os.remove(temp_video)
                except: pass

            return output_path if os.path.exists(output_path) else None

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
                try: os.remove(temp_video)
                except: pass
            return None

    def create_cache(self, video_path, cache_file, target_frames=0,
                     resize_height=0, pads=(0, 10, 0, 0), det_batch_size=16):
        resize_h = resize_height if resize_height > 0 else None
        frames, fps = self._load_video(video_path, resize_h)

        if target_frames > 0 and target_frames > len(frames):
            looped = []
            while len(looped) < target_frames:
                looped.extend(frames)
            frames = looped[:target_frames]
        elif target_frames > 0:
            frames = frames[:target_frames]

        if os.path.exists(cache_file):
            os.remove(cache_file)

        results = self._face_detect(frames, cache_file=cache_file, pads=pads, batch_size=int(det_batch_size))
        return len(results)
