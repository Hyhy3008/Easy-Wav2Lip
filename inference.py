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

print("\rloading sys         ", end="")
import sys

print("\rloading gc          ", end="")
import gc

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)

print("\rloading GFPGAN      ", end="")
from gfpgan import GFPGANer

print("\rloading load_model  ", end="")
from easy_functions import load_model, g_colab

print("\rimports loaded!     ")


# ============================================================================
# GLOBAL HELPER FUNCTIONS (Dùng chung cho cả Class và CLI)
# ============================================================================

def match_color(target, source):
    """
    Ep mau anh source theo mau anh target.
    Dung khong gian mau Lab de giu do tu nhien.
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


def get_smoothened_boxes(boxes, T):
    """Lam muot toa do face detection"""
    smoothed = []
    for i in range(len(boxes)):
        start = max(0, i - T // 2)
        end = min(len(boxes), i + T // 2 + 1)
        window = boxes[start:end]
        mean_box = np.mean(window, axis=0)
        smoothed.append(np.round(mean_box).astype(np.int32))
    return np.array(smoothed)


def load_sr_model():
    """Load GFPGAN super resolution model"""
    sr_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading GFPGAN on device: " + sr_device)
    run_params = GFPGANer(
        model_path="checkpoints/GFPGANv1.4.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=torch.device(sr_device)
    )
    return run_params


def upscale_with_gfpgan(image, gfpgan_model):
    """Upscale image using GFPGAN"""
    try:
        _, restored_faces, _ = gfpgan_model.enhance(
            image,
            has_aligned=True,
            only_center_face=False,
            paste_back=False
        )
        return restored_faces[0]
    except Exception as e:
        print("Error in upscale: " + str(e))
        return image


# ============================================================================
# CLASS WAV2LIP ENGINE (CORE - ALWAYS-ON ARCHITECTURE)
# ============================================================================

class Wav2LipEngine:
    """
    Wav2Lip Engine - Load models 1 lan, xu ly nhieu lan
    
    Usage:
        engine = Wav2LipEngine(gpu_id=0, checkpoint_path="...")
        output = engine.process(video_path, audio_path, output_path, settings)
    """
    
    def __init__(self, gpu_id, checkpoint_path, load_sr=True):
        """
        Khoi tao engine - Load models vao VRAM 1 lan duy nhat
        
        Args:
            gpu_id: ID cua GPU (0 hoac 1)
            checkpoint_path: Duong dan den checkpoint Wav2Lip
            load_sr: Co load GFPGAN khong
        """
        print(f"\n{'='*60}")
        print(f"INITIALIZING WAV2LIP ENGINE ON GPU {gpu_id}")
        print(f"{'='*60}")
        
        # Device setup
        self.gpu_id = gpu_id
        if torch.cuda.is_available():
            self.device = f'cuda:{gpu_id}'
            torch.cuda.set_device(gpu_id)
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        print(f"Device: {self.device}")
        
        # Load Wav2Lip model
        print("Loading Wav2Lip model...")
        self.model = load_model(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("  - Wav2Lip loaded!")
        
        # Load GFPGAN (optional)
        self.sr_model = None
        if load_sr:
            print("Loading GFPGAN...")
            self.sr_model = load_sr_model()
            print("  - GFPGAN loaded!")
        
        # Load Face Detector
        print("Loading Face Detector...")
        self.detector = RetinaFace(
            gpu_id=gpu_id if torch.cuda.is_available() else -1,
            model_path="checkpoints/mobilenet.pth",
            network="mobilenet"
        )
        self.detector_model = self.detector.model
        print("  - Face Detector loaded!")
        
        # Load Dlib predictor (for masking)
        print("Loading Dlib Predictor...")
        with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
            self.predictor = pickle.load(f)
        with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
            self.mouth_detector = pickle.load(f)
        print("  - Dlib loaded!")
        
        # State variables for masking
        self.last_mask = None
        self.kernel = None
        
        print(f"{'='*60}")
        print(f"ENGINE READY ON {self.device}")
        print(f"{'='*60}\n")
    
    
    def process(self, video_path, audio_path, output_path, settings):
        """
        Xu ly video - Ham chinh goi di goi lai nhieu lan
        
        Args:
            video_path: Duong dan video input
            audio_path: Duong dan audio input
            output_path: Duong dan video output
            settings: Dictionary chua cac tham so
                - quality: "Fast", "Improved", "Enhanced"
                - sharpen_amount: 1.0 - 3.0
                - enable_color_match: True/False
                - cache_file: "cache.pkl"
                - fps, batch_size, pads, mask_dilation, etc.
        
        Returns:
            output_path: Duong dan video da xu ly
        """
        try:
            # Reset state
            self.last_mask = None
            
            # 1. Load video
            print("\n1. Loading video...")
            full_frames, fps = self._load_video(video_path, settings)
            print(f"   Loaded {len(full_frames)} frames at {fps} FPS")
            
            # 2. Load audio
            print("\n2. Loading audio...")
            mel_chunks = self._load_audio(audio_path, fps, settings)
            print(f"   Generated {len(mel_chunks)} mel chunks")
            
            # 3. Face detection
            print("\n3. Face detection...")
            face_det_results = self._face_detect(full_frames, settings)
            print(f"   Detected {len(face_det_results)} face entries")
            
            # 4. Sync frames and cache
            print("\n4. Syncing frames and cache...")
            full_frames, face_det_results = self._sync_frames_cache(
                full_frames, face_det_results, mel_chunks
            )
            print(f"   Synced: {len(full_frames)} frames")
            
            # 5. Inference
            print("\n5. Running Wav2Lip inference...")
            output_frames = self._inference_loop(
                full_frames, face_det_results, mel_chunks, settings
            )
            
            # 6. Save video
            print("\n6. Saving video...")
            final_output = self._save_video(
                output_frames, audio_path, output_path, fps, settings
            )
            
            print(f"\n{'='*60}")
            print(f"DONE! Saved to: {final_output}")
            print(f"{'='*60}\n")
            
            return final_output
            
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    
    def _load_video(self, video_path, settings):
        """Load video frames"""
        # Check if image
        if os.path.isfile(video_path) and video_path.split(".")[-1] in ["jpg", "png", "jpeg"]:
            full_frames = [cv2.imread(video_path)]
            fps = settings.get('fps', 25.0)
            return full_frames, fps
        
        # Load video
        video_stream = cv2.VideoCapture(video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            
            # Resize if needed
            if settings.get('fullres', 3) != 1:
                out_height = settings.get('out_height', 480)
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (int(out_height * aspect_ratio), out_height))
            
            # Rotate if needed
            if settings.get('rotate', False):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Crop if needed
            crop = settings.get('crop', [0, -1, 0, -1])
            y1, y2, x1, x2 = crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            
            full_frames.append(frame)
        
        return full_frames, fps
    
    
    def _load_audio(self, audio_path, fps, settings):
        """Load and process audio to mel spectrogram"""
        # Convert to wav if needed
        if not audio_path.endswith(".wav"):
            print("   Converting audio to .wav...")
            subprocess.check_call([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", audio_path, "temp/temp.wav"
            ])
            audio_path = "temp/temp.wav"
        
        # Load wav
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError("Mel contains nan!")
        
        # Split into chunks
        mel_chunks = []
        mel_step_size = 16
        mel_idx_multiplier = 80.0 / fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        
        return mel_chunks
    
    
    def _face_detect(self, images, settings):
        """Detect faces in all frames"""
        cache_file = settings.get('cache_file', 'last_detected_face.pkl')
        
        # Check cache
        if os.path.exists(cache_file):
            print(f"   Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        
        # Run detection
        print("   No cache found. Running face detection...")
        results = []
        pads = settings.get('pads', [0, 10, 0, 0])
        pady1, pady2, padx1, padx2 = pads
        
        # Face detection
        for image, rect in tqdm(
            zip(images, self._face_rect(images, settings)),
            total=len(images),
            desc="   Detecting faces",
            ncols=100
        ):
            if rect is None:
                raise ValueError("Face not detected!")
            
            y1 = int(max(0, rect[1] - pady1))
            y2 = int(min(image.shape[0], rect[3] + pady2))
            x1 = int(max(0, rect[0] - padx1))
            x2 = int(min(image.shape[1], rect[2] + padx2))
            results.append([x1, y1, x2, y2])
        
        # Smoothing
        boxes = np.array(results, dtype=np.int32)
        if str(settings.get('nosmooth', False)) == "False":
            boxes = get_smoothened_boxes(boxes, T=5)
        
        # Create final results
        final_results = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_crop = images[i][y1:y2, x1:x2].copy()
            final_results.append([face_crop, (y1, y2, x1, x2)])
        
        # Save cache
        with open(cache_file, "wb") as f:
            pickle.dump(final_results, f)
        
        return final_results
    
    
    def _face_rect(self, images, settings):
        """Generator for face rectangles"""
        batch_size = settings.get('face_det_batch_size', 16)
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
    
    
    def _sync_frames_cache(self, frames, cache, mel_chunks):
        """Sync frames and cache, loop if needed"""
        # If audio longer than video, loop video
        if len(mel_chunks) > len(frames):
            print(f"   Audio longer - looping video: {len(frames)} -> {len(mel_chunks)} frames")
            original_frames = frames.copy()
            looped_frames = []
            while len(looped_frames) < len(mel_chunks):
                looped_frames.extend(original_frames)
            frames = looped_frames[:len(mel_chunks)]
            
            # Loop cache accordingly
            looped_cache = []
            while len(looped_cache) < len(mel_chunks):
                looped_cache.extend(cache)
            cache = looped_cache[:len(mel_chunks)]
        else:
            # Trim to mel_chunks length
            frames = frames[:len(mel_chunks)]
            cache = cache[:len(mel_chunks)]
        
        # Final sync check
        min_len = min(len(frames), len(cache))
        frames = frames[:min_len]
        cache = cache[:min_len]
        
        return frames, cache
    
    
    def _inference_loop(self, frames, face_det_results, mel_chunks, settings):
        """Main inference loop"""
        output_frames = []
        img_size = 96
        batch_size = settings.get('batch_size', 1)
        quality = settings.get('quality', 'Fast')
        
        num_synced = len(frames)
        
        for i in tqdm(range(len(mel_chunks)), desc="   Processing", ncols=100):
            idx = i % num_synced
            
            frame = frames[idx].copy()
            face, coords = face_det_results[idx]
            mel = mel_chunks[i]
            
            y1, y2, x1, x2 = coords
            target_h = y2 - y1
            target_w = x2 - x1
            
            # Prepare input
            face_resized = cv2.resize(face.copy(), (img_size, img_size))
            
            # To tensor
            img_masked = face_resized.copy()
            img_masked[:, img_size // 2:] = 0
            img_input = np.concatenate((img_masked, face_resized), axis=2)
            img_input = img_input.astype(np.float32) / 255.0
            img_input = np.transpose(img_input, (2, 0, 1))
            
            mel_input = mel.reshape(mel.shape[0], mel.shape[1], 1)
            mel_input = np.transpose(mel_input, (2, 0, 1))
            
            # To torch
            img_t = torch.FloatTensor(img_input).unsqueeze(0).to(self.device)
            mel_t = torch.FloatTensor(mel_input).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                pred = self.model(mel_t, img_t)
            
            # To numpy
            pred = pred.cpu().numpy()[0].transpose(1, 2, 0) * 255.0
            pred = pred.astype(np.uint8)
            
            # Get original face region
            cf = frame[y1:y2, x1:x2].copy()
            
            # Enhancement pipeline
            if quality == "Enhanced":
                # Step 1: GFPGAN upscale (96x96 -> 512x512)
                if self.sr_model is not None:
                    pred = upscale_with_gfpgan(pred, self.sr_model)
                
                # Step 2: Downscale with LANCZOS4
                pred = cv2.resize(pred, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Step 3: Kernel sharpening
                strength = settings.get('sharpen_amount', 1.5)
                sharpen_kernel = np.array([
                    [0, -1, 0],
                    [-1, strength + 3.5, -1],
                    [0, -1, 0]
                ]) / (strength + 0.5)
                pred = cv2.filter2D(pred, -1, sharpen_kernel)
                
                # Step 4: Match color
                if settings.get('enable_color_match', False):
                    pred = match_color(cf, pred)
                
                # Step 5: Masking
                pred = self._create_mask(pred, cf, settings)
            
            elif quality == "Improved":
                pred = cv2.resize(pred, (target_w, target_h))
                pred = self._create_mask(pred, cf, settings)
            
            else:  # Fast
                pred = cv2.resize(pred, (target_w, target_h))
            
            # Paste back
            frame[y1:y2, x1:x2] = pred
            output_frames.append(frame)
        
        return output_frames
    
    
    def _create_mask(self, img, original_img, settings):
        """Create and apply mouth mask"""
        if self.last_mask is None:
            faces = self.mouth_detector(img)
            if len(faces) == 0:
                return img
            
            face = faces[0]
            shape = self.predictor(img, face)
            
            mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
            x, y, w, h = cv2.boundingRect(mouth_points)
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)
            
            mask_dilation = settings.get('mask_dilation', 150)
            kernel_size = int(max(w, h) * mask_dilation)
            if kernel_size < 1:
                kernel_size = 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel)
            
            mask_feathering = settings.get('mask_feathering', 151)
            if mask_feathering != 0:
                blur = int(max(w, h) * mask_feathering)
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
        
        return out
    
    
    def _save_video(self, frames, audio_path, output_path, fps, settings):
        """Save video with audio"""
        # Write video
        frame_h, frame_w = frames[0].shape[:-1]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("temp/result.mp4", fourcc, fps, (frame_w, frame_h))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # Merge with audio
        if not os.path.exists("temp/result.mp4"):
            raise FileNotFoundError("temp/result.mp4 not found")
        
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", "temp/result.mp4",
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        return output_path


# ============================================================================
# CACHE ONLY MODE (Tao cache doc lap)
# ============================================================================

def main_cache_only():
    """Che do --only_detect: Chi tao cache face detection"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--face", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--outfile", type=str, default="master_cache.pkl")
    parser.add_argument("--target_frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--out_height", type=int, default=480)
    parser.add_argument("--crop", nargs="+", type=int, default=[0, -1, 0, -1])
    parser.add_argument("--rotate", action="store_true", default=False)
    parser.add_argument("--pads", nargs="+", type=int, default=[0, 10, 0, 0])
    parser.add_argument("--nosmooth", type=str, default=False)
    parser.add_argument("--face_det_batch_size", type=int, default=16)
    parser.add_argument("--fullres", type=int, default=3)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("CACHE CREATION MODE")
    print(f"{'='*60}")
    
    # Initialize engine (chi de dung detector)
    engine = Wav2LipEngine(
        gpu_id=0,
        checkpoint_path=args.checkpoint_path,
        load_sr=False  # Khong can GFPGAN
    )
    
    # Load video
    settings = {
        'fullres': args.fullres,
        'out_height': args.out_height,
        'rotate': args.rotate,
        'crop': args.crop,
        'fps': args.fps
    }
    
    print("\nLoading video...")
    full_frames, fps = engine._load_video(args.face, settings)
    print(f"Loaded {len(full_frames)} frames")
    
    # Loop if needed
    target = args.target_frames if args.target_frames > 0 else len(full_frames)
    print(f"Target frames: {target}")
    
    if target > len(full_frames):
        print(f"Looping video: {len(full_frames)} -> {target} frames...")
        looped_frames = []
        while len(looped_frames) < target:
            looped_frames.extend(full_frames)
        full_frames = looped_frames[:target]
    else:
        full_frames = full_frames[:target]
    
    # Detect faces
    cache_file = args.outfile if args.outfile.endswith('.pkl') else "master_cache.pkl"
    
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Deleted old cache: {cache_file}")
    
    settings['cache_file'] = cache_file
    settings['pads'] = args.pads
    settings['nosmooth'] = args.nosmooth
    settings['face_det_batch_size'] = args.face_det_batch_size
    
    face_det_results = engine._face_detect(full_frames, settings)
    
    print(f"\n{'='*60}")
    print(f"CACHE CREATED: {cache_file}")
    print(f"Total entries: {len(face_det_results)}")
    print(f"File size: {os.path.getsize(cache_file) / 1024:.2f} KB")
    print(f"{'='*60}\n")


# ============================================================================
# CLI WRAPPER (BACKWARD COMPATIBILITY - BAO GIO CUNG CHAY DUOC)
# ============================================================================

def main_cli():
    """
    Wrapper cho phep goi tu command line nhu cu:
    python inference.py --face video.mp4 --audio audio.wav ...
    """
    parser = argparse.ArgumentParser(
        description="Inference code to lip-sync videos in the wild using Wav2Lip models"
    )

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--segmentation_path", type=str, default="checkpoints/face_segmentation.pth")
    parser.add_argument("--face", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--outfile", type=str, default="results/result_voice.mp4")
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
    parser.add_argument("--sharpen_amount", type=float, default=1.5)
    parser.add_argument("--enable_color_match", action="store_true")
    
    args = parser.parse_args()
    
    # Convert args to settings dictionary
    settings = {
        'quality': args.quality,
        'sharpen_amount': args.sharpen_amount,
        'enable_color_match': args.enable_color_match,
        'cache_file': 'last_detected_face.pkl',
        'fps': args.fps,
        'batch_size': args.wav2lip_batch_size,
        'pads': args.pads,
        'mask_dilation': args.mask_dilation,
        'mask_feathering': args.mask_feathering,
        'mouth_tracking': args.mouth_tracking,
        'debug_mask': args.debug_mask,
        'static': args.static,
        'out_height': args.out_height,
        'crop': args.crop,
        'box': args.box,
        'rotate': args.rotate,
        'nosmooth': args.nosmooth,
        'fullres': args.fullres,
        'resize_factor': args.resize_factor,
        'face_det_batch_size': args.face_det_batch_size,
    }
    
    # Initialize engine
    print("Initializing Wav2Lip Engine...")
    engine = Wav2LipEngine(
        gpu_id=0,
        checkpoint_path=args.checkpoint_path,
        load_sr=(args.quality == "Enhanced")
    )
    
    # Process
    print("\nProcessing video...")
    output_path = engine.process(
        video_path=args.face,
        audio_path=args.audio,
        output_path=args.outfile,
        settings=settings
    )
    
    print(f"\n*** FINAL OUTPUT: {output_path} ***\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if '--only_detect' in sys.argv:
        # Che do tao cache
        main_cache_only()
    else:
        # Che do inference binh thuong
        main_cli()
