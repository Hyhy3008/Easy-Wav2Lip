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

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
gpu_id = 0 if torch.cuda.is_available() else -1

if device == 'cpu':
    print('Warning: No GPU detected so inference will be done on the CPU which is VERY SLOW!')

parser = argparse.ArgumentParser(
    description="Inference code to lip-sync videos in the wild using Wav2Lip models"
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Name of saved checkpoint to load weights from",
    required=True,
)

parser.add_argument(
    "--segmentation_path",
    type=str,
    default="checkpoints/face_segmentation.pth",
    help="Name of saved checkpoint of segmentation network",
    required=False,
)

parser.add_argument(
    "--face",
    type=str,
    help="Filepath of video/image that contains faces to use",
    required=True,
)

parser.add_argument(
    "--audio",
    type=str,
    help="Filepath of video/audio file to use as raw audio source",
    required=True,
)

parser.add_argument(
    "--outfile",
    type=str,
    help="Video path to save result.",
    default="results/result_voice.mp4",
)

parser.add_argument(
    "--static",
    type=bool,
    help="If True, then use only first video frame for inference",
    default=False,
)

parser.add_argument(
    "--fps",
    type=float,
    help="Can be specified only if input is a static image (default: 25)",
    default=25.0,
    required=False,
)

parser.add_argument(
    "--pads",
    nargs="+",
    type=int,
    default=[0, 10, 0, 0],
    help="Padding (top, bottom, left, right).",
)

parser.add_argument(
    "--resize_factor",
    default=1,
    type=int,
    help="Reduce the resolution by this factor."
)

parser.add_argument(
    "--wav2lip_batch_size",
    type=int,
    help="Batch size for Wav2Lip model(s)",
    default=1
)

parser.add_argument(
    "--face_det_batch_size",
    type=int,
    help="Batch size for face detection",
    default=16
)

parser.add_argument(
    "--out_height",
    default=480,
    type=int,
    help="Output video height.",
)

parser.add_argument(
    "--crop",
    nargs="+",
    type=int,
    default=[0, -1, 0, -1],
    help="Crop video to a smaller region.",
)

parser.add_argument(
    "--box",
    nargs="+",
    type=int,
    default=[-1, -1, -1, -1],
    help="Specify a constant bounding box for the face.",
)

parser.add_argument(
    "--rotate",
    default=False,
    action="store_true",
    help="Flip video right by 90deg.",
)

parser.add_argument(
    "--nosmooth",
    type=str,
    default=False,
    help="Prevent smoothing face detections.",
)

parser.add_argument(
    "--no_seg",
    default=False,
    action="store_true",
    help="Prevent using face segmentation",
)

parser.add_argument(
    "--no_sr",
    default=False,
    action="store_true",
    help="Prevent using super resolution"
)

parser.add_argument(
    "--sr_model",
    type=str,
    default="gfpgan",
    help="Name of upscaler",
    required=False,
)

parser.add_argument(
    "--fullres",
    default=3,
    type=int,
    help="Full resolution flag",
)

parser.add_argument(
    "--debug_mask",
    type=str,
    default=False,
    help="Makes background grayscale",
)

parser.add_argument(
    "--preview_settings",
    type=str,
    default=False,
    help="Processes only one frame"
)

parser.add_argument(
    "--mouth_tracking",
    type=str,
    default=False,
    help="Tracks the mouth in every frame",
)

parser.add_argument(
    "--mask_dilation",
    default=150,
    type=float,
    help="size of mask around mouth",
    required=False,
)

parser.add_argument(
    "--mask_feathering",
    default=151,
    type=int,
    help="amount of feathering of mask",
    required=False,
)

parser.add_argument(
    "--quality",
    type=str,
    help="Choose between Fast, Improved and Enhanced",
    default="Fast",
)

parser.add_argument(
    "--only_detect",
    action="store_true",
    help="Only create face cache and exit."
)

parser.add_argument(
    "--target_frames",
    type=int,
    default=0,
    help="Number of frames to create cache for."
)

parser.add_argument(
    "--sharpen_amount",
    type=float,
    default=1.5,
    help="Do sac net (0.0 = khong net, 1.5 = vua, 3.0 = cuc gat)"
)

# ============================================================================
# THAM SO MOI: MATCH COLOR (KHOP MAU)
# ============================================================================
parser.add_argument(
    "--enable_color_match",
    action="store_true",
    help="Bat tinh nang khop mau giua mat AI va mat goc"
)
# ============================================================================

args = parser.parse_args()

with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
    predictor = pickle.load(f)

with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
    mouth_detector = pickle.load(f)

kernel = last_mask = x = y = w = h = None

g_colab = g_colab()

if not g_colab:
    config = configparser.ConfigParser()
    config.read('config.ini')
    preview_window = config.get('OPTIONS', 'preview_window')

all_mouth_landmarks = []
model = detector = detector_model = None


def do_load(checkpoint_path):
    global model, detector, detector_model
    model = load_model(checkpoint_path)
    detector = RetinaFace(
        gpu_id=gpu_id, model_path="checkpoints/mobilenet.pth", network="mobilenet"
    )
    detector_model = detector.model


def face_rect(images):
    face_batch_size = args.face_det_batch_size
    num_batches = math.ceil(len(images) / face_batch_size)
    prev_ret = None
    for i in range(num_batches):
        batch = images[i * face_batch_size : (i + 1) * face_batch_size]
        all_faces = detector(batch)
        for faces in all_faces:
            if faces:
                box, landmarks, score = faces[0]
                prev_ret = tuple(map(int, box))
            yield prev_ret


# ============================================================================
# HAM MATCH COLOR - KHOP MAU GIUA MAT AI VA MAT GOC
# Cuc nhanh vi dung thuan Numpy, chi mat 1-2ms moi frame
# ============================================================================
def match_color(target, source):
    """
    Ep mau anh source theo mau anh target.
    Dung khong gian mau Lab de giu do tu nhien.
    
    target: Mat goc (mau chuan)
    source: Mat AI (can khop mau)
    return: Mat AI da khop mau
    """
    # Dam bao ca 2 anh co cung kich thuoc
    if target.shape != source.shape:
        source = cv2.resize(source, (target.shape[1], target.shape[0]))
    
    # Chuyen sang khong gian mau Lab de tach biet do sang va mau sac
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Tinh toan gia tri trung binh va do lech chuan cho tung kenh
    for i in range(3):
        target_mean = target_lab[:, :, i].mean()
        target_std = target_lab[:, :, i].std()
        source_mean = source_lab[:, :, i].mean()
        source_std = source_lab[:, :, i].std()

        # Cong thuc ep mau: (source - mean_source) * (std_target / std_source) + mean_target
        if source_std > 1e-5:
            source_lab[:, :, i] = (source_lab[:, :, i] - source_mean) * (target_std / source_std) + target_mean
        else:
            source_lab[:, :, i] = target_mean

    # Clip ve range [0, 255] va chuyen nguoc lai BGR
    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
    
    return result
# ============================================================================


def create_mask(img, original_img):
    global last_mask, x, y, w, h
    
    if last_mask is None:
        faces = mouth_detector(img)
        if len(faces) == 0:
            return img, None
        
        face = faces[0]
        shape = predictor(img, face)
        
        mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
        x, y, w, h = cv2.boundingRect(mouth_points)
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)
        
        kernel_size = int(max(w, h) * args.mask_dilation)
        if kernel_size < 1:
            kernel_size = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel)
        
        if args.mask_feathering != 0:
            blur = int(max(w, h) * args.mask_feathering)
            if blur % 2 == 0:
                blur += 1
            if blur < 1:
                blur = 1
            last_mask = cv2.GaussianBlur(dilated_mask, (blur, blur), 0)
        else:
            last_mask = dilated_mask
    
    mask_to_use = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
    mask_3ch = cv2.cvtColor(mask_to_use, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    out = (img.astype(float) * mask_3ch + original_img.astype(float) * (1 - mask_3ch)).astype(np.uint8)
    
    return out, last_mask


def create_tracked_mask(img, original_img):
    global kernel, last_mask, x, y, w, h
    
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
        
        kernel_size = int(max(w, h) * args.mask_dilation)
        if kernel_size < 1:
            kernel_size = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)
        
        dilated_mask = cv2.dilate(mask, kernel)
        
        blur = args.mask_feathering
        if blur % 2 == 0:
            blur += 1
        blur = int(max(w, h) * blur)
        if blur % 2 == 0:
            blur += 1
        if blur < 1:
            blur = 1
        
        mask_to_use = cv2.GaussianBlur(dilated_mask, (blur, blur), 0)
        last_mask = mask_to_use
    
    mask_3ch = cv2.cvtColor(mask_to_use, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    out = (img.astype(float) * mask_3ch + original_img.astype(float) * (1 - mask_3ch)).astype(np.uint8)
    
    return out, last_mask


def get_smoothened_boxes(boxes, T):
    smoothed = []
    for i in range(len(boxes)):
        start = max(0, i - T // 2)
        end = min(len(boxes), i + T // 2 + 1)
        window = boxes[start:end]
        mean_box = np.mean(window, axis=0)
        smoothed.append(np.round(mean_box).astype(np.int32))
    return np.array(smoothed)


def face_detect(images, results_file="last_detected_face.pkl"):
    if os.path.exists(results_file):
        print("Loading face cache from: " + results_file)
        with open(results_file, "rb") as f:
            cached = pickle.load(f)
        print("Cache loaded: " + str(len(cached)) + " entries")
        return cached

    print("No cache found. Detecting faces for " + str(len(images)) + " frames...")
    results = []
    pady1, pady2, padx1, padx2 = args.pads
    
    tqdm_partial = partial(tqdm, position=0, leave=True)
    for image, rect in tqdm_partial(
        zip(images, face_rect(images)),
        total=len(images),
        desc="detecting face",
        ncols=100
    ):
        if rect is None:
            cv2.imwrite("temp/faulty_frame.jpg", image)
            raise ValueError("Face not detected! Ensure the video contains a face.")
        
        y1 = int(max(0, rect[1] - pady1))
        y2 = int(min(image.shape[0], rect[3] + pady2))
        x1 = int(max(0, rect[0] - padx1))
        x2 = int(min(image.shape[1], rect[2] + padx2))
        results.append([x1, y1, x2, y2])

    boxes = np.array(results, dtype=np.int32)
    
    if str(args.nosmooth) == "False":
        boxes = get_smoothened_boxes(boxes, T=5)
    
    final_results = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        face_crop = images[i][y1:y2, x1:x2].copy()
        final_results.append([face_crop, (y1, y2, x1, x2)])
    
    with open(results_file, "wb") as f:
        pickle.dump(final_results, f)
    
    print("Saved cache: " + str(len(final_results)) + " entries to " + results_file)
    return final_results


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames)
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print("Using specified bounding box...")
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    num_frames = len(frames)
    num_cache = len(face_det_results)
    
    print("SYNC INFO:")
    print("  - Video frames: " + str(num_frames))
    print("  - Cache entries: " + str(num_cache))
    print("  - Mel chunks: " + str(len(mels)))

    for i, m in enumerate(mels):
        if args.static:
            idx = 0
        else:
            idx = i % num_frames
        
        cache_idx = idx % num_cache
        
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[cache_idx]
        
        face = face.copy()
        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2 :] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2 :] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16


def _load(checkpoint_path):
    if device != "cpu":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_sr():
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


def upscale(image, properties):
    try:
        _, restored_faces, _ = properties.enhance(
            image,
            has_aligned=True,
            only_center_face=False,
            paste_back=False
        )
        return restored_faces[0]
    except Exception as e:
        print("Error in upscale: " + str(e))
        return image


def main():
    args.img_size = 96

    if os.path.isfile(args.face) and args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        args.static = True

    if not os.path.isfile(args.face):
        raise ValueError("--face argument must be a valid path to video/image file")

    elif args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        if args.fullres != 1:
            print("Resizing video...")
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            if args.fullres != 1:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (int(args.out_height * aspect_ratio), args.out_height))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print("Loaded " + str(len(full_frames)) + " frames at " + str(fps) + " FPS")

    if args.only_detect:
        target = args.target_frames if args.target_frames > 0 else len(full_frames)
        print("=" * 60)
        print("CACHE CREATION MODE")
        print("=" * 60)
        print("Original video: " + str(len(full_frames)) + " frames")
        print("Target cache: " + str(target) + " frames")
        
        if target > len(full_frames):
            print("Looping video to reach " + str(target) + " frames...")
            looped_frames = []
            while len(looped_frames) < target:
                looped_frames.extend(full_frames)
            full_frames = looped_frames[:target]
            print("Created " + str(len(full_frames)) + " frames from loop")
        else:
            full_frames = full_frames[:target]
            print("Using first " + str(len(full_frames)) + " frames")
        
        cache_file = args.outfile if args.outfile.endswith('.pkl') else "master_cache.pkl"
        
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print("Deleted old cache: " + cache_file)
        
        face_detect(full_frames, results_file=cache_file)
        
        print("=" * 60)
        print("CACHE CREATED: " + cache_file)
        print("Total entries: " + str(len(full_frames)))
        print("=" * 60)
        return

    if not args.audio.endswith(".wav"):
        print("Converting audio to .wav")
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", args.audio, "temp/temp.wav"
        ])
        args.audio = "temp/temp.wav"

    print("Analysing audio...")
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError("Mel contains nan!")

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

    print("Audio: " + str(len(mel_chunks)) + " mel chunks")
    print("Video: " + str(len(full_frames)) + " frames")

    if len(mel_chunks) > len(full_frames):
        print("Audio longer than video - Looping video...")
        original_len = len(full_frames)
        looped_frames = []
        while len(looped_frames) < len(mel_chunks):
            looped_frames.extend(full_frames)
        full_frames = looped_frames[:len(mel_chunks)]
        print("Looped: " + str(original_len) + " -> " + str(len(full_frames)) + " frames")
    else:
        full_frames = full_frames[:len(mel_chunks)]
        print("Trimmed to: " + str(len(full_frames)) + " frames")

    if str(args.preview_settings) == "True":
        full_frames = [full_frames[0]]
        mel_chunks = [mel_chunks[0]]
    
    print("Processing " + str(len(full_frames)) + " frames...")
    
    if args.quality == "Enhanced":
        print("Sharpen amount: " + str(args.sharpen_amount))
        if args.enable_color_match:
            print("Color matching: ENABLED")
    
    batch_size = args.wav2lip_batch_size
    if str(args.preview_settings) == "True":
        gen = datagen(full_frames, mel_chunks)
    else:
        gen = datagen(full_frames.copy(), mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(
        tqdm(
            gen,
            total=int(np.ceil(float(len(mel_chunks)) / batch_size)),
            desc="Processing Wav2Lip",
            ncols=100,
        )
    ):
        if i == 0:
            if not args.quality == "Fast":
                print("mask size: " + str(args.mask_dilation) + ", feathering: " + str(args.mask_feathering))
                if not args.quality == "Improved":
                    print("Loading " + args.sr_model)
                    run_params = load_sr()

            print("Starting...")
            frame_h, frame_w = full_frames[0].shape[:-1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("temp/result.mp4", fourcc, fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            
            target_h = y2 - y1
            target_w = x2 - x1

            if str(args.debug_mask) == "True":
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

            p = cv2.resize(p.astype(np.uint8), (target_w, target_h))
            cf = f[y1:y2, x1:x2]

            # ================================================================
            # BUOC 1: GFPGAN - NANG CAP KHUON MAT
            # ================================================================
            if args.quality == "Enhanced":
                p = upscale(p, run_params)
                p = cv2.resize(p, (target_w, target_h))
            
            # ================================================================
            # BUOC 2: MATCH COLOR - KHOP MAU VOI MAT GOC
            # Chi mat 1-2ms, khong anh huong toc do
            # ================================================================
            if args.quality == "Enhanced" and args.enable_color_match:
                p = match_color(cf, p)
            # ================================================================

            # ================================================================
            # BUOC 3: MASK - BLEND VOI ANH GOC
            # ================================================================
            if args.quality in ["Enhanced", "Improved"]:
                if p.shape[:2] != cf.shape[:2]:
                    p = cv2.resize(p, (target_w, target_h))
                
                if str(args.mouth_tracking) == "True":
                    p, _ = create_tracked_mask(p, cf)
                else:
                    p, _ = create_mask(p, cf)

            # ================================================================
            # BUOC 4: SHARPENING - LAM SAC NET CHI TIET
            # Chi mat 0.1ms, khong anh huong toc do
            # ================================================================
            if args.quality == "Enhanced" and args.sharpen_amount > 1.0:
                p_blurred = cv2.GaussianBlur(p, (0, 0), 2.0)
                alpha = args.sharpen_amount
                beta = 1.0 - args.sharpen_amount
                p = cv2.addWeighted(p, alpha, p_blurred, beta, 0)
            # ================================================================

            if p.shape[0] != target_h or p.shape[1] != target_w:
                p = cv2.resize(p, (target_w, target_h))

            f[y1:y2, x1:x2] = p

            if not g_colab:
                if preview_window == "Face":
                    cv2.imshow("face preview - press Q to abort", p)
                elif preview_window == "Full":
                    cv2.imshow("full preview - press Q to abort", f)
                elif preview_window == "Both":
                    cv2.imshow("face preview - press Q to abort", p)
                    cv2.imshow("full preview - press Q to abort", f)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    exit()

            if str(args.preview_settings) == "True":
                cv2.imwrite("temp/preview.jpg", f)
                if not g_colab:
                    cv2.imshow("preview - press Q to close", f)
                    if cv2.waitKey(-1) & 0xFF == ord('q'):
                        exit()
            else:
                out.write(f)

    cv2.destroyAllWindows()
    out.release()

    if str(args.preview_settings) == "False":
        print("Converting to final video...")

        if not os.path.exists("temp/result.mp4"):
            print("ERROR: temp/result.mp4 not found!")
            raise FileNotFoundError("temp/result.mp4 not found")
        
        if not os.path.exists(args.audio):
            print("ERROR: Audio file not found: " + args.audio)
            raise FileNotFoundError("Audio file not found")

        print("temp/result.mp4 size: " + str(os.path.getsize("temp/result.mp4")) + " bytes")

        ffmpeg_cmd = "ffmpeg -y -i temp/result.mp4 -i " + args.audio + " -c:v libx264 -preset medium -crf 18 -c:a aac -b:a 192k -shortest " + args.outfile
        
        print("FFMPEG: " + ffmpeg_cmd)

        try:
            subprocess.run(ffmpeg_cmd, shell=True, check=True)
            print("Done! Saved to: " + args.outfile)
            
            if os.path.exists(args.outfile):
                print("Output size: " + str(os.path.getsize(args.outfile)) + " bytes")
            
        except subprocess.CalledProcessError as e:
            print("FFMPEG FAILED with code " + str(e.returncode))
            raise


if __name__ == "__main__":
    do_load(args.checkpoint_path)
    main()
