import argparse
import configparser
import os
import sys
import platform
import subprocess

# Import Pipeline mới
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.pipeline import Wav2LipPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Cảnh báo: Không tải được chế độ tối ưu hóa. Chi tiết: {e}")
    PIPELINE_AVAILABLE = False

def get_default_args():
    config = configparser.ConfigParser()
    config_path = 'config.ini'
    defaults = {}
    if os.path.exists(config_path):
        config.read(config_path)
        if 'OPTIONS' in config:
            defaults = dict(config['OPTIONS'])
            bool_keys = ['static', 'nosmooth', 'enhance']
            int_keys = ['out_height', 'wav2lip_batch_size', 'face_det_batch_size', 'resize_factor']
            float_keys = ['fps']
            for key in bool_keys:
                if key in defaults: defaults[key] = defaults[key].lower() in ['true', '1', 'yes', 'y']
            for key in int_keys:
                if key in defaults: defaults[key] = int(defaults[key])
            for key in float_keys:
                if key in defaults: defaults[key] = float(defaults[key])
    return defaults

def main():
    defaults = get_default_args()
    parser = argparse.ArgumentParser(description='Easy Wav2Lip - Fork Optimized by Hyhy3008')
    
    parser.add_argument('--checkpoint_path', type=str, default=defaults.get('checkpoint_path', None))
    parser.add_argument('--face', type=str, default=defaults.get('video_file', None))
    parser.add_argument('--audio', type=str, default=defaults.get('vocal_file', None))
    parser.add_argument('--outfile', type=str, default=defaults.get('outfile', 'results/result_voice.mp4'))
    parser.add_argument('--static', type=bool, default=defaults.get('static', False))
    parser.add_argument('--fps', type=float, default=defaults.get('fps', 25.0))
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0])
    parser.add_argument('--face_det_batch_size', type=int, default=defaults.get('face_det_batch_size', 16))
    parser.add_argument('--wav2lip_batch_size', type=int, default=defaults.get('wav2lip_batch_size', 128))
    parser.add_argument('--resize_factor', default=defaults.get('resize_factor', 1), type=int)
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1])
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1])
    parser.add_argument('--rotate', default=False, action='store_true')
    parser.add_argument('--nosmooth', default=False, action='store_true')
    parser.add_argument('--enhance', type=bool, default=defaults.get('enhance', False))
    parser.add_argument('--segmentation_path', type=str, default='checkpoints/face_segmentation.pth')
    parser.add_argument('--fast', action='store_true', default=False)
    
    args = parser.parse_args()

    if not args.checkpoint_path or not args.face or not args.audio:
        parser.print_help()
        sys.exit(1)

    if args.fast and PIPELINE_AVAILABLE:
        print("\n" + "="*50)
        print("   CHẾ ĐỘ TỐI ƯU HÓA (FAST PIPELINE)")
        print("="*50)
        
        try:
            pipeline = Wav2LipPipeline(args)
            pipeline.run()
        except Exception as e:
            print(f"\n❌ Lỗi Fast Pipeline: {e}")
            print("🔄 Đang chuyển về chế độ chuẩn (Standard)...")
            # SỬA LỖI: Dùng subprocess để chạy inference.py chuẩn, tránh lỗi biến global
            subprocess.run(['python', 'inference.py'], check=False)
            
    elif args.fast and not PIPELINE_AVAILABLE:
        print("\nCảnh báo: Không tìm thấy module Pipeline. Chạy chế độ chuẩn.")
        subprocess.run(['python', 'inference.py'], check=False)
        
    else:
        print("\n" + "="*50)
        print("   CHẾ ĐỘ CHUẨN (STANDARD MODE)")
        print("="*50)
        # SỬA LỖI: Dùng subprocess để chạy inference.py chuẩn
        subprocess.run(['python', 'inference.py'], check=False)

if __name__ == "__main__":
    main()
