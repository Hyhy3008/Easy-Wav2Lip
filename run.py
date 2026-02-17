import argparse
import configparser
import os
import sys
import platform

# Import module inference gốc
try:
    import inference
except ImportError:
    print("Lỗi: Không tìm thấy file inference.py. Hãy đảm bảo file này nằm cùng thư mục.")
    sys.exit(1)

# Import Pipeline mới
try:
    # Thêm thư mục hiện tại vào sys.path để tìm thấy thư mục core
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.pipeline import Wav2LipPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Cảnh báo: Không tải được chế độ tối ưu hóa (Fast Pipeline). Chi tiết: {e}")
    print("Sẽ sử dụng chế độ chuẩn (Standard Mode).")
    PIPELINE_AVAILABLE = False

def get_default_args():
    """Lấy các tham số mặc định từ config.ini nếu có"""
    config = configparser.ConfigParser()
    config_path = 'config.ini'
    defaults = {}
    
    if os.path.exists(config_path):
        config.read(config_path)
        if 'OPTIONS' in config:
            defaults = dict(config['OPTIONS'])
            
            # Chuyển đổi các giá số sang đúng kiểu dữ liệu
            bool_keys = ['static', 'nosmooth', 'enhance']
            int_keys = ['out_height', 'wav2lip_batch_size', 'face_det_batch_size', 'resize_factor']
            float_keys = ['fps']
            
            for key in bool_keys:
                if key in defaults:
                    defaults[key] = defaults[key].lower() in ['true', '1', 'yes', 'y']
            
            for key in int_keys:
                if key in defaults:
                    defaults[key] = int(defaults[key])
                    
            for key in float_keys:
                if key in defaults:
                    defaults[key] = float(defaults[key])
                    
    return defaults

def main():
    # Lấy giá trị mặc định
    defaults = get_default_args()
    
    parser = argparse.ArgumentParser(description='Easy Wav2Lip - Fork Optimized by Hyhy3008')
    
    # --- Tham số đầu vào ---
    parser.add_argument('--checkpoint_path', type=str, default=defaults.get('checkpoint_path', None),
                        help='Đường dẫn file checkpoint model wav2lip')
    parser.add_argument('--face', type=str, default=defaults.get('video_file', None),
                        help='Đường dẫn video hoặc ảnh khuôn mặt gốc')
    parser.add_argument('--audio', type=str, default=defaults.get('vocal_file', None),
                        help='Đường dẫn file âm thanh (audio/video)')
    
    # --- Tham số đầu ra ---
    parser.add_argument('--outfile', type=str, default=defaults.get('outfile', 'results/result_voice.mp4'),
                        help='Đường dẫn file video đầu ra')
    
    # --- Tham số xử lý ---
    parser.add_argument('--static', type=bool, default=defaults.get('static', False))
    parser.add_argument('--fps', type=float, default=defaults.get('fps', 25.0))
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (trên, dưới, trái, phải)')
    parser.add_argument('--face_det_batch_size', type=int, default=defaults.get('face_det_batch_size', 16))
    parser.add_argument('--wav2lip_batch_size', type=int, default=defaults.get('wav2lip_batch_size', 128))
    parser.add_argument('--resize_factor', default=defaults.get('resize_factor', 1), type=int)
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1])
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1])
    parser.add_argument('--rotate', default=False, action='store_true')
    parser.add_argument('--nosmooth', default=False, action='store_true')
    
    # --- Tham số nâng cao (Enhance) ---
    parser.add_argument('--enhance', type=bool, default=defaults.get('enhance', False),
                        help='Bật chế độ nâng cao chất lượng')
    parser.add_argument('--segmentation_path', type=str, default='checkpoints/face_segmentation.pth',
                        help='Đường dẫn model segmentation')
    
    # ==========================================
    # THÊM MỚI: CHẾ ĐỘ TỐI ƯU HÓA (FAST PIPELINE)
    # ==========================================
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Sử dụng kiến trúc Pipeline đa luồng mới để tăng tốc độ xử lý.')
    
    args = parser.parse_args()

    # Logic chọn chế độ chạy
    if args.fast and PIPELINE_AVAILABLE:
        print("\n" + "="*50)
        print("   CHẾ ĐỘ TỐI ƯU HÓA (FAST PIPELINE)")
        print("="*50)
        
        try:
            # Khởi tạo Pipeline
            pipeline = Wav2LipPipeline(args)
            # Chạy
            pipeline.run()
            
        except Exception as e:
            print(f"\nLỗi khi chạy Fast Pipeline: {e}")
            print("Đang chuyển về chế độ chuẩn để đảm bảo hoạt động...")
            # SỬA LỖI: Gọi main không có tham số
            inference.main()
            
    elif args.fast and not PIPELINE_AVAILABLE:
        print("\nCảnh báo: Bạn đã chọn --fast nhưng các thư viện cần thiết chưa sẵn sàng.")
        print("Đang chạy bằng chế độ chuẩn...")
        # SỬA LỖI: Gọi main không có tham số
        inference.main()
        
    else:
        print("\n" + "="*50)
        print("   CHẾ ĐỘ CHUẨN (STANDARD MODE)")
        print("="*50)
        # SỬA LỖI: Gọi main không có tham số
        inference.main()

if __name__ == "__main__":
    main()
