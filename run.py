import argparse
import configparser
import os
import sys
import platform

# Kiểm tra và import module inference gốc
try:
    import inference
except ImportError:
    print("Lỗi: Không tìm thấy file inference.py. Hãy đảm bảo file này nằm cùng thư mục.")
    sys.exit(1)

# Kiểm tra và import Pipeline mới (Tối ưu hóa)
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
        if 'DEFAULT' in config:
            # Lấy các key trong section DEFAULT
            defaults = dict(config['DEFAULT'])
            
            # Chuyển đổi các giá số sang đúng kiểu dữ liệu (int, float, bool)
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
    
    # --- Tham số đầu vào bắt buộc ---
    parser.add_argument('--checkpoint_path', type=str, default=defaults.get('checkpoint_path', None),
                        help='Đường dẫn file checkpoint model wav2lip')
    parser.add_argument('--face', type=str, default=defaults.get('face', None),
                        help='Đường dẫn video hoặc ảnh khuôn mặt gốc')
    parser.add_argument('--audio', type=str, default=defaults.get('audio', None),
                        help='Đường dẫn file âm thanh (audio/video)')
    
    # --- Tham số đầu ra ---
    parser.add_argument('--outfile', type=str, default=defaults.get('outfile', 'results/result_voice.mp4'),
                        help='Đường dẫn file video đầu ra')
    
    # --- Tham số xử lý hình ảnh ---
    parser.add_argument('--static', type=bool, default=defaults.get('static', False),
                        help='Nếu True, chỉ dùng frame đầu tiên để inference (dành cho ảnh tĩnh)')
    parser.add_argument('--fps', type=float, default=defaults.get('fps', 25.0),
                        help='FPS cho video đầu ra (dùng khi input là ảnh)')
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (trên, dưới, trái, phải) cho vùng khuôn mặt')
    parser.add_argument('--face_det_batch_size', type=int, default=defaults.get('face_det_batch_size', 16),
                        help='Batch size cho việc detect khuôn mặt')
    parser.add_argument('--wav2lip_batch_size', type=int, default=defaults.get('wav2lip_batch_size', 128),
                        help='Batch size cho model Wav2Lip')
    parser.add_argument('--resize_factor', default=defaults.get('resize_factor', 1), type=int,
                        help='Hệ số giảm độ phân giải (ví dụ: 2 để giảm xuống 1/2)')
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Cắt video vùng nhỏ hơn (trên, dưới, trái, phải)')
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Chỉ định thủ công bounding box cho khuôn mặt (trên, dưới, trái, phải)')
    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Xoay video 90 độ (nếu video bị ngược)')
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Tắt tính năng làm mịn việc detect khuôn mặt')
    
    # --- Tham số nâng cao (Enhance) ---
    parser.add_argument('--enhance', type=bool, default=defaults.get('enhance', False),
                        help='Bật chế độ nâng cao chất lượng (GFPGAN/RestoreFormer)')
    parser.add_argument('--segmentation_path', type=str, default='checkpoints/face_segmentation.pth',
                        help='Đường dẫn model segmentation (cho chế độ enhance)')
    
    # ==========================================
    # THÊM MỚI: CHẾ ĐỘ TỐI ƯU HÓA (FAST PIPELINE)
    # ==========================================
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Sử dụng kiến trúc Pipeline đa luồng mới để tăng tốc độ xử lý.')
    
    args = parser.parse_args()

    # Kiểm tra tham số bắt buộc
    if not args.checkpoint_path or not args.face or not args.audio:
        parser.print_help()
        print("\nLỗi: Bạn cần cung cấp đầy đủ --checkpoint_path, --face và --audio.")
        sys.exit(1)

    # Logic chọn chế độ chạy
    if args.fast and PIPELINE_AVAILABLE:
        print("\n" + "="*50)
        print("   CHẾ ĐỘ TỐI ƯU HÓA (FAST PIPELINE) ĐÃ BẬT")
        print("="*50)
        
        try:
            # Khởi tạo Pipeline
            pipeline = Wav2LipPipeline(args)
            # Chạy
            pipeline.run()
            
        except Exception as e:
            print(f"\nLỗi khi chạy Fast Pipeline: {e}")
            print("Đang chuyển về chế độ chuẩn để đảm bảo hoạt động...")
            inference.main(args)
            
    elif args.fast and not PIPELINE_AVAILABLE:
        print("\nCảnh báo: Bạn đã chọn --fast nhưng các thư viện cần thiết chưa sẵn sàng.")
        print("Đang chạy bằng chế độ chuẩn...")
        inference.main(args)
        
    else:
        print("\n" + "="*50)
        print("   CHẾ ĐỘ CHUẨN (STANDARD MODE)")
        print("="*50)
        inference.main(args)

if __name__ == "__main__":
    main()
