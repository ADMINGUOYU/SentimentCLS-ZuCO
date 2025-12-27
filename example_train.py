#!/usr/bin/env python
"""
Example script showing basic usage of the training pipeline.
"""

import os
import sys

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("Checking prerequisites...")
    
    # Check if data file exists
    data_path = './data/tmp/zuco_merged.df'
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("\nPlease run preprocessing first:")
        print("  cd data")
        print("  python preprocess_mat.py")
        print("  python preprocess_gen_lbl.py")
        print("  python preprocess_merge.py")
        return False
    
    print(f"✓ Data file found: {data_path}")
    
    # Check if required packages are installed
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import lightning
        print(f"✓ Lightning installed: {lightning.__version__}")
    except ImportError:
        print("❌ Lightning not installed")
        return False
    
    try:
        import tensorboard
        print("✓ TensorBoard installed")
    except ImportError:
        print("❌ TensorBoard not installed")
        return False
    
    return True


def main():
    """Main function."""
    print("=" * 80)
    print("Sentiment Classification Training - Example Script")
    print("=" * 80)
    print()
    
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install required packages and run preprocessing.")
        sys.exit(1)
    
    print("\n✓ All prerequisites met!")
    print("\n" + "=" * 80)
    print("Training Options")
    print("=" * 80)
    
    print("\n1. Quick start (train with default settings):")
    print("   python train.py")
    
    print("\n2. Train with custom settings:")
    print("   python train.py \\")
    print("     --batch_size 32 \\")
    print("     --lr 2e-5 \\")
    print("     --max_epochs 50 \\")
    print("     --early_stopping \\")
    print("     --experiment_name my_experiment")
    
    print("\n3. Train on GPU with mixed precision:")
    print("   python train.py \\")
    print("     --accelerator gpu \\")
    print("     --devices 1 \\")
    print("     --precision bf16-mixed")
    
    print("\n4. View all options:")
    print("   python train.py --help")
    
    print("\n" + "=" * 80)
    print("Viewing Training Progress")
    print("=" * 80)
    
    print("\n1. Launch TensorBoard:")
    print("   ./view_logs.sh")
    print("   # or")
    print("   tensorboard --logdir ./logs")
    
    print("\n2. Open browser to: http://localhost:6006")
    
    print("\n" + "=" * 80)
    print("Example Training Command")
    print("=" * 80)
    
    print("\nTo start training now with recommended settings, run:")
    print("\n  python train.py \\")
    print("    --batch_size 48 \\")
    print("    --val_batch_size 24 \\")
    print("    --lr 1e-5 \\")
    print("    --max_epochs 100 \\")
    print("    --early_stopping \\")
    print("    --patience 10 \\")
    print("    --experiment_name sentiment_cls")
    print()


if __name__ == '__main__':
    main()
