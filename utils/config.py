import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument('--image_dir', type=str, default='data/am2k/train/img', help='Path to image directory')
    parser.add_argument('--mask_dir', type=str, default='data/am2k/train/trimap', help='Path to mask directory')
    parser.add_argument('--alpha_dir', type=str, default='data/am2k/train/alpha', help='Path to alpha directory')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Testing batch size')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=800, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--test_interval', type=int, default=10, help='Interval for testing')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to tensorboard logs')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save outputs')

    return parser.parse_args()

args = parse_args()
