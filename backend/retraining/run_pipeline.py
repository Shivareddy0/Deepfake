import os
import sys
import argparse
import subprocess

def run_script(script_name, args):
    """
    Run a python retraining script with arguments as a subprocess.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    cmd = [sys.executable, script_path] + args
    print(f"\n[~] Launching: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Print output in real-time
    for line in process.stdout:
        print(f"  [{script_name}] {line.strip()}")
        
    process.wait()
    if process.returncode != 0:
        print(f"[X] Error: {script_name} failed with return code {process.returncode}")
        return False
    print(f"[OK] {script_name} finished successfully.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Antigravity Deepfake Shield Master Training Pipeline")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Path to image dataset (CIFAKE/FaceForensics++/Celeb-DF/DFDC/DiffusionDB)")
    parser.add_argument("--audio_dir", type=str, default="data/asvspoof", help="Path to audio dataset (ASVspoof/WaveFake)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dummy", action="store_true", help="Force fallback to synthetic dummy dataset training")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of dummy samples to generate per dataset")
    
    parser.add_argument("--image_arch", type=str, default="convnext_tiny", choices=["efficientnet_b0", "efficientnet_b4", "convnext_tiny", "convnext_base"], help="Target image model architecture")
    parser.add_argument("--audio_arch", type=str, default="resnet18", choices=["resnet18"], help="Target audio model architecture")
    parser.add_argument("--force_fallback", action="store_true", help="Force fallback to Mini-architectures")
    parser.add_argument("--weighted_sampling", action="store_true", default=True, help="Use WeightedRandomSampler to balance classes")
    parser.add_argument("--augment", action="store_true", default=True, help="Apply advanced data augmentations to training")
    parser.add_argument("--mixed_precision", action="store_true", default=True, help="Enable mixed precision training (FP16)")
    parser.add_argument("--patience", type=int, default=5, help="Epoch patience count for early stopping")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")

    args = parser.parse_args()

    print("==========================================================")
    print("      ANTIGRAVITY DEEPFAKE SHIELD MASTER RETRAINING CORES ")
    print("==========================================================")
    print(f"Image Dataset: {args.image_dir} (is_dummy={args.dummy})")
    print(f"Audio Dataset: {args.audio_dir} (is_dummy={args.dummy})")
    print(f"Image Arch: {args.image_arch} | Audio Arch: {args.audio_arch}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")

    # Build common argument list to pass to subprocesses
    sub_args = [
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num_samples", str(args.num_samples),
        "--patience", str(args.patience)
    ]
    if args.dummy:
        sub_args.append("--dummy")
    if args.force_fallback:
        sub_args.append("--force_fallback")
    if args.weighted_sampling:
        sub_args.append("--weighted_sampling")
    if args.mixed_precision:
        sub_args.append("--mixed_precision")
    if args.resume:
        sub_args.append("--resume")

    # 1. Run Spatial CNN training
    spatial_args = [
        "--dataset_dir", args.image_dir,
        "--arch", args.image_arch
    ] + sub_args
    if args.augment:
        spatial_args.append("--augment")
        
    spatial_ok = run_script("train_spatial.py", spatial_args)

    if not spatial_ok:
        print("[!] Stopping pipeline due to Spatial training failure.")
        sys.exit(1)

    # 2. Run Audio ResNet training
    audio_args = [
        "--dataset_dir", args.audio_dir,
        "--arch", args.audio_arch
    ] + sub_args
    audio_ok = run_script("train_audio.py", audio_args)

    if not audio_ok:
        print("[!] Stopping pipeline due to Audio training failure.")
        sys.exit(1)

    print("\n==========================================================")
    print("      MASTER TRAINING PIPELINE COMPLETED SUCCESSFULLY     ")
    print("  Weights successfully saved and integrated to detectors. ")
    print("==========================================================")

if __name__ == "__main__":
    main()
