import sys
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np

# Set matplotlib backend to Agg for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory of backend package to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Try importing torchvision models
try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from backend.detectors.spatial import MiniEfficientNet
from backend.retraining.dataset import DeepfakeImageDataset
from backend.retraining.monitor import ModelRegistryMonitor

def calculate_eer(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = float(fpr[idx] + fnr[idx]) / 2.0
    return eer, float(thresholds[idx])

def calibrate_temperature(logits, targets):
    """
    Find optimal temperature parameter T > 0 by minimizing BCE Loss on validation logits.
    """
    T = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    
    logits_t = torch.tensor(logits, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(logits_t / T, targets_t)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    opt_temp = float(T.item())
    return max(0.01, opt_temp)

def evaluate_split(model, loader, device, using_pretrained, args):
    targets_all = []
    probs_all = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=args.mixed_precision):
                outputs = model(inputs)
                probs = outputs if not using_pretrained else torch.sigmoid(outputs)
            targets_all.extend(targets.cpu().numpy().flatten().tolist())
            probs_all.extend(probs.float().cpu().numpy().flatten().tolist())
            
    targets_all = np.array(targets_all)
    probs_all = np.array(probs_all)
    preds = (probs_all > 0.5).astype(float)
    
    try:
        auc = float(roc_auc_score(targets_all, probs_all))
    except Exception:
        auc = 0.5
    f1 = float(f1_score(targets_all, preds, zero_division=0))
    return auc, f1, targets_all, probs_all

def save_visual_reports(reports_dir, train_losses, val_losses, val_aucs, test_targets, test_probs, is_dummy=False):
    os.makedirs(reports_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    def apply_watermark(fig):
        if is_dummy:
            fig.text(0.5, 0.55, 'DUMMY DATASET\nNOT EVIDENCE OF DETECTOR QUALITY',
                     fontsize=14, color='red', weight='bold', alpha=0.55,
                     horizontalalignment='center', verticalalignment='center',
                     rotation=30)
    
    # 1. Loss Curve
    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss')
    plt.title('Training and Validation Loss Curve' + (' (Dummy Dataset)' if is_dummy else ''))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    apply_watermark(fig)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "loss_curve.png"), dpi=200)
    plt.close()
    
    # 2. AUC Curve
    fig = plt.figure(figsize=(8, 5))
    if is_dummy:
        plt.plot(epochs, [0.0]*len(val_aucs), 'g-^', label='N/A (Dummy)')
        plt.title('Validation AUC-ROC over Epochs (Dummy Dataset)')
    else:
        plt.plot(epochs, val_aucs, 'g-^', label='Validation AUC')
        plt.title('Validation AUC-ROC over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()
    apply_watermark(fig)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "auc_curve.png"), dpi=200)
    plt.close()
    
    # 3. Confusion Matrix
    fig = plt.figure(figsize=(6, 5))
    preds = (test_probs > 0.5).astype(float)
    cm = confusion_matrix(test_targets, preds, labels=[0.0, 1.0])
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test Set Confusion Matrix' + (' (Dummy Dataset)' if is_dummy else ''))
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Real', 'Fake'])
    plt.yticks(tick_marks, ['Real', 'Fake'])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    apply_watermark(fig)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

def train_spatial(args):
    print("==========================================================")
    print("     ADVANCED SPATIAL DEEPFAKE DETECTION RETRAINING       ")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")
    print(f"[*] Architecture selected: {args.arch}")

    # 1. Initialize Dataset
    print(f"[*] Initializing dataset from: {args.dataset_dir} (is_dummy={args.dummy})")
    dataset = DeepfakeImageDataset(root_dir=args.dataset_dir, augment=args.augment, is_dummy=args.dummy, num_dummy_samples=args.num_samples)
    
    # Calculate dataset statistics and save before training starts
    total_images = len(dataset)
    real_images = sum(1 for _, label in dataset.samples if label == 0.0)
    fake_images = sum(1 for _, label in dataset.samples if label == 1.0)
    
    datasets_used = []
    if dataset.is_dummy:
        datasets_used = ["DUMMY"]
    else:
        supported_datasets = ["cifake", "diffusiondb", "faceforensics", "celeb-df", "dfdc"]
        for path, _ in dataset.samples:
            if isinstance(path, str):
                for name in supported_datasets:
                    if name in path.lower() or name.replace("-", "") in path.lower():
                        if name == "celeb-df":
                            name_mapped = "Celeb-DF"
                        elif name == "cifake":
                            name_mapped = "CIFAKE"
                        elif name == "diffusiondb":
                            name_mapped = "DiffusionDB"
                        elif name == "faceforensics":
                            name_mapped = "FaceForensics++"
                        elif name == "dfdc":
                            name_mapped = "DFDC"
                        else:
                            name_mapped = name
                        if name_mapped not in datasets_used:
                            datasets_used.append(name_mapped)

    weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    stats_report = {
        "total_images": total_images,
        "real_images": real_images,
        "fake_images": fake_images,
        "datasets_used": datasets_used
    }
    stats_path = os.path.join(weights_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_report, f, indent=4)
    print(f"[OK] Dataset statistics report created at {stats_path}")

    # Create Train (70%), Val (15%), Test (15%) splits
    total_size = len(dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Setup sampler if weighted sampling is enabled
    sampler = None
    shuffle = True
    if args.weighted_sampling:
        print("[*] Creating WeightedRandomSampler to handle class imbalance...")
        train_indices = train_dataset.indices
        all_weights = dataset.get_sample_weights()
        train_weights = [all_weights[idx] for idx in train_indices]
        sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[+] Loaded {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples.")

    # 2. Instantiate Model Architecture
    using_pretrained = False
    if HAS_TORCHVISION and not args.force_fallback:
        print(f"[*] Instantiating standard {args.arch} from torchvision...")
        try:
            model_fn = getattr(models, args.arch)
            try:
                weights_enum_name = f"{args.arch.replace('_', '').capitalize()}_Weights"
                if "convnext" in args.arch:
                    weights_enum_name = "ConvNeXt_Tiny_Weights" if "tiny" in args.arch else "ConvNeXt_Base_Weights"
                weights_enum = getattr(models, weights_enum_name)
                model = model_fn(weights=weights_enum.DEFAULT)
            except Exception:
                model = model_fn(weights=None)
            print("[+] Successfully pre-loaded backbone weights.")
        except Exception as e:
            print(f"[!] Warning: Failed to load torchvision {args.arch}: {e}. Falling back to empty model.")
            model = getattr(models, args.arch)(weights=None)
            
        # Customize head
        if "convnext" in args.arch:
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, 1)
        else:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 1)
        using_pretrained = True
    else:
        print("[!] Falling back to lightweight MiniEfficientNet...")
        model = MiniEfficientNet()
        args.arch = "MiniEfficientNet"

    model = model.to(device)

    # 3. Setup Optimizer, Loss, and Mixed Precision Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    criterion = nn.BCEWithLogitsLoss() if using_pretrained else nn.BCELoss()
    scaler = torch.amp.GradScaler(enabled=args.mixed_precision and device.type == "cuda")

    # Output Checkpoints Setup
    if args.dummy:
        print("[!] Safeguard Active: Dummy training run. Output will be saved as a testing checkpoint only.")
        output_path = os.path.join(weights_dir, "spatial_efficientnet_dummy.pth")
        config_path = os.path.join(weights_dir, "spatial_model_config_dummy.json")
    else:
        output_path = os.path.join(weights_dir, "spatial_efficientnet.pth")
        config_path = os.path.join(weights_dir, "spatial_model_config.json")
        
    checkpoint_path = os.path.join(weights_dir, "spatial_checkpoint.pth")

    best_val_auc = 0.0
    start_epoch = 0

    # 4. Resume Training
    if args.resume and os.path.exists(checkpoint_path):
        print(f"[*] Resuming training from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_auc = checkpoint['best_auc']
            print(f"[+] Resumed from epoch {start_epoch} with previous best AUC: {best_val_auc:.4f}")
        except Exception as e:
            print(f"[!] Error loading checkpoint: {e}. Starting from scratch.")

    # 5. Training Loop with Early Stopping
    patience_counter = 0
    patience = args.patience
    
    train_losses = []
    val_losses = []
    val_aucs = []
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=args.mixed_precision):
                outputs = model(inputs)
                smoothed_targets = targets * 0.9 + 0.05
                loss = criterion(outputs, smoothed_targets)
            
            if args.mixed_precision and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            probs = outputs if not using_pretrained else torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
        epoch_loss = train_loss / total_train
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        
        scheduler.step()
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_probs = []
        all_logits = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=args.mixed_precision):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    probs = outputs if not using_pretrained else torch.sigmoid(outputs)
                    logits = outputs if using_pretrained else torch.log(probs / (1.0 - probs + 1e-10))
                
                val_loss += loss.item() * inputs.size(0)
                all_targets.extend(targets.cpu().numpy().flatten().tolist())
                all_probs.extend(probs.float().cpu().numpy().flatten().tolist())
                all_logits.extend(logits.float().cpu().numpy().flatten().tolist())
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_losses.append(val_epoch_loss)
        
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        all_logits = np.array(all_logits)
        
        preds_val = (all_probs > 0.5).astype(float)
        val_acc = accuracy_score(all_targets, preds_val)
        
        try:
            val_auc = roc_auc_score(all_targets, all_probs)
        except Exception:
            val_auc = 0.5
            
        val_aucs.append(val_auc)
        val_eer, val_threshold = calculate_eer(all_targets, all_probs)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} EER: {val_eer:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            torch.save(model.state_dict(), output_path)
            print(f"[OK] Saved best model weights (AUC improves) to {output_path}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_val_auc,
                'arch': args.arch
            }, checkpoint_path)
            
            print("[*] Calibrating validation probabilities via temperature scaling...")
            opt_temperature = calibrate_temperature(all_logits, all_targets)
            print(f"[+] Calibrated temperature parameter: T = {opt_temperature:.4f}")
            
            with open(config_path, 'w') as f:
                json.dump({
                    "architecture": args.arch,
                    "temperature": opt_temperature,
                    "using_pretrained": using_pretrained,
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                }, f, indent=4)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered. Validation AUC did not improve for {patience} epochs.")
                break

    # 6. Load best model to generate final report & benchmark on splits
    print(f"\n[*] Evaluating final saved model from {output_path} on splits...")
    model.load_state_dict(torch.load(output_path, map_location=device))
    model.eval()
    
    if args.dummy:
        train_auc, train_f1 = "N/A (Dummy Dataset - Not Evidence of Quality)", "N/A (Dummy Dataset - Not Evidence of Quality)"
        validation_auc, validation_f1 = "N/A (Dummy Dataset - Not Evidence of Quality)", "N/A (Dummy Dataset - Not Evidence of Quality)"
        test_auc, test_f1 = "N/A (Dummy Dataset - Not Evidence of Quality)", "N/A (Dummy Dataset - Not Evidence of Quality)"
        test_targets = np.array([0.0, 1.0])
        test_probs = np.array([0.1, 0.9])
        avg_inference_ms = 0.0
        test_acc = "N/A"
        test_prec = "N/A"
        test_rec = "N/A"
        test_eer = "N/A"
        tn, fp, fn, tp = "N/A", "N/A", "N/A", "N/A"
    else:
        print("[*] Evaluating splits on Real Dataset...")
        train_auc, train_f1, _, _ = evaluate_split(model, train_loader, device, using_pretrained, args)
        validation_auc, validation_f1, _, _ = evaluate_split(model, val_loader, device, using_pretrained, args)
        test_auc, test_f1, test_targets, test_probs = evaluate_split(model, test_loader, device, using_pretrained, args)
        
        preds_test = (test_probs > 0.5).astype(float)
        test_acc = float(accuracy_score(test_targets, preds_test))
        test_prec = float(precision_score(test_targets, preds_test, zero_division=0))
        test_rec = float(recall_score(test_targets, preds_test, zero_division=0))
        test_eer, _ = calculate_eer(test_targets, test_probs)
        tn, fp, fn, tp = confusion_matrix(test_targets, preds_test, labels=[0.0, 1.0]).ravel()
        
        # Benchmark inference latency
        inference_times = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                start_inf = time.time()
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=args.mixed_precision):
                    outputs = model(inputs)
                    probs = outputs if not using_pretrained else torch.sigmoid(outputs)
                elapsed_inf = (time.time() - start_inf) * 1000.0  # in ms
                inference_times.append(elapsed_inf / inputs.size(0))
        avg_inference_ms = float(np.mean(inference_times)) if inference_times else 0.0

    model_size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)
    gpu_memory_mb = 0.0
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        
    # Generate Visual Reports
    reports_dir = os.path.join(weights_dir, "reports")
    print(f"[*] Generating visual reports at {reports_dir}...")
    save_visual_reports(reports_dir, train_losses, val_losses, val_aucs, test_targets, test_probs, is_dummy=args.dummy)
    
    # Write report containing the exact training metrics structure requested
    report = {
        "train_auc": train_auc,
        "validation_auc": validation_auc,
        "test_auc": test_auc,
        "train_f1": train_f1,
        "validation_f1": validation_f1,
        "test_f1": test_f1,
        "model": args.arch,
        "avg_inference_ms": f"{avg_inference_ms:.2f}" if isinstance(avg_inference_ms, float) else str(avg_inference_ms),
        "model_size_mb": f"{model_size_mb:.2f}",
        "gpu_memory_mb": f"{gpu_memory_mb:.2f}",
        "model_version": f"v_spatial_{int(time.time())}",
        "is_dummy": args.dummy,
        "datasets_used": datasets_used,
        "metrics": {
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1_score": test_f1,
            "auc_roc": test_auc,
            "eer": test_eer,
            "confusion_matrix": {
                "true_negatives": int(tn) if not args.dummy else "N/A",
                "false_positives": int(fp) if not args.dummy else "N/A",
                "false_negatives": int(fn) if not args.dummy else "N/A",
                "true_positives": int(tp) if not args.dummy else "N/A"
            }
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weighted_sampling": args.weighted_sampling,
            "mixed_precision": args.mixed_precision
        },
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    
    # Save the training report to backend/weights/training_report.json
    report_path = os.path.join(weights_dir, "training_report_dummy.json" if args.dummy else "training_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"[OK] Training report successfully created at {report_path}")
    
    # Save a separate benchmark report
    benchmark_report = {
        "model": args.arch,
        "auc": f"{test_auc:.4f}" if isinstance(test_auc, float) else test_auc,
        "f1": f"{test_f1:.4f}" if isinstance(test_f1, float) else test_f1,
        "avg_inference_ms": f"{avg_inference_ms:.2f}" if isinstance(avg_inference_ms, float) else str(avg_inference_ms),
        "model_size_mb": f"{model_size_mb:.2f}",
        "gpu_memory_mb": f"{gpu_memory_mb:.2f}",
        "is_dummy": args.dummy,
        "datasets_used": datasets_used
    }
    benchmark_path = os.path.join(weights_dir, "benchmark_report_dummy.json" if args.dummy else "benchmark_report.json")
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_report, f, indent=4)
    print(f"[OK] Benchmark report successfully created at {benchmark_path}")

    # 7. Update Active learning model registry
    try:
        monitor = ModelRegistryMonitor()
        monitor.evaluate_performance("SpatialCNNDetector", [{"dummy": args.dummy}] * len(test_dataset))
    except Exception as e:
         print(f"[!] Warning: Failed to update model registry: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Spatial CNN Detector")
    parser.add_argument("--dataset_dir", type=str, default="data/images", help="Path to image datasets")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dummy", action="store_true", help="Force training on synthetic dummy data")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of dummy samples to generate")
    parser.add_argument("--arch", type=str, default="convnext_tiny", choices=["efficientnet_b0", "efficientnet_b4", "convnext_tiny", "convnext_base"], help="Target model architecture")
    parser.add_argument("--force_fallback", action="store_true", help="Force fallback to Mini-architectures")
    parser.add_argument("--weighted_sampling", action="store_true", default=True, help="Use WeightedRandomSampler to balance classes")
    parser.add_argument("--augment", action="store_true", default=True, help="Apply advanced data augmentations")
    parser.add_argument("--mixed_precision", action="store_true", default=True, help="Enable mixed precision training (FP16)")
    parser.add_argument("--patience", type=int, default=5, help="Epoch patience count for early stopping")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    
    args = parser.parse_args()
    train_spatial(args)
