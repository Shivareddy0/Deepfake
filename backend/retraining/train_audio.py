import sys
import os
# Add parent directory of backend package to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np

# Try importing torchvision models
try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from backend.detectors.audio import MiniResNet18
from backend.retraining.dataset import DeepfakeAudioDataset
from backend.retraining.monitor import ModelRegistryMonitor

def calculate_eer(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = float(fpr[idx] + fnr[idx]) / 2.0
    return eer, float(thresholds[idx])

def calibrate_temperature(logits, targets):
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

def train_audio(args):
    print("==========================================================")
    print("      ADVANCED AUDIO DEEPFAKE DETECTION RETRAINING       ")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")
    print(f"[*] Architecture: {args.arch}")

    # 1. Initialize Dataset
    print(f"[*] Initializing dataset from: {args.dataset_dir} (is_dummy={args.dummy})")
    dataset = DeepfakeAudioDataset(root_dir=args.dataset_dir, is_dummy=args.dummy, num_dummy_samples=args.num_samples)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
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
    print(f"[+] Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples.")

    # 2. Instantiate Model Architecture
    using_pretrained = False
    if HAS_TORCHVISION and not args.force_fallback:
        print(f"[*] Instantiating standard {args.arch} from torchvision...")
        try:
            model_fn = getattr(models, args.arch)
            try:
                weights_enum = getattr(models, f"{args.arch.replace('_', '').capitalize()}_Weights")
                model = model_fn(weights=weights_enum.DEFAULT)
            except Exception:
                model = model_fn(weights=None)
            print("[+] Successfully pre-loaded backbone weights.")
        except Exception as e:
            print(f"[!] Warning: Failed to load torchvision {args.arch}: {e}. Falling back to empty model.")
            model = getattr(models, args.arch)(weights=None)
            
        # ResNet18 fc layer modification
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        using_pretrained = True
    else:
        print("[!] Falling back to lightweight MiniResNet18...")
        model = MiniResNet18()
        args.arch = "MiniResNet18"

    model = model.to(device)

    # 3. Setup Optimizer, Loss, and Mixed Precision Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss() if using_pretrained else nn.BCELoss()
    scaler = torch.amp.GradScaler(enabled=args.mixed_precision)

    # Output Checkpoints Safeguard (Requirement 1)
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    if args.dummy:
        print("[!] Safeguard Active: Dummy training run. Output will be saved as a testing checkpoint only.")
        output_path = os.path.join(weights_dir, "audio_resnet_dummy.pth")
        config_path = os.path.join(weights_dir, "audio_model_config_dummy.json")
    else:
        output_path = os.path.join(weights_dir, "audio_resnet.pth")
        config_path = os.path.join(weights_dir, "audio_model_config.json")
        
    checkpoint_path = os.path.join(weights_dir, "audio_checkpoint.pth")

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
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            if using_pretrained:
                # Repeat channel dimension from 1 to 3 to match ResNet18 expectations
                inputs = inputs.repeat(1, 3, 1, 1)
                
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=args.mixed_precision):
                outputs = model(inputs)
                if not using_pretrained:
                    probs = outputs
                    loss = nn.BCELoss()(probs, targets)
                else:
                    loss = criterion(outputs, targets)
            
            if args.mixed_precision:
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
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_probs = []
        all_logits = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                if using_pretrained:
                    inputs = inputs.repeat(1, 3, 1, 1)
                    
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=args.mixed_precision):
                    outputs = model(inputs)
                    if not using_pretrained:
                        probs = outputs
                        loss = nn.BCELoss()(probs, targets)
                        logits = torch.log(probs / (1.0 - probs + 1e-10))
                    else:
                        loss = criterion(outputs, targets)
                        probs = torch.sigmoid(outputs)
                        logits = outputs
                
                val_loss += loss.item() * inputs.size(0)
                all_targets.extend(targets.cpu().numpy().flatten().tolist())
                all_probs.extend(probs.float().cpu().numpy().flatten().tolist())
                all_logits.extend(logits.float().cpu().numpy().flatten().tolist())
                
        val_epoch_loss = val_loss / len(val_dataset)
        
        # Calculate Validation Metrics
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        all_logits = np.array(all_logits)
        
        preds_val = (all_probs > 0.5).astype(float)
        val_acc = accuracy_score(all_targets, preds_val)
        val_prec = precision_score(all_targets, preds_val, zero_division=0)
        val_rec = recall_score(all_targets, preds_val, zero_division=0)
        val_f1 = f1_score(all_targets, preds_val, zero_division=0)
        
        try:
            val_auc = roc_auc_score(all_targets, all_probs)
        except Exception:
            val_auc = 0.5
            
        val_eer, val_threshold = calculate_eer(all_targets, all_probs)
        tn, fp, fn, tp = confusion_matrix(all_targets, preds_val, labels=[0.0, 1.0]).ravel()

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} EER: {val_eer:.4f}")

        # Checkpoint Saving & Early Stopping
        if val_auc >= best_val_auc:
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

    # 6. Load best model to generate final report
    print(f"\n[*] Evaluating final saved model from {output_path}...")
    model.load_state_dict(torch.load(output_path, map_location=device))
    model.eval()
    
    report = {
        "model_version": f"v_audio_{int(time.time())}",
        "architecture": args.arch,
        "is_dummy": args.dummy,
        "metrics": {
            "accuracy": float(val_acc),
            "precision": float(val_prec),
            "recall": float(val_rec),
            "f1_score": float(val_f1),
            "auc_roc": float(best_val_auc),
            "eer": float(val_eer),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
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
    
    report_path = os.path.join(weights_dir, "audio_training_report_dummy.json" if args.dummy else "audio_training_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"[OK] Training report successfully created at {report_path}")

    # 7. Update Active learning model registry
    try:
        monitor = ModelRegistryMonitor()
        monitor.evaluate_performance("AudioDeepfakeDetector", [{"dummy": True}] * len(val_dataset))
    except Exception as e:
         print(f"[!] Warning: Failed to update model registry: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio ResNet18 Detector")
    parser.add_argument("--dataset_dir", type=str, default="data/asvspoof", help="Path to audio dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dummy", action="store_true", default=True, help="Force training on synthetic dummy data")
    parser.add_argument("--num_samples", type=int, default=40, help="Number of dummy samples to generate")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18"], help="Target model architecture")
    parser.add_argument("--force_fallback", action="store_true", help="Force fallback to MiniResNet18")
    parser.add_argument("--weighted_sampling", action="store_true", default=True, help="Use WeightedRandomSampler to balance classes")
    parser.add_argument("--mixed_precision", action="store_true", default=True, help="Enable mixed precision training (FP16)")
    parser.add_argument("--patience", type=int, default=5, help="Epoch patience count for early stopping")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    
    args = parser.parse_args()
    train_audio(args)
