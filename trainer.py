import os
import csv
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchmetrics import JaccardIndex, Accuracy, Precision, Recall

# === Setup global save path ===
SAVE_DIR = "/Users/bhavyabhatia/Documents/Thesis/Project_Thesis/saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(SAVE_DIR, "training_log.csv")


class Trainer:
    def __init__(self, model, train_loader, test_loader, device, num_classes=19):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        # Setup metrics
        self.miou = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.precision = Precision(task="multiclass", average='macro', num_classes=num_classes).to(device)
        self.recall = Recall(task="multiclass", average='macro', num_classes=num_classes).to(device)

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.7)

        # CSV log header flag
        self.write_header = not os.path.exists(CSV_LOG_PATH)

        # Best mIoU so far
        self.best_miou = 0.0

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        train_loss = 0.0

        for batch_idx, (images, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, targets = images.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images, target_shape=targets.shape[1:])  # If your model supports target shape
            loss = nn.functional.cross_entropy(outputs, targets, ignore_index=255)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')

        return train_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        inference_times = []

        with torch.no_grad():
            for images, targets in self.test_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                start_time = time.time()
                outputs = self.model(images, target_shape=targets.shape[1:])
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                loss = nn.functional.cross_entropy(outputs, targets, ignore_index=255)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(targets)

        # === Flatten and clean predictions ===
        all_preds = torch.cat([p.view(-1) for p in all_preds], dim=0)
        all_targets = torch.cat([t.view(-1) for t in all_targets], dim=0)

        # Remove ignored labels (255) from metrics
        valid_mask = all_targets != 255
        all_preds = all_preds[valid_mask]
        all_targets = all_targets[valid_mask]

        # Clamp predictions to valid range [0, 18]
        all_preds = torch.clamp(all_preds, min=0, max=18)

        # === Metrics ===
        metrics = {
            'loss': val_loss / len(self.test_loader),
            'miou': self.miou(all_preds, all_targets),
            'accuracy': self.accuracy(all_preds, all_targets),
            'precision': self.precision(all_preds, all_targets),
            'recall': self.recall(all_preds, all_targets),
            'inference_time': sum(inference_times) / len(inference_times)
        }

        # === Save to CSV ===
        with open(CSV_LOG_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            if self.write_header:
                writer.writerow(['Epoch', 'Loss', 'mIoU', 'Accuracy', 'Precision', 'Recall', 'FPS'])
                self.write_header = False

            writer.writerow([
                epoch + 1,
                metrics['loss'],
                float(metrics['miou']),
                float(metrics['accuracy']),
                float(metrics['precision']),
                float(metrics['recall']),
                1.0 / metrics['inference_time']
            ])

        return metrics
