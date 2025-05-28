import os
import time
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from efficientnet_pytorch import EfficientNet
import torch.backends.mps
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import efficientnet_b0


###############################
# PART 1: DATASET HANDLING
###############################

class KeypointDataset(Dataset):

    def __init__(self, src_dir, json_file, images_dir, split, transform=None):
        self.src_dir = Path(src_dir)
        self.json_path = self.src_dir / json_file
        self.image_dir = self.src_dir / "images" / images_dir
        self.split = split
        self.transform = transform

        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        for entry in self.data:
            try:
                img_path = self.image_dir / entry['filename']
                if img_path.exists():
                    with Image.open(img_path) as img:
                        self.image_width, self.image_height = img.size
                        break
            except (FileNotFoundError, KeyError):
                continue
        
        self.data = [entry for entry in self.data if self._is_valid_entry(entry)]
        print(f"Loaded {len(self.data)} valid entries for {split} split")
    
    
    def _is_valid_entry(self, entry):
        if 'filename' not in entry or 'keypoints_projected2D' not in entry:
            return False
        
        if self.split in ['train', 'val'] and 'bbox_gt' not in entry:
            return False
            
        if self.split == 'test' and 'bbox_pred' not in entry:
            return False
        
        img_path = self.image_dir / entry['filename']
        if not img_path.exists():
            return False
        
        return True
    
    def __len__(self):
        return len(self.data)
    
        
    def _get_bbox(self, entry):
        if self.split == 'test':
            if 'bbox_pred' in entry and entry['bbox_pred'] is not None:
                return entry['bbox_pred']
            else:
                return [120, 0, 520, 400]
        else:
            return entry['bbox_gt']
        
    def _rescaled_keypoints(self, keypoints, bbox):
        """Rescale keypoints from original image coordinates to crop coordinates"""
        keypoints_array = np.array(keypoints, dtype=np.float32)
        x_min, y_min = bbox[0], bbox[1]
        
        rescaled = keypoints_array.copy()
        rescaled[:, 0] -= x_min
        rescaled[:, 1] -= y_min
        
        return rescaled
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = self.image_dir / entry['filename']
        image = Image.open(img_path).convert('RGB')
        bbox = self._get_bbox(entry)
        x_min, y_min, x_max, y_max = bbox
        keypoints = entry['keypoints_projected2D']
        rescaled_keypoints = self._rescaled_keypoints(keypoints, bbox)
        
        # Crop the image using original bbox
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        crop_width, crop_height = cropped_image.size
        
        # Make the image square by adding padding
        max_dim = max(crop_width, crop_height)
        square_image = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))  # Black padding
        
        # Paste the cropped image at the center of the square image
        paste_x = (max_dim - crop_width) // 2
        paste_y = (max_dim - crop_height) // 2
        square_image.paste(cropped_image, (paste_x, paste_y))
        
        # Adjust keypoints for the padding
        padded_keypoints = rescaled_keypoints.copy()
        padded_keypoints[:, 0] += paste_x
        padded_keypoints[:, 1] += paste_y

        square_image_np = np.array(square_image)
        
        if self.transform:
            transformed = self.transform(image=square_image_np, keypoints=padded_keypoints)
            square_image = transformed['image']
            padded_keypoints = np.array(transformed['keypoints'], dtype=np.float32)
        
        sample = {
            'image': square_image,
            'keypoints': torch.tensor(padded_keypoints, dtype=torch.float32),
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'filename': entry['filename'],
            'crop_size': torch.tensor([max_dim, max_dim], dtype=torch.float32),
            'original_crop_size': torch.tensor([crop_width, crop_height], dtype=torch.float32),
            'padding_offset': torch.tensor([paste_x, paste_y], dtype=torch.float32) 
        }
        return sample


def get_transforms():
    """Returns augmentation pipelines for training and validation."""
    train_transforms = A.Compose(
        [
            A.Resize(width=224, height=224),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.3),  

            A.Affine(
                shear=5,                 
                scale=(0.85, 1.15),        
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, 
                rotate=0, 
                p=0.4,                  
                border_mode=cv2.BORDER_CONSTANT
            ),


            A.RandomBrightnessContrast(
                contrast_limit=0.3,      
                brightness_limit=0.3,   
                p=0.7                    
            ),

            A.HueSaturationValue(
                hue_shift_limit=(0, 0),
                sat_shift_limit=(0, 0),
                val_shift_limit=(0, 100),  # Reduced from 100
                p=0.6                     # Reduced from 0.6
            ),
            
            A.OneOf([
                A.GaussNoise(per_channel=False,p=0.1),
                A.CLAHE(p=0.5),
                A.ImageCompression(p=0.5),
                A.RandomGamma(p=0.5),
                A.Posterize(num_bits=5, p=0.5),
                A.Blur(blur_limit=1, p=0.5),  
            ], p=0.7),



            # Remove this entire block:
            A.RandomSunFlare(
                 flare_roi=(0, 0, 1, 1),
                 angle_range=(0, 1),
                 num_flare_circles_range=(3, 8),
                 src_radius=100,
                 src_color=(255, 255, 255),
                 method="physics_based",
                 p=0.6
             ),

            A.OneOf([
                A.GaussNoise(per_channel=False,p=0.1),
                A.CLAHE(p=0.5),
                A.ImageCompression(p=0.5),
                A.RandomGamma(p=0.5),
                A.Posterize(num_bits=5, p=0.5),
            ], p=0.5),

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], 
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    val_transforms = A.Compose(
        [
            A.Resize(width=224, height=224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], 
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    
    return train_transforms, val_transforms


def create_data_loaders(args, mode=None):
    """
    Create data loaders based on the specified mode
    """
    if mode is None:
        mode = args.mode
    
    train_transforms, val_transforms = get_transforms()
    train_loader, val_loader, test_loader = None, None, None
    
    if mode in ['train', 'train-test']:
        train_dataset = KeypointDataset(
            src_dir=args.src,
            json_file='train.json',
            images_dir='train',
            split='train',
            transform=train_transforms
        )
        
        val_dataset = KeypointDataset(
            src_dir=args.src,
            json_file='val.json',
            images_dir='val',
            split='val',
            transform=val_transforms
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    if mode in ['test', 'train-test']:
        test_dataset = KeypointDataset(
            src_dir=args.src,
            json_file='test.json',
            images_dir='test',
            split='test',
            transform=val_transforms
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


def visualize_dataset_samples(dataset, num_samples=5):

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        entry = dataset.data[idx]
        img_path = dataset.image_dir / entry['filename']
        image = Image.open(img_path).convert('RGB')
        bbox = dataset._get_bbox(entry)
        keypoints = entry['keypoints_projected2D']
        rescaled_keypoints = dataset._rescaled_keypoints(keypoints, bbox)
        x_min, y_min, x_max, y_max = bbox
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        crop_width, crop_height = cropped_image.size
        max_dim = max(crop_width, crop_height)
        square_image = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        paste_x = (max_dim - crop_width) // 2
        paste_y = (max_dim - crop_height) // 2
        square_image.paste(cropped_image, (paste_x, paste_y))
        padded_keypoints = rescaled_keypoints.copy()
        padded_keypoints[:, 0] += paste_x
        padded_keypoints[:, 1] += paste_y
        sample = dataset[idx]
        
        fig, ax = plt.subplots(1, 4, figsize=(24, 6))
        
        # Original image with bbox and keypoints
        ax[0].imshow(np.array(image))
        ax[0].set_title(f"Original Image: {entry['filename']}")
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                               linewidth=2, edgecolor='g', facecolor='none')
        ax[0].add_patch(rect)
        keypoints_array = np.array(keypoints)
        ax[0].scatter(keypoints_array[:, 0], keypoints_array[:, 1], c='r', marker='x', s=40)
        
        # Cropped image with keypoints
        ax[1].imshow(np.array(cropped_image))
        ax[1].set_title(f"Cropped Image: {entry['filename']}")
        ax[1].scatter(rescaled_keypoints[:, 0], rescaled_keypoints[:, 1], c='r', marker='x', s=40)
        
        # Manual square padded image with keypoints
        ax[2].imshow(np.array(square_image))
        ax[2].set_title(f"Padded Square Image: {entry['filename']}")
        ax[2].scatter(padded_keypoints[:, 0], padded_keypoints[:, 1], c='r', marker='x', s=40)
        
        # Augmented square padded image with keypoints (from dataset)
        if isinstance(sample['image'], torch.Tensor):
            aug_img = sample['image'].permute(1, 2, 0).numpy()
            mean = np.array([0.4897, 0.4897, 0.4897])
            std = np.array([0.2330, 0.2330, 0.2330])
            aug_img = std * aug_img + mean
            aug_img = np.clip(aug_img, 0, 1)
        else:
            aug_img = sample['image']
        ax[3].imshow(aug_img)
        ax[3].set_title(f"Augmented Padded Image: {entry['filename']}")
        aug_keypoints = sample['keypoints'].numpy()
        ax[3].scatter(aug_keypoints[:, 0], aug_keypoints[:, 1], c='r', marker='x', s=40)
        
        plt.tight_layout()
        plt.show()


###############################
# PART 2: MODEL DEFINITION AND TRAINING
###############################


def create_model(num_keypoints=11, pretrained=True):
    model = efficientnet_b0(pretrained=pretrained)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3), # 1280 → 1280
        nn.SiLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features, 1024), 
        nn.SiLU(),
        nn.Linear(1024, 512),
        nn.SiLU(),
        nn.Linear(512, num_keypoints * 2)
    )

    return model


def train_step(model, dataloader, criterion, optimizer, device, num_keypoints=11):
    """Perform one training epoch"""
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        images = batch['image'].to(device)
        target_keypoints = batch['keypoints'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        pred_keypoints = outputs.view(images.shape[0], num_keypoints, 2)
        
        loss = criterion(pred_keypoints, target_keypoints)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(dataloader)

def val_step(model, dataloader, criterion, device, num_keypoints=11):
    """Perform validation"""
    model.eval()
    running_loss = 0.0
    joint_position_errors = []
    validation_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            target_keypoints = batch['keypoints'].to(device)
            
            outputs = model(images)
            pred_keypoints = outputs.view(images.shape[0], num_keypoints, 2)
            
            loss = criterion(pred_keypoints, target_keypoints)
            running_loss += loss.item()
            
            # Compute mean per joint position error (MPJPE) in pixels
            error = torch.sqrt(((pred_keypoints - target_keypoints) ** 2).sum(dim=2)).mean(dim=1)
            joint_position_errors.extend(error.cpu().numpy())

            for i in range(len(images)):
                validation_predictions.append({
                    'pred_keypoints': pred_keypoints[i].cpu().numpy(),
                    'gt_keypoints': target_keypoints[i].cpu().numpy()
                })
            
    avg_loss = running_loss / len(dataloader)
    mpjpe = np.mean(joint_position_errors)
    thresholds = [2, 5, 10]
    pck_results, _ = calculate_pck(validation_predictions, thresholds)
    
    return avg_loss, mpjpe, pck_results



def train_model(model, train_loader, val_loader, args, num_keypoints=11):
    """Train the model with early stopping and checkpoints"""
    device = torch.device('cuda' if torch.cuda.is_available() and args.device >= 0 else 
                         'mps' if torch.backends.mps.is_available() and args.device >= 0 else 'cpu')
    print(f"Training on: {device}")
    
    model = model.to(device)
    
    criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    early_stopping_patience = 50 
    no_improve_count = 0

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, 'training_log.txt')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write(f"Epoch,Train Loss,Val Loss,MPJPE,PCK@2,PCK@5,PCK@10,Learning Rate,Time (s)\n")

    start_epoch = 0
    resume_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
    if os.path.exists(resume_path) and args.resume:
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch}")


    for epoch in range(start_epoch, args.epochs + args.additional_epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        start_time = time.time()
    
        
        train_loss = train_step(model, train_loader, criterion, optimizer, device, num_keypoints)
        val_loss, mpjpe, pck_results = val_step(model, val_loader, criterion, device, num_keypoints)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.6f}")
            no_improve_count = 0 
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        if (epoch + 1) % 10 == 0: 
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss, 
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }, resume_path)

        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"MPJPE: {mpjpe:.4f} pixels | " 
            f"PCK@2: {pck_results[2]:.4f} | "
            f"PCK@5: {pck_results[5]:.4f} | "
            f"PCK@10: {pck_results[10]:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s")
        
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{mpjpe:.4f},{pck_results[2]:.4f},{pck_results[5]:.4f},{pck_results[10]:.4f},{current_lr:.6f},{epoch_time:.2f}\n")
    
    model.load_state_dict(torch.load(best_model_path))
    return model 


###############################
# PART 3: EVALUATION AND INFERENCE
###############################


def test_model(model, test_loader, output_dir, args, device, num_keypoints=11):
    """Test the model and update test.json with predictions and error metrics"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    test_json_path = os.path.join(args.src, 'test.json')
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    test_data_map = {entry['filename']: entry for entry in test_data}
    
    model_space_errors = []
    image_space_errors = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            target_keypoints = batch['keypoints'].to(device)
            filenames = batch['filename']
            bboxes = batch['bbox'].numpy()
            crop_sizes = batch['original_crop_size'].numpy()
            padding_offsets = batch['padding_offset'].numpy()
            
            outputs = model(images)
            pred_keypoints = outputs.view(images.shape[0], num_keypoints, 2)

            for i in range(len(filenames)):
                filename = filenames[i]
                if filename in test_data_map:
                    entry = test_data_map[filename]
                    
                    pred = pred_keypoints[i].cpu().numpy()
                    target = target_keypoints[i].cpu().numpy()
                    bbox = bboxes[i]
                    crop_size = crop_sizes[i]
                    padding_offset = padding_offsets[i]
                    
                    # Model space error (224x224)
                    model_error = float(np.sqrt(((pred - target) ** 2).sum(axis=1)).mean())
                    model_space_errors.append(model_error)
                    
                    # Scale keypoints to original image coordinates
                    pred_original = rescale_to_original(pred, bbox, crop_size, padding_offset=padding_offset)
                    
                    # Image space error
                    gt_keypoints = np.array(entry['keypoints_projected2D'])
                    image_error = float(np.sqrt(((pred_original - gt_keypoints) ** 2).sum(axis=1)).mean())
                    image_space_errors.append(image_error)
                    
                    entry['keypoints_projected2D_pred'] = pred_original.tolist()
                    entry['mpjpe_model'] = model_error
                    entry['mpjpe_image'] = image_error
                    
                    all_predictions.append({
                        'pred_keypoints': pred_original,
                        'gt_keypoints': gt_keypoints
                    })
    
    with open(os.path.join(args.src, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Calculate PCK with different thresholds
    thresholds = [1, 2, 5, 10, 15, 20, 25, 30]
    pck_results, per_keypoint_pck = calculate_pck(all_predictions, thresholds)
    
    # Calculate statistics
    mean_model_error = np.mean(model_space_errors)
    median_model_error = np.median(model_space_errors)
    mean_image_error = np.mean(image_space_errors)
    median_image_error = np.median(image_space_errors)
    
    print(f"\nTest Results:")
    print(f"Mean Model Space Error: {mean_model_error:.4f} pixels")
    print(f"Mean Image Space Error: {mean_image_error:.4f} pixels")
    
    for threshold in thresholds:
        print(f"PCK@{threshold}: {pck_results[threshold]:.4f}")

    with open(os.path.join(args.src, 'test_metrics.txt'), 'w') as f:
        f.write(f"Mean Model Space Error (mpjpe_model): {mean_model_error:.4f} pixels\n")
        f.write(f"Median Model Space Error: {median_model_error:.4f} pixels\n")
        f.write(f"Mean Image Space Error (mpjpe_image): {mean_image_error:.4f} pixels\n")
        f.write(f"Median Image Space Error: {median_image_error:.4f} pixels\n")
        
        for threshold in thresholds:
            f.write(f"PCK@{threshold}: {pck_results[threshold]:.4f}\n")
        
        f.write("\nPer-Keypoint PCK:\n")
        for k in range(num_keypoints):
            f.write(f"Keypoint {k+1}: ")
            for threshold in thresholds:
                f.write(f"@{threshold}: {per_keypoint_pck[threshold][k]:.4f}  ")
            f.write("\n")
    
    return pck_results, per_keypoint_pck


def rescale_to_original(keypoints, bbox, original_crop_size, padding_offset=None):
    """Rescale keypoints from model output (224x224) to original image coordinates"""
    x_min, y_min = bbox[0], bbox[1]
    crop_width, crop_height = original_crop_size
    max_dim = max(crop_width, crop_height)
    
    # Step 1: Scale from 224x224 back to padded square size
    scale_factor = max_dim / 224.0
    keypoints_scaled = keypoints.copy()
    keypoints_scaled[:, 0] *= scale_factor
    keypoints_scaled[:, 1] *= scale_factor
    
    # Step 2: Remove padding (which was added before scaling)
    if padding_offset is not None:
        keypoints_scaled[:, 0] -= padding_offset[0]
        keypoints_scaled[:, 1] -= padding_offset[1]
    
    # Step 3: Add bbox offset to return to original image coordinates
    keypoints_scaled[:, 0] += x_min
    keypoints_scaled[:, 1] += y_min
    
    return keypoints_scaled

def calculate_pck(predictions, thresholds):
    """Calculate Percentage of Correct Keypoints for different thresholds"""
    results = {t: 0 for t in thresholds}
    total_keypoints = 0
    
    num_keypoints = len(predictions[0]['pred_keypoints'])
    per_keypoint_correct = {t: np.zeros(num_keypoints) for t in thresholds}
    per_keypoint_total = np.zeros(num_keypoints)
    
    for pred in predictions:
        pred_keypoints = np.array(pred['pred_keypoints'])
        gt_keypoints = np.array(pred['gt_keypoints'])
        
        # Calculate Euclidean distance for each keypoint
        distances = np.sqrt(np.sum((pred_keypoints - gt_keypoints) ** 2, axis=1))
        
        for t in thresholds:
            correct = distances < t
            results[t] += np.sum(correct)
            per_keypoint_correct[t] += correct
        
        per_keypoint_total += 1
        total_keypoints += len(distances)
    
    # Overall PCK
    for t in thresholds:
        results[t] /= total_keypoints
    
    # Per-keypoint PCK
    per_keypoint_pck = {t: per_keypoint_correct[t] / per_keypoint_total for t in thresholds}
    
    return results, per_keypoint_pck



###############################
# PART 4: MAIN EXECUTION LOGIC
###############################
    
    
def main():
    parser = argparse.ArgumentParser(description='Keypoint Detection: Dataset, Training and Inference')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'train-test', 'inference', 'visualize'], required=True, help='Operation mode')
    parser.add_argument('--src', type=str, required=True, help='Source directory')
    parser.add_argument('--output-dir', type=str, default='runs/keypoints', help='Output directory')
    
    # Dataset visualization parameters
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], help='Dataset split for visualization')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to visualize')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs') 
    parser.add_argument('--additional-epochs', type=int, default=0, help='Additional epochs to train when resuming')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate') 
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for optimizer') 
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=int, default=0, help='GPU device or -1 for CPU')

    # Model parameters
    parser.add_argument('--num-keypoints', type=int, default=11, help='Number of keypoints')
    parser.add_argument('--model-path', type=str, help='Path to model weights')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device >= 0 else 
                         'mps' if torch.backends.mps.is_available() and args.device >= 0 else 'cpu')
    print(f"Using device: {device}")

    if args.mode == 'visualize':
        if not args.split:
            raise ValueError("Split must be provided for visualization mode")
        
        train_transforms, val_transforms = get_transforms()
        
        if args.split == 'train':
            dataset = KeypointDataset(
                src_dir=args.src,
                json_file='train.json',
                images_dir='train',
                split='train',
                transform=train_transforms
            )
        elif args.split == 'val':
            dataset = KeypointDataset(
                src_dir=args.src,
                json_file='val.json',
                images_dir='val',
                split='val',
                transform=val_transforms
            )
        else:  # test
            dataset = KeypointDataset(
                src_dir=args.src,
                json_file='test.json',
                images_dir='test',
                split='test',
                transform=val_transforms
            )
        
        visualize_dataset_samples(dataset, args.samples)
        return

    model = create_model(num_keypoints=args.num_keypoints, pretrained=True)
    
    if args.mode in ['test']:
        if not args.model_path:
            raise ValueError(f"Model path must be provided for {args.mode} mode")
        print(f"Loading model weights from: {args.model_path}")
        
        checkpoint = torch.load(args.model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    if args.mode in ['train', 'train-test']:

        if args.model_path:
            print(f"Loading pre-trained weights from: {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
            else:
                model.load_state_dict(checkpoint)
            print("Pre-trained weights loaded successfully for fine-tuning!")

        print("Creating training data loaders...")
        train_loader, val_loader, _ = create_data_loaders(args, mode='train')
        
        print(f"Starting training for {args.epochs} epochs...")
        model = train_model(model, train_loader, val_loader, args, num_keypoints=args.num_keypoints)
        
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')
        print(f"Training complete. Best model saved at: {best_model_path}")

    if args.mode in ['test', 'train-test']:
        if args.mode == 'test' and not args.model_path:
            raise ValueError("Model path must be provided for test mode")
        
        print("Creating test data loader...")
        if args.mode == 'train-test':
            _, _, test_loader = create_data_loaders(args)
        else:
            _, _, test_loader = create_data_loaders(args, mode='test')
        
        print(f"Starting testing...")
        pck_results, per_keypoint_pck = test_model(model, test_loader, args.output_dir, args, device, num_keypoints=args.num_keypoints)
        print(f"Testing complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

# Example Usage:

# Visualization: python keypoint_detection.py --mode visualize --src synthetic --split train --samples 10

# Train:         python keypoint_detection.py --mode train --src synthetic --output-dir runs/keypoints --epochs 200
# Resume:        python keypoint_detection.py --mode train --src synthetic --output-dir runs/keypoints --resume --epochs 200
# Resume:        python keypoint_detection.py --mode train --src synthetic --output-dir runs/keypoints --resume --additional-epochs 20

# Test:          python keypoint_detection.py --mode test --src sunlamp --model-path runs/keypoints/best_model.pth
# Test:          python keypoint_detection.py --mode test --src synthetic --model-path runs/keypoints/best_model.pth
# Test:          python keypoint_detection.py --mode test --src lightbox --model-path runs/keypoints/best_model.pth
