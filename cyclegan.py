import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import itertools
import os
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
import time
from albumentations.pytorch import ToTensorV2
import random
from torchvision import transforms
from PIL import Image

###############################
# PART 1: DATASET HANNDLING
###############################

class SatelliteDataset(Dataset):
    """Dataset for loading synthetic and real satellite images for CycleGAN"""
    def __init__(self, synthetic_dir, real_dir_type, transform=None, multiplier=1):
        self.synthetic_dir = synthetic_dir
        self.real_dir_type = real_dir_type
        self.transform = transform
        
        if real_dir_type == "sunlamp":
            self.real_dir = "sunlamp/images/test/"
        elif real_dir_type == "lightbox":
            self.real_dir = "lightbox/images/test/"
        else:
            raise ValueError("real_dir_type must be 'sunlamp' or 'lightbox'")
        
        self.synthetic_images = []
        for split in ["train", "test", "val"]:
            split_path = os.path.join(synthetic_dir, "images", split)
            if os.path.exists(split_path):
                self.synthetic_images.extend(
                    [os.path.join(split, img) for img in os.listdir(split_path)]
                )
        
        self.real_images = os.listdir(self.real_dir)
        self.real_len = len(self.real_images)
        
        target_synthetic_len = int(self.real_len * multiplier)
        
        if len(self.synthetic_images) > target_synthetic_len:
            random.seed(42)
            self.synthetic_images = random.sample(self.synthetic_images, target_synthetic_len)
            
        self.synthetic_len = len(self.synthetic_images)
        self.length_dataset = max(self.synthetic_len, self.real_len)
        
        print(f"Using {self.synthetic_len} synthetic images and {self.real_len} {real_dir_type} images")
    
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        synthetic_img_name = self.synthetic_images[index % self.synthetic_len]
        real_img_name = self.real_images[index % self.real_len]
        
        synthetic_path = os.path.join(self.synthetic_dir, "images", synthetic_img_name)
        real_path = os.path.join(self.real_dir, real_img_name)
        
        synthetic_img = np.array(Image.open(synthetic_path).convert("L"))
        real_img = np.array(Image.open(real_path).convert("L"))
        
        synthetic_img = synthetic_img[:, :, np.newaxis]
        real_img = real_img[:, :, np.newaxis]
        
        if self.transform:
            augmentations = self.transform(image=synthetic_img, image0=real_img)
            synthetic_img = augmentations["image"]
            real_img = augmentations["image0"]
        
        return synthetic_img, real_img
   

def get_transforms(mode="train"):
    """Returns image transformations for CycleGAN"""
    if mode == "train":
        transforms = A.Compose(
            [
                A.Resize(width=640, height=400),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )
    else:  # translate mode
        transforms = A.Compose(
            [
                A.Resize(width=640, height=400),
                A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )
    
    return transforms

def create_data_loaders(args, mode="train"):
    transforms = get_transforms(mode=mode)
    
    dataset = SatelliteDataset(
        synthetic_dir=args.synthetic_dir,
        real_dir_type=args.real_dir_type,
        transform=transforms,
        multiplier=1.0  
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=(mode == "train"),
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return dataloader


###############################
# PART 2: MODEL ARCHITECTURE
###############################

##### Generator architecture #####

class Conv2DGen(nn.Module):
   def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
       super().__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
           if down
           else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
           nn.InstanceNorm2d(out_channels),
           nn.ReLU(inplace=True) if use_act else nn.Identity(),
       )
   def forward(self, x):
       return self.conv(x)

class ResidualBlock(nn.Module):
   def __init__(self, channels):
       super().__init__()
       self.block = nn.Sequential(
           Conv2DGen(channels, channels, kernel_size=3, padding=1),
           Conv2DGen(channels, channels, use_act=False, kernel_size=3, padding=1),
       )
   def forward(self, x):
       return x + self.block(x)

class Generator(nn.Module):
   def __init__(self, img_channels=1, num_features=64, num_residuals=9):
       super().__init__()
       self.initial = nn.Sequential(
           nn.Conv2d(
               img_channels,
               num_features,
               kernel_size=7,
               stride=1,
               padding=3,
               padding_mode="reflect",
           ),
           nn.InstanceNorm2d(num_features),
           nn.ReLU(inplace=True),
       )
       self.down_blocks = nn.ModuleList(
           [
               Conv2DGen(
                   num_features, num_features * 2, kernel_size=3, stride=2, padding=1
               ),
               Conv2DGen(
                   num_features * 2,
                   num_features * 4,
                   kernel_size=3,
                   stride=2,
                   padding=1,
               ),
           ]
       )
       self.res_blocks = nn.Sequential(
           *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
       )
       self.up_blocks = nn.ModuleList(
           [
               Conv2DGen(
                   num_features * 4,
                   num_features * 2,
                   down=False,
                   kernel_size=3,
                   stride=2,
                   padding=1,
                   output_padding=1,
               ),
               Conv2DGen(
                   num_features * 2,
                   num_features * 1,
                   down=False,
                   kernel_size=3,
                   stride=2,
                   padding=1,
                   output_padding=1,
               ),
           ]
       )
       self.last = nn.Conv2d(
           num_features * 1,
           img_channels,
           kernel_size=7,
           stride=1,
           padding=3,
           padding_mode="reflect",
       )
   def forward(self, x):
       x = self.initial(x)
       for layer in self.down_blocks:
           x = layer(x)
       x = self.res_blocks(x)
       for layer in self.up_blocks:
           x = layer(x)
       return torch.tanh(self.last(x))
   

#### Discriminator architecture (PatchGAN) ####

class Conv2DDisc(nn.Module):
   def __init__(self, in_channels, out_channels, stride):
       super().__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(
               in_channels,
               out_channels,
               4,
               stride,
               1,
               bias=True,
               padding_mode="reflect",
           ),
           nn.InstanceNorm2d(out_channels),
           nn.LeakyReLU(0.2, inplace=True),
       )
   def forward(self, x):
       return self.conv(x)

class Discriminator(nn.Module):
   def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
       super().__init__()
       self.initial = nn.Sequential(
           nn.Conv2d(
               in_channels,
               features[0],
               kernel_size=4,
               stride=2,
               padding=1,
               padding_mode="reflect",
           ),
           nn.LeakyReLU(0.2, inplace=True),
       )
       layers = []
       in_channels = features[0]
       for feature in features[1:]:
           layers.append(
               Conv2DDisc(in_channels, feature, stride=1 if feature == features[-1] else 2)
           )
           in_channels = feature
       layers.append(
           nn.Conv2d(
               in_channels,
               1,
               kernel_size=4,
               stride=1,
               padding=1,
               padding_mode="reflect",
           )
       )
       self.model = nn.Sequential(*layers)
   def forward(self, x):
       x = self.initial(x)
       return self.model(x)



    
###############################
# PART 3: TRAINING LOOP
###############################


def train_cyclegan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "training_images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "weights"), exist_ok=True)

    gen_S2R = Generator(img_channels=1, num_features=64, num_residuals=9).to(device)  # Synthetic to Real
    gen_R2S = Generator(img_channels=1, num_features=64, num_residuals=9).to(device)  # Real to Synthetic
    disc_S = Discriminator(in_channels=1).to(device)  # Synthetic discriminator
    disc_R = Discriminator(in_channels=1).to(device)  # Real discriminator

    opt_gen = optim.Adam(
        list(gen_S2R.parameters()) + list(gen_R2S.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    opt_disc = optim.Adam(
        list(disc_S.parameters()) + list(disc_R.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    
    # Learning rate schedulers
    def lambda_lr(epoch):
        if epoch < 100:
            return 1.0  # Constant LR for first 100 epochs
        else:
            # Linear decay from 1.0 to 1e-6/initial_lr over epochs 100-200
            return 1.0 - (1.0 - 1e-6/args.lr) * (epoch - 100) / 100

    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda_lr)
    scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda_lr)
    
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    
    fake_S_buffer = ImageBuffer(50)
    fake_R_buffer = ImageBuffer(50)
    
    if args.resume:
        checkpoint_dir = os.path.join(args.output_dir, "weights")
        latest_checkpoint = os.path.join(checkpoint_dir, "latest_model.pth")
        if os.path.exists(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            gen_S2R.load_state_dict(checkpoint["gen_S2R"])
            gen_R2S.load_state_dict(checkpoint["gen_R2S"])
            disc_S.load_state_dict(checkpoint["disc_S"])
            disc_R.load_state_dict(checkpoint["disc_R"])
            opt_gen.load_state_dict(checkpoint["opt_gen"])
            opt_disc.load_state_dict(checkpoint["opt_disc"])
            
            if "scheduler_gen" in checkpoint and "scheduler_disc" in checkpoint:
                scheduler_gen.load_state_dict(checkpoint["scheduler_gen"])
                scheduler_disc.load_state_dict(checkpoint["scheduler_disc"])
                
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed training from epoch {start_epoch}")
        else:
            start_epoch = 0
            print("No checkpoint found, starting from scratch")
    else:
        start_epoch = 0
    
    train_loader = create_data_loaders(args)
    
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()
    
    log_path = os.path.join(args.output_dir, "training_log.txt")
    with open(log_path, "a") as f:
        f.write(f"Epoch,D_S_Loss,D_R_Loss,G_S2R_Loss,G_R2S_Loss,Cycle_S_Loss,Cycle_R_Loss,Identity_S_Loss,Identity_R_Loss,Total_G_Loss,Total_D_Loss,D_S_Real_Acc,D_S_Fake_Acc,D_R_Real_Acc,D_R_Fake_Acc,Gen_LR,Disc_LR,Time(s)\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        epoch_d_s_loss = 0.0
        epoch_d_r_loss = 0.0
        epoch_g_s2r_loss = 0.0
        epoch_g_r2s_loss = 0.0
        epoch_cycle_s_loss = 0.0
        epoch_cycle_r_loss = 0.0
        epoch_identity_s_loss = 0.0
        epoch_identity_r_loss = 0.0
        epoch_total_g_loss = 0.0
        epoch_total_d_loss = 0.0
        
        d_s_real_correct = 0
        d_s_real_total = 0
        d_s_fake_correct = 0
        d_s_fake_total = 0
        d_r_real_correct = 0
        d_r_real_total = 0
        d_r_fake_correct = 0
        d_r_fake_total = 0
        
        gen_lr = opt_gen.param_groups[0]['lr']
        disc_lr = opt_disc.param_groups[0]['lr']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} | LR: {gen_lr:.6f}")
        
        for idx, (synthetic_img, real_img) in enumerate(pbar):
            batch_size = synthetic_img.size(0)
            synthetic_img = synthetic_img.to(device)
            real_img = real_img.to(device)
            
            ###### Train Discriminators #######
            with torch.amp.autocast(device_type='cuda'):
                # Generate fake images
                fake_real = gen_S2R(synthetic_img)
                fake_synthetic = gen_R2S(real_img)
                
                # Get fake images from buffer
                fake_synthetic_buffer = fake_S_buffer.update_and_get(fake_synthetic.detach())
                fake_real_buffer = fake_R_buffer.update_and_get(fake_real.detach())
                
                # Discriminator S
                D_S_real = disc_S(synthetic_img)
                D_S_fake = disc_S(fake_synthetic_buffer)
                D_S_real_loss = mse_loss(D_S_real, torch.ones_like(D_S_real))
                D_S_fake_loss = mse_loss(D_S_fake, torch.zeros_like(D_S_fake))
                D_S_loss = (D_S_real_loss + D_S_fake_loss)
                
                # Discriminator R
                D_R_real = disc_R(real_img)
                D_R_fake = disc_R(fake_real_buffer)
                D_R_real_loss = mse_loss(D_R_real, torch.ones_like(D_R_real))
                D_R_fake_loss = mse_loss(D_R_fake, torch.zeros_like(D_R_fake))
                D_R_loss = (D_R_real_loss + D_R_fake_loss) 
                
                # Combined discriminator loss
                D_loss = (D_S_loss + D_R_loss) / 2
            
            # Update discriminators
            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
            
            ###### Train Generators ######
            with torch.amp.autocast(device_type='cuda'):
                # Adversarial loss
                D_S_fake = disc_S(fake_synthetic)
                D_R_fake = disc_R(fake_real)
                G_S2R_loss = mse_loss(D_R_fake, torch.ones_like(D_R_fake))
                G_R2S_loss = mse_loss(D_S_fake, torch.ones_like(D_S_fake))
                
                # Cycle consistency loss
                cycle_synthetic = gen_R2S(fake_real)
                cycle_real = gen_S2R(fake_synthetic)
                cycle_S_loss = l1_loss(synthetic_img, cycle_synthetic) * args.lambda_cycle
                cycle_R_loss = l1_loss(real_img, cycle_real) * args.lambda_cycle
                
                # Identity loss (using lambda_identity = 0 for grayscale)
                identity_synthetic = gen_R2S(synthetic_img)
                identity_real = gen_S2R(real_img)
                identity_S_loss = l1_loss(synthetic_img, identity_synthetic) * args.lambda_identity
                identity_R_loss = l1_loss(real_img, identity_real) * args.lambda_identity
                
                # Combined generator loss
                G_loss = (
                    G_S2R_loss + G_R2S_loss + 
                    cycle_S_loss + cycle_R_loss + 
                    identity_S_loss + identity_R_loss
                )
            
            # Update generators
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
            
            # Update metrics
            epoch_d_s_loss += D_S_loss.item() * batch_size
            epoch_d_r_loss += D_R_loss.item() * batch_size
            epoch_g_s2r_loss += G_S2R_loss.item() * batch_size
            epoch_g_r2s_loss += G_R2S_loss.item() * batch_size
            epoch_cycle_s_loss += cycle_S_loss.item() * batch_size
            epoch_cycle_r_loss += cycle_R_loss.item() * batch_size
            epoch_identity_s_loss += identity_S_loss.item() * batch_size
            epoch_identity_r_loss += identity_R_loss.item() * batch_size
            epoch_total_g_loss += G_loss.item() * batch_size
            epoch_total_d_loss += D_loss.item() * batch_size
            
            # Update discriminator accuracy metrics
            d_s_real_correct += ((D_S_real > 0.5).sum().item())
            d_s_real_total += D_S_real.numel()
            d_s_fake_correct += ((D_S_fake < 0.5).sum().item())
            d_s_fake_total += D_S_fake.numel()
            d_r_real_correct += ((D_R_real > 0.5).sum().item())
            d_r_real_total += D_R_real.numel()
            d_r_fake_correct += ((D_R_fake < 0.5).sum().item())
            d_r_fake_total += D_R_fake.numel()
            
            if idx % 100 == 0:
                save_image(
                    fake_real * 0.5 + 0.5, 
                    os.path.join(args.output_dir, "training_images", f"epoch{epoch+1}_{idx}_fake_real.png")
                )
                save_image(
                    fake_synthetic * 0.5 + 0.5, 
                    os.path.join(args.output_dir, "training_images", f"epoch{epoch+1}_{idx}_fake_synthetic.png")
                )
                
            pbar.set_postfix(
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
                D_S_acc=f"{(d_s_real_correct + d_s_fake_correct) / (d_s_real_total + d_s_fake_total):.3f}"
            )
        
        
        scheduler_gen.step()
        scheduler_disc.step()
        
        total_samples = len(train_loader.dataset)
        epoch_d_s_loss /= total_samples
        epoch_d_r_loss /= total_samples
        epoch_g_s2r_loss /= total_samples
        epoch_g_r2s_loss /= total_samples
        epoch_cycle_s_loss /= total_samples
        epoch_cycle_r_loss /= total_samples
        epoch_identity_s_loss /= total_samples
        epoch_identity_r_loss /= total_samples
        epoch_total_g_loss /= total_samples
        epoch_total_d_loss /= total_samples
        
        # Calculate discriminator accuracies
        d_s_real_acc = d_s_real_correct / d_s_real_total if d_s_real_total > 0 else 0
        d_s_fake_acc = d_s_fake_correct / d_s_fake_total if d_s_fake_total > 0 else 0
        d_r_real_acc = d_r_real_correct / d_r_real_total if d_r_real_total > 0 else 0
        d_r_fake_acc = d_r_fake_correct / d_r_fake_total if d_r_fake_total > 0 else 0
        
        epoch_time = time.time() - epoch_start_time
        
        # Get current learning rates for logging
        gen_lr = opt_gen.param_groups[0]['lr']
        disc_lr = opt_disc.param_groups[0]['lr']
        
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{epoch_d_s_loss:.6f},{epoch_d_r_loss:.6f},{epoch_g_s2r_loss:.6f},{epoch_g_r2s_loss:.6f},"
                    f"{epoch_cycle_s_loss:.6f},{epoch_cycle_r_loss:.6f},{epoch_identity_s_loss:.6f},{epoch_identity_r_loss:.6f},"
                    f"{epoch_total_g_loss:.6f},{epoch_total_d_loss:.6f},{d_s_real_acc:.6f},{d_s_fake_acc:.6f},"
                    f"{d_r_real_acc:.6f},{d_r_fake_acc:.6f},{gen_lr:.6f},{disc_lr:.6f},{epoch_time:.2f}\n")
        
        print(f"\nEpoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
        print(f"G_loss: {epoch_total_g_loss:.6f}, D_loss: {epoch_total_d_loss:.6f}")
        print(f"D_S acc: {(d_s_real_acc + d_s_fake_acc)/2:.6f}, D_R acc: {(d_r_real_acc + d_r_fake_acc)/2:.6f}")
        print(f"Learning rates - Gen: {gen_lr:.6f}, Disc: {disc_lr:.6f}")
        
        # Save checkpoint every epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            checkpoint_dir = os.path.join(args.output_dir, "weights")
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch{epoch+1}.pth")
            
            checkpoint = {
                "epoch": epoch,
                "gen_S2R": gen_S2R.state_dict(),
                "gen_R2S": gen_R2S.state_dict(),
                "disc_S": disc_S.state_dict(),
                "disc_R": disc_R.state_dict(),
                "opt_gen": opt_gen.state_dict(),
                "opt_disc": opt_disc.state_dict(),
                "scheduler_gen": scheduler_gen.state_dict(),
                "scheduler_disc": scheduler_disc.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Also save as latest checkpoint for resuming
            torch.save(checkpoint, os.path.join(checkpoint_dir, "latest_models.pth"))
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    print("Training complete!")
    return gen_S2R, gen_R2S

class ImageBuffer:
    """Buffer of previously generated images to reduce model oscillation"""
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        self.buffer = []
        
    def update_and_get(self, images):
        result = []
        for image in images:
            image = torch.unsqueeze(image.detach(), 0)
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(image)
                result.append(image)
            else:
                if torch.rand(1).item() < 0.5:
                    random_id = torch.randint(0, len(self.buffer), (1,)).item()
                    tmp = self.buffer[random_id].clone()
                    self.buffer[random_id] = image
                    result.append(tmp)
                else:
                    result.append(image)
        return torch.cat(result, 0)


###############################
# PART 4: INFERENCE AND IMAGE TRANSLATION
###############################

def translate_images(model, dataloader, target_dir, device, domain="real"):
    """Translate images and save original and translated versions stacked vertically"""
    os.makedirs(target_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (synthetic_img, real_img) in enumerate(tqdm(dataloader, desc=f"Translating to {domain}")):
            batch_size = synthetic_img.shape[0]
            
            for i in range(batch_size):
                dataset_idx = batch_idx * dataloader.batch_size + i
                
                if domain == "real":
                    if dataset_idx >= len(dataloader.dataset.synthetic_images):
                        continue
                        
                    input_img = synthetic_img[i:i+1].to(device)
                    orig_path = dataloader.dataset.synthetic_images[dataset_idx]
                    img_path = os.path.join(dataloader.dataset.synthetic_dir, "images", orig_path)
                    
                    # Keep directory structure relative to base dir
                    rel_dir = os.path.dirname(orig_path)
                    save_dir = os.path.join(target_dir, rel_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, os.path.basename(orig_path))
                else:
                    if dataset_idx >= len(dataloader.dataset.real_images):
                        continue
                        
                    input_img = real_img[i:i+1].to(device)
                    orig_path = dataloader.dataset.real_images[dataset_idx]
                    img_path = os.path.join(dataloader.dataset.real_dir, orig_path)
                    save_path = os.path.join(target_dir, os.path.basename(orig_path))
                
                # Generate translation
                output_img = model(input_img)
                
                output_np = output_img[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
                output_np = np.clip(output_np, 0, 1) * 255
                output_pil = Image.fromarray(output_np.astype(np.uint8).squeeze(), mode="L")
                
                try:
                    
                    original_pil = Image.open(img_path).convert("L")
                    
                    original_pil = original_pil.resize((output_pil.width, output_pil.height), Image.BICUBIC)
                    
                    width = output_pil.width
                    height = output_pil.height * 2
                    stacked_img = Image.new('L', (width, height))
                    stacked_img.paste(original_pil, (0, 0))
                    stacked_img.paste(output_pil, (0, output_pil.height))

                    stacked_img.save(save_path)
                    
                    if (batch_idx * batch_size + i) % 10 == 0:
                        print(f"Stacked comparison saved: {save_path}")
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


def process_test_data(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_S2R = Generator(img_channels=1, num_features=64, num_residuals=9).to(device)
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_dir = os.path.join(args.output_dir, "weights")
        checkpoint_path = os.path.join(checkpoint_dir, "latest_models.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen_S2R.load_state_dict(checkpoint["gen_S2R"])
    print(f"Loaded generator from {checkpoint_path}")

    transforms = get_transforms(mode="translate")
    test_dataset = SatelliteDataset(
        synthetic_dir=args.synthetic_dir,
        real_dir_type=args.real_dir_type,
        transform=transforms,
        multiplier=1.0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    target_dir = args.translated_dir if args.translated_dir else f"cyclegan_{args.real_dir_type}"
    print(f"Translating images to: {target_dir}")

    translate_images(gen_S2R, test_loader, target_dir, device, domain="real")
    
    print(f"Translation complete. Images saved to {target_dir}")
    return target_dir




###############################
# PART 5: MAIN EXECUTION LOGIC
###############################

def main():
    parser = argparse.ArgumentParser(description='CycleGAN for satellite domain adaptation')
    # Basic parameters
    parser.add_argument('--mode', type=str, choices=['train', 'translate'], default='train', help='Operation mode: train or translate')
    parser.add_argument('--synthetic_dir', type=str, required=True, help='Directory containing synthetic satellite images')
    parser.add_argument('--real_dir_type', type=str, choices=['sunlamp', 'lightbox'], required=True, help='Type of real images to use')
    parser.add_argument('--output_dir', type=str, default='runs/cyclegan', help='Output directory for results')
    
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Cycle consistency loss weight')
    parser.add_argument('--lambda_identity', type=float, default=0.0, help='Identity loss weight')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    
    # Translation parameters
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for translation mode')
    parser.add_argument('--translated_dir', type=str, help='Directory to save translated images (default: cyclegan_[real_dir_type])')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate translation quality')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        print(f"Starting CycleGAN training with {args.real_dir_type} real images...")
        print(f"Training for {args.epochs} epochs with lr={args.lr}")
        print(f"Lambda cycle: {args.lambda_cycle}, Lambda identity: {args.lambda_identity}")
        
        gen_S2R, gen_R2S = train_cyclegan(args)
        print(f"Training complete. Models saved in {args.output_dir}/weights")
        
    elif args.mode == 'translate':
        if not args.checkpoint:
            args.checkpoint = os.path.join(args.output_dir, "weights", "latest_models.pth")
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint not found at {args.checkpoint}")
                print("Please specify a valid checkpoint with --checkpoint")
                return
        
        print(f"Translating images using checkpoint: {args.checkpoint}")
        translated_dir = process_test_data(args)
        
        #if args.evaluate:
        #   evaluate_translation_quality(args, translated_dir)

    print(f"CycleGAN processing complete.")

if __name__ == "__main__":
    main()


# Example Usage:

# python cyclegan.py --mode train --synthetic_dir synthetic --real_dir_type sunlamp --output_dir runs/cyclegan_sunlamp
# python cyclegan.py --mode train --synthetic_dir synthetic --real_dir_type sunlamp --output_dir runs/cyclegan_sunlamp --resume

# python cyclegan.py --mode translate --synthetic_dir synthetic --real_dir_type lightbox --output_dir runs/cyclegan_sunlamp --translated_dir cyclegan_sunlamp