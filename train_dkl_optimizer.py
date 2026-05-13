"""
Training script for AdaptiveDKLPowerOptimizer

This script demonstrates how to train the k_base, w_weights, and alpha parameters
using the specified loss function:
    L = 0.5 * SSIM_loss + 50 * L1_loss
    
where L1_loss compares the optimized image with beta * original_image.
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from PIL import Image

from adaptor import AdaptiveDKLPowerOptimizer
from interface import ScreenPowerReductionInterface


class DKLOptimizerTrainer:
    """
    Trainer for AdaptiveDKLPowerOptimizer parameters
    """
    
    def __init__(self, scale, w_weights, alpha=0.5, device='cpu', lr=1e-4):
        """
        Initialize trainer
        
        :param scale: Initial k factors (kL, kRG, kBY)
        :param w_weights: Initial power weights (wR, wG, wB)
        :param alpha: Initial Weber slope
        :param device: 'cpu' or 'cuda'
        :param lr: Learning rate
        """
        self.device = device
        self.lr = lr
        
        # Create trainable optimizer
        self.optimizer = AdaptiveDKLPowerOptimizer(
            scale=scale,
            w_weights=w_weights,
            alpha=alpha,
            device=device,
            trainable=True
        )
        self.optimizer.to(device)
        
        # Create PyTorch optimizer with smaller steps for the reparameterized variables
        self.pytorch_optimizer = optim.AdamW([
            {'params': [self.optimizer.log_scale], 'lr': lr * 0.25},
            {'params': [self.optimizer.w_weights], 'lr': lr * 0.10},
            {'params': [self.optimizer.alpha], 'lr': lr * 0.05},
        ], betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
        
        self.training_losses = []
    
    def load_image(self, image_path, device):
        """
        Load image from file
        
        :param image_path: Path to image file
        :param device: PyTorch device
        :return: Normalized tensor (H, W, 3) in [0, 1]
        """
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img) / 255.0
        img_tensor = torch.from_numpy(img_np).float().to(device)
        return img_tensor
    
    def train_epoch(self, image_batch, beta=0.3):
        """
        Train for one epoch on a batch of images
        
        :param image_batch: List of image tensors or paths
        :param beta: Display optimization degree (default 0.3 = 30%)
        :return: Average loss for the epoch
        """
        epoch_losses = []
        
        for idx, image_ori in enumerate(image_batch):
            # Load image if it's a path
            if isinstance(image_ori, str):
                image_ori = self.load_image(image_ori, self.device)
            
            # Ensure image is in valid range
            image_ori = torch.clamp(image_ori, 0.0, 1.0)
            
            # Single training step
            loss = self.optimizer.train_step(
                self.pytorch_optimizer,
                image_ori,
                beta=beta
            )
            
            epoch_losses.append(loss)
            
            if (idx + 1) % max(1, len(image_batch) // 10) == 0:
                print(f"  Batch {idx+1}/{len(image_batch)}, Loss: {loss:.6f}")
        
        avg_loss = np.mean(epoch_losses)
        
        self.training_losses.append(avg_loss)
        return avg_loss
    
    def train(self, image_paths, num_epochs=10, beta=0.3):
        """
        Train the optimizer for multiple epochs
        
        :param image_paths: List of image file paths or tensors
        :param num_epochs: Number of training epochs
        :param beta: Display optimization degree
        """
        print("Starting training...")
        print(f"Initial scale: {self.optimizer.scale_tensor.data}")
        print(f"Initial w_weights: {self.optimizer.w_weights.data}")
        print(f"Initial alpha: {self.optimizer.alpha.data}")
        print(f"Beta (optimization degree): {beta:.1%}")
        print()
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(image_paths, beta=beta)
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")
        
        print("\nTraining completed!")
        print(f"Final scale: {self.optimizer.scale_tensor.data}")
        print(f"Final w_weights: {self.optimizer.w_weights.data}")
        print(f"Final alpha: {self.optimizer.alpha.data}")
    
    def evaluate(self, image_paths, beta=0.3):
        """
        Evaluate the optimizer on test images
        
        :param image_paths: List of test image paths
        :param beta: Display optimization degree
        :return: Average loss and average power reduction
        """
        losses = []
        power_reductions = []
        
        self.optimizer.eval()
        with torch.no_grad():
            for image_path in image_paths:
                image_ori = self.load_image(image_path, self.device)
                image_opt = self.optimizer.forward_torch(image_ori)
                loss = self.optimizer.compute_loss(image_opt, image_ori, beta=beta)
                losses.append(loss.item())
                
                # Compute power reduction (simplified)
                w_rgb = self.optimizer.w_weights.cpu().numpy()
                ori_power = w_rgb @ image_ori.mean(dim=(0, 1)).cpu().numpy()
                opt_power = w_rgb @ image_opt.mean(dim=(0, 1)).cpu().numpy()
                reduction = (ori_power - opt_power) / ori_power * 100
                power_reductions.append(reduction)
        
        self.optimizer.train()
        
        avg_loss = np.mean(losses)
        avg_reduction = np.mean(power_reductions)
        
        print(f"Evaluation - Avg Loss: {avg_loss:.6f}, Avg Power Reduction: {avg_reduction:.2f}%")
        return avg_loss, avg_reduction
    
    def save_checkpoint(self, checkpoint_path):
        """Save model checkpoint"""
        torch.save({
            'k_base': self.optimizer.scale_tensor.data,
            'w_weights': self.optimizer.w_weights.data,
            'alpha': self.optimizer.alpha.data,
            'optimizer_state': self.pytorch_optimizer.state_dict(),
            'losses': self.training_losses
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.optimizer.set_parameters(
            scale=checkpoint['k_base'],
            w_weights=checkpoint['w_weights'],
            alpha=checkpoint['alpha']
        )
        self.pytorch_optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_losses = checkpoint['losses']
        print(f"Checkpoint loaded from {checkpoint_path}")


def main():
    """Example training script"""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Initial parameters
    # k_base = (1.0, 1.5, 4.0)
    k_base = (0.003, 0.003, 0.001)
    w_weights = (2.3153, 2.4567, 5.3075)
    alpha = 0.5
    
    # Create trainer
    trainer = DKLOptimizerTrainer(
        scale=k_base,
        w_weights=w_weights,
        alpha=alpha,
        device=device,
        lr=1e-6
    )
    
    # Load training data (example)
    dataset_path = Path("datasets/genshin_impact")
    if dataset_path.exists():
        # Get list of images
        supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        image_paths = [
            str(f) for f in dataset_path.iterdir()
            if f.suffix.lower() in supported_formats
        ]
        
        if not image_paths:
            print("No images found in dataset directory.")
            print("Creating synthetic training data instead...")
            # Create synthetic training images
            image_batch = [
                torch.rand(512, 512, 3).to(device)
                for _ in range(5)
            ]
        else:
            print(f"Found {len(image_paths)} training images")
            image_batch = image_paths[:10]  # Use first 10 images
    else:
        print("Dataset directory not found. Creating synthetic training data...")
        # Create synthetic training images
        image_batch = [
            torch.rand(512, 512, 3).to(device)
            for _ in range(5)
        ]
    
    # Training
    beta = 0.8  # 30% optimization degree
    trainer.train(image_batch, num_epochs=10, beta=beta)
    
    # Save trained model
    checkpoint_path = "output/dkl_optimizer_checkpoint.pth"
    trainer.save_checkpoint(checkpoint_path)
    
    # # Evaluation (if you have test data)
    # if isinstance(image_batch[0], str):
    #     print("\nEvaluating on training data...")
    #     trainer.evaluate(image_batch[:3], beta=beta)


if __name__ == "__main__":
    main()
