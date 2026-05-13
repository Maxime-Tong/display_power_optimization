# AdaptiveDKLPowerOptimizer - Training Guide

## Overview

The `AdaptiveDKLPowerOptimizer` class has been enhanced to support training of its core parameters (`k_base`, `w_weights`, `alpha`) using PyTorch and image-based loss functions.

## Key Changes

### 1. **PyTorch Integration**
- The class now inherits from both `BaseScreenAdaptor` and `torch.nn.Module`
- Parameters can be converted to `torch.nn.Parameter` for gradient-based optimization
- Dual mode operation: inference mode (fast numpy LUT) and training mode (differentiable torch)

### 2. **Trainable Parameters**
When `trainable=True`, the following become trainable:
- **k_base**: Perceptual thresholds (kL, kRG, kBY) that scale with luminance
- **w_weights**: RGB channel power weights (wR, wG, wB)
- **alpha**: Weber's law slope controlling luminance sensitivity

### 3. **Loss Function**
The implemented loss function is:
```
L = 0.5 * SSIM_loss + 50 * L1_loss

where:
  SSIM_loss = 1 - SSIM(I_opt, I_ori)
  L1_loss = ||I_opt - β * I_ori||₁
  β = optimization degree (default 0.3 = 30%)
```

This loss balances:
- **SSIM**: Structural similarity (perceptual quality preservation)
- **L1**: Magnitude of optimization (power reduction)

## Usage Examples

### Inference Mode (No Training)

```python
from adaptor import AdaptiveDKLPowerOptimizer
import numpy as np

# Create optimizer (non-trainable, faster inference)
optimizer = AdaptiveDKLPowerOptimizer(
    k_base=(1.0, 1.5, 4.0),
    w_weights=(0.3, 0.4, 0.8),
    trainable=False  # Default
)

# Process image (uses fast LUT-based inference)
input_image = np.random.rand(512, 512, 3).astype(np.float32)
output_image = optimizer.apply(input_image)
```

### Training Mode

#### Basic Training

```python
import torch
from adaptor import AdaptiveDKLPowerOptimizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create trainable optimizer
optimizer = AdaptiveDKLPowerOptimizer(
    k_base=(1.0, 1.5, 4.0),
    w_weights=(0.3, 0.4, 0.8),
    alpha=0.5,
    device=device,
    trainable=True
)
optimizer.to(device)

# Create PyTorch optimizer
pytorch_optimizer = torch.optim.Adam(
    optimizer.get_trainable_parameters(),
    lr=1e-4
)

# Load training image
image_ori = torch.randn(512, 512, 3).to(device)

# Training loop
for epoch in range(num_epochs):
    loss = optimizer.train_step(pytorch_optimizer, image_ori, beta=0.3)
    print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
```

#### Using the Training Script

```bash
python train_dkl_optimizer.py
```

This runs a complete training pipeline with:
- Batch training on dataset images
- Checkpoint saving/loading
- Evaluation on test data

### Advanced Usage

#### Custom Training Loop

```python
# Forward pass
image_opt = optimizer.forward_torch(image_ori)

# Compute loss
loss = optimizer.compute_loss(image_opt, image_ori, beta=0.3)

# Manual backward pass
loss.backward()
pytorch_optimizer.step()
```

#### Batch Processing

```python
# Train on multiple images
image_batch = [image1, image2, image3]  # List of tensors

for epoch in range(num_epochs):
    for image_ori in image_batch:
        loss = optimizer.train_step(pytorch_optimizer, image_ori, beta=0.3)
```

#### Saving and Loading

```python
# Save checkpoint
torch.save({
    'k_base': optimizer.k_base.data,
    'w_weights': optimizer.w_weights.data,
    'alpha': optimizer.alpha.data
}, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
optimizer.k_base.data = checkpoint['k_base']
optimizer.w_weights.data = checkpoint['w_weights']
optimizer.alpha.data = checkpoint['alpha']
```

## Loss Function Details

### SSIM Loss Component (Weight: 0.5)
- Computes structural similarity between optimized and original images
- Ensures perceptual quality is preserved
- Implemented using Gaussian-weighted convolution

### L1 Loss Component (Weight: 50)
- Measures difference between optimized image and scaled original
- The scaling factor `β` controls optimization aggressiveness:
  - `β = 0.1`: 10% optimization (conservative)
  - `β = 0.3`: 30% optimization (default, balanced)
  - `β = 0.5`: 50% optimization (aggressive)

## Parameters Explanation

### k_base (tuple of 3 floats)
Perceptual thresholds in DKL space:
- **kL**: Luminance threshold
- **kRG**: L-M (red-green) threshold
- **kBY**: S cone (blue-yellow) threshold

Typical values: `(1.0, 1.5, 4.0)`

### w_weights (tuple of 3 floats)
Relative power consumption of RGB channels:
- Reflects the power cost of displaying each color
- Blue typically has 2-3x higher power cost than red/green
- Typical values: `(0.3, 0.4, 0.8)`

### alpha (float, 0.4-0.6)
Weber's law slope parameter:
- Controls how perceptual thresholds scale with luminance
- Higher values: thresholds increase more with brightness
- Typical value: `0.5`

## Model Architecture

The optimizer uses:
- **Closed-form Lagrange multiplier solution** for optimal deltas
- **Trilinear interpolation** (LUT) for fast inference
- **Differentiable PyTorch operations** for gradient computation

## Training Tips

1. **Learning Rate**: Start with `1e-4` to `1e-3`
   - Reduce if loss oscillates
   - Increase if convergence is slow

2. **Beta (Optimization Degree)**:
   - `0.2-0.3`: Balances quality and power reduction
   - Adjust based on application requirements

3. **Batch Size**: Train on diverse images for robustness
   - At least 5-10 images per epoch
   - Mix different image types (dark, bright, colorful, etc.)

4. **Monitoring**:
   - Track SSIM loss and L1 loss separately
   - Validate on unseen test images
   - Monitor actual power reduction percentage

5. **Convergence**:
   - Typical convergence in 10-50 epochs
   - May plateau early, continue training often helps
   - Use early stopping if validation loss increases

## Integration with Existing Pipeline

### Using Trained Weights with Inference Pipeline

```python
from interface import ScreenPowerReductionInterface
from adaptor import EllipseAdaptor, AdaptiveDKLPowerOptimizer

# Create trainable optimizer and train...
# (see training examples above)

# Convert trained parameters to inference mode
inference_optimizer = AdaptiveDKLPowerOptimizer(
    k_base=trained_optimizer.k_base.detach().cpu().numpy(),
    w_weights=trained_optimizer.w_weights.detach().cpu().numpy(),
    alpha=trained_optimizer.alpha.detach().cpu().numpy().item(),
    trainable=False  # Inference mode
)

# Use in pipeline
interface = ScreenPowerReductionInterface(
    dataset_path="datasets/genshin_impact",
    screen_adaptor=ScreenAdaptorPipeline([inference_optimizer])
)

results = interface.process_all_images()
```

## Performance Considerations

### Training Efficiency
- **GPU**: ~10-20x faster than CPU
- **Memory**: ~500MB for 512x512 images
- **Time**: ~1-5 seconds per image per epoch

### Inference Efficiency
After training, use non-trainable mode:
- **Speed**: ~50-100ms per 1080p image on CPU
- **Memory**: ~50MB
- **Method**: Fast LUT interpolation (no gradient computation)

## Troubleshooting

### Loss Not Decreasing
- Increase learning rate
- Check that parameters are actually changing
- Verify loss computation is correct

### NaN or Inf in Loss
- Check for division by zero
- Reduce learning rate
- Ensure input images are properly normalized [0, 1]

### Gradient Explosion
- Use gradient clipping
- Reduce learning rate
- Check parameter initialization

### Memory Issues
- Reduce image resolution
- Use smaller batch sizes
- Switch to CPU if needed

## Future Enhancements

- [ ] Multi-GPU training support
- [ ] Data augmentation strategies
- [ ] Perceptual metric alternatives to SSIM
- [ ] Per-image adaptive beta
- [ ] Real-time training on video streams

## References

The implementation is based on:
- **DKL Color Space**: Derrington, Krauskopf & Lennie (1984)
- **JND Ellipsoids**: Visual discrimination thresholds
- **Weber's Law**: Perceptual sensitivity scaling with luminance
