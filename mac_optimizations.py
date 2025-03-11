import torch
import platform
import os
import gc
from functools import wraps
import numpy as np
import time

class MacOptimizer:
    """
    Helper class for Apple Silicon GPU (MPS) optimizations in PyTorch.
    
    This class provides methods to optimize training on M1/M2/M3 Macs with 
    Metal Performance Shaders (MPS) support.
    """
    
    def __init__(self, memory_threshold=0.8):
        """
        Initialize the optimizer.
        
        Args:
            memory_threshold: Fraction of GPU memory to use before cleaning up
        """
        self.is_mps_available = self._check_mps_availability()
        self.device = self._get_device()
        self.memory_threshold = memory_threshold
        
        # Only relevant for MPS
        self.last_memory_cleanup = time.time()
        self.cleanup_interval = 30  # seconds
        
        if self.is_mps_available:
            print("🍎 MPS (Metal Performance Shaders) is available!")
            print(f"🎯 Using device: {self.device}")
        else:
            print(f"Using device: {self.device}")
    
    def _check_mps_availability(self):
        """Check if MPS is available."""
        if not torch.backends.mps.is_available():
            return False
        
        # On older PyTorch versions, is_available might return True but MPS isn't fully implemented
        try:
            # Try creating a small tensor on MPS to verify it actually works
            torch.ones(1, device="mps")
            return True
        except:
            return False
    
    def _get_device(self):
        """Get the appropriate device."""
        if self.is_mps_available:
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def optimize_model(self, model):
        """
        Apply model-specific optimizations.
        
        Args:
            model: PyTorch model
        
        Returns:
            Optimized model
        """
        model = model.to(self.device)
        
        if self.is_mps_available:
            # Specific MPS optimizations for model parameters
            for param in model.parameters():
                if param.requires_grad:
                    # Ensure contiguous tensors for better MPS performance
                    if not param.is_contiguous():
                        param.data = param.data.contiguous()
        
        return model
    
    def optimize_dataloader(self, dataloader, num_workers=None, prefetch_factor=None):
        """
        Return optimized dataloader settings for Mac.
        
        Args:
            dataloader: PyTorch DataLoader
            num_workers: Number of workers (if None, will be auto-set)
            prefetch_factor: Prefetch factor (if None, will be auto-set)
        
        Returns:
            Dictionary of optimized dataloader settings
        """
        is_mac = platform.system() == 'Darwin'
        
        if not is_mac:
            # Non-Mac systems - use standard settings
            if num_workers is None:
                num_workers = min(os.cpu_count(), 8)
            return {'num_workers': num_workers, 'pin_memory': True}
        
        # Mac-specific optimizations
        if self.is_mps_available:
            # For Apple Silicon with MPS
            # Avoid too many workers which can cause issues with MPS
            if num_workers is None:
                num_workers = min(2, os.cpu_count() or 1)
            
            # Prefetch factor tuned for MPS
            if prefetch_factor is None:
                prefetch_factor = 2
            
            return {
                'num_workers': num_workers,
                'pin_memory': False,  # Pin memory can cause issues with MPS
                'prefetch_factor': prefetch_factor,
                'persistent_workers': num_workers > 0
            }
        else:
            # For Intel Macs without MPS
            if num_workers is None:
                num_workers = min(os.cpu_count() or 1, 4)
            return {'num_workers': num_workers, 'pin_memory': False}
    
    def clear_memory(self, force=False):
        """
        Clear GPU memory cache if on MPS.
        
        Args:
            force: Force cleanup even if interval hasn't elapsed
        """
        if not self.is_mps_available:
            return
        
        current_time = time.time()
        if force or (current_time - self.last_memory_cleanup > self.cleanup_interval):
            # Manually trigger garbage collection
            gc.collect()
            
            # Empty CUDA cache
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            self.last_memory_cleanup = current_time
    
    def to_device(self, data):
        """
        Move data to the device in an optimized way.
        
        Args:
            data: PyTorch tensor or collection of tensors
        
        Returns:
            Data on the correct device
        """
        if isinstance(data, (list, tuple)) and len(data) > 0:
            return [self.to_device(x) for x in data]
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            # For MPS, ensure data is contiguous
            if self.is_mps_available and not data.is_contiguous():
                data = data.contiguous()
            return data.to(self.device, non_blocking=True)
        return data
    
    def optimize_training_loop(self, train_fn):
        """
        Decorator to optimize a training loop function.
        
        Args:
            train_fn: Training function to optimize
        
        Returns:
            Optimized training function
        """
        @wraps(train_fn)
        def optimized_training(*args, **kwargs):
            # Apply device-specific setup
            if self.is_mps_available:
                # Set specific environment variables for MPS
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            # Run the training function
            result = train_fn(*args, **kwargs)
            
            # Clean up after training
            self.clear_memory(force=True)
            
            return result
        
        return optimized_training


# Example of how to modify your training loop with these optimizations
def example_training_loop_with_mac_optimizations():
    # Initialize optimizer
    mac_opt = MacOptimizer()
    
    # Get device
    device = mac_opt.device
    
    # Optimize dataloader
    dataloader_kwargs = mac_opt.optimize_dataloader(None)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        **dataloader_kwargs
    )
    
    # Initialize and optimize model
    model = YourModel()
    model = mac_opt.optimize_model(model)
    
    # Training loop with optimizations
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_dataloader):
            # Move data to device in an optimized way
            data, target = mac_opt.to_device((data, target))
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Periodically clear memory
            if batch_idx % 10 == 0:
                mac_opt.clear_memory()
        
        # Clear memory at the end of each epoch
        mac_opt.clear_memory(force=True)