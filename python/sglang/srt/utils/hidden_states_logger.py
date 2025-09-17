"""
Hidden states logger for debugging deterministic behavior across different batch sizes.
"""

import os
import torch
import hashlib
from typing import Optional
from datetime import datetime


class HiddenStatesLogger:
    def __init__(self):
        self.enabled = os.getenv("SGLANG_LOG_HIDDEN_STATES", "false").lower() == "true"
        self.log_dir = os.getenv("SGLANG_HIDDEN_STATES_LOG_DIR", "./hidden_states_logs")
        self.max_tokens_to_log = int(os.getenv("SGLANG_MAX_TOKENS_TO_LOG", "10"))  # Only log first N tokens
        self.max_dims_to_log = int(os.getenv("SGLANG_MAX_DIMS_TO_LOG", "20"))     # Only log first N dimensions
        
        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Hidden states logging enabled. Session ID: {self.session_id}")
            print(f"Log directory: {self.log_dir}")
            print(f"Max tokens to log: {self.max_tokens_to_log}")
            print(f"Max dimensions to log: {self.max_dims_to_log}")
    
    def log_hidden_states(self, 
                         hidden_states: torch.Tensor, 
                         layer_id: int, 
                         stage: str = "output",
                         batch_size: Optional[int] = None,
                         extra_info: str = ""):
        """
        Log hidden states to file for debugging.
        
        Args:
            hidden_states: The tensor to log [batch_size, seq_len, hidden_dim] or [total_tokens, hidden_dim]
            layer_id: The layer number (0-based)
            stage: Stage of the layer ("input", "attention_output", "output")
            batch_size: Batch size if known
            extra_info: Additional information to include in the log
        """
        if not self.enabled:
            return
            
        try:
            # Get tensor info
            shape = hidden_states.shape
            dtype = hidden_states.dtype
            device = hidden_states.device
            
            # Determine actual batch size
            if batch_size is None:
                if len(shape) == 3:  # [batch_size, seq_len, hidden_dim]
                    batch_size = shape[0]
                elif len(shape) == 2:  # [total_tokens, hidden_dim] - packed format
                    batch_size = "packed"
                else:
                    batch_size = "unknown"
            
            # Create filename
            filename = f"layer_{layer_id:02d}_{stage}_bs{batch_size}_{self.session_id}.log"
            filepath = os.path.join(self.log_dir, filename)
            
            # Convert to CPU for logging
            hidden_states_cpu = hidden_states.detach().cpu()
            
            # Truncate for logging (only log a subset to avoid huge files)
            if len(shape) == 3:  # [batch_size, seq_len, hidden_dim]
                logged_tensor = hidden_states_cpu[:, :self.max_tokens_to_log, :self.max_dims_to_log]
            elif len(shape) == 2:  # [total_tokens, hidden_dim]
                logged_tensor = hidden_states_cpu[:self.max_tokens_to_log, :self.max_dims_to_log]
            else:
                logged_tensor = hidden_states_cpu
            
            # Calculate hash for quick comparison
            tensor_hash = hashlib.md5(hidden_states_cpu.numpy().tobytes()).hexdigest()[:16]
            
            # Log to file
            with open(filepath, "a") as f:
                f.write(f"=== Layer {layer_id} - {stage} ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Shape: {shape}\n")
                f.write(f"Dtype: {dtype}\n")
                f.write(f"Device: {device}\n")
                f.write(f"Batch size: {batch_size}\n")
                f.write(f"Hash (first 16 chars): {tensor_hash}\n")
                if extra_info:
                    f.write(f"Extra info: {extra_info}\n")
                f.write(f"Tensor stats:\n")
                f.write(f"  Mean: {hidden_states_cpu.mean().item():.6f}\n")
                f.write(f"  Std: {hidden_states_cpu.std().item():.6f}\n")
                f.write(f"  Min: {hidden_states_cpu.min().item():.6f}\n")
                f.write(f"  Max: {hidden_states_cpu.max().item():.6f}\n")
                f.write(f"Sample values (first {self.max_tokens_to_log} tokens, first {self.max_dims_to_log} dims):\n")
                f.write(f"{logged_tensor}\n")
                f.write("=" * 50 + "\n\n")
                
        except Exception as e:
            print(f"Error logging hidden states for layer {layer_id}: {e}")
    
    def log_batch_info(self, batch_info: str):
        """Log batch information to a separate file."""
        if not self.enabled:
            return
            
        try:
            filename = f"batch_info_{self.session_id}.log"
            filepath = os.path.join(self.log_dir, filename)
            
            with open(filepath, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] {batch_info}\n")
        except Exception as e:
            print(f"Error logging batch info: {e}")


# Global logger instance
hidden_states_logger = HiddenStatesLogger()
