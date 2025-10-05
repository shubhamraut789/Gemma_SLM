import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext
import os
import random
import math
import logging

# Clear GPU memory at start
torch.cuda.empty_cache()
gc.collect()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Kaggle dataset paths
DATASET_PATH = '/kaggle/input/llm-input'
TRAIN_BIN_PATH = f'{DATASET_PATH}/train.bin'
VAL_BIN_PATH = f'{DATASET_PATH}/val.bin'
TOKENIZER_PATH = f'{DATASET_PATH}/multi_lang_spm.model'

# Checkpoint management - use directory instead of single file
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_latest_checkpoint():
    """Find the latest checkpoint file"""
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('gemma3_')]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(CHECKPOINT_DIR, latest)

print("Checking dataset files...")
files_to_check = {
    "Train binary": TRAIN_BIN_PATH,
    "Val binary": VAL_BIN_PATH, 
    "Tokenizer": TOKENIZER_PATH,
}

try:
    for name, path in files_to_check.items():
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"✓ {name}: {size_gb:.2f} GB")
        else:
            raise FileNotFoundError(f"✗ {name}: Not found at {path}")
            
    # Load and test tokenizer with error handling
    print("\nTesting tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.load(TOKENIZER_PATH):
        raise Exception("Failed to load tokenizer")
        
    vocab_size = tokenizer.get_piece_size()
    print(f"Tokenizer loaded: {vocab_size} vocab size")
    
    # Test multilingual tokenization
    test_texts = [
        "Hello world how are you",
        "नमस्ते मित्रा कसे आहात", 
        "आज मौसम खूप सुंदर आहे"
    ]
    
    print("\nTokenizer test:")
    for text in test_texts:
        tokens = tokenizer.encode_as_ids(text)
        decoded = tokenizer.decode_ids(tokens)
        print(f"'{text}' -> {len(tokens)} tokens -> '{decoded}'")
        
    # Check for existing checkpoints
    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
        print(f"\nFound existing checkpoint: {latest_checkpoint}")
    else:
        print("\nNo existing checkpoints found - will start fresh")
        
except Exception as e:
    logger.error(f"Setup failed: {e}")
    raise

# ===== BLOCK 2: MODEL ARCHITECTURE COMPONENTS =====

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return x_norm * self.scale

def compute_rope_params(head_dim, theta_base=10_000, context_length=512, dtype=torch.float32):
    # Add at start of compute_rope_params
    assert head_dim % 2 == 0, "head_dim must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[:head_dim//2].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    x1 = x[..., :head_dim//2]
    x2 = x[..., head_dim//2:]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype).contiguous()

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=64, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        
        self.scaling = head_dim ** -0.5

    def forward(self, x, mask, cos, sin):
        b, seq_len, _ = x.shape

        q = self.W_query(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        q = q * self.scaling

        attn_scores = q @ k.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = (attn_weights @ v).transpose(1, 2).reshape(b, seq_len, self.d_out)
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.gate = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_gate = self.gate(x)
        x = F.gelu(x_fc1, approximate="tanh") * x_gate
        return self.fc3(x)

print("Model components defined successfully")

# ===== BLOCK 3 REVISED: FIXED MODEL CONFIG =====

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"], 
            num_heads=cfg["n_heads"], 
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"], 
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])  # Pre-attention norm
        self.norm2 = RMSNorm(cfg["emb_dim"])  # Pre-feedforward norm

    def forward(self, x, mask, cos, sin):
        # Pre-norm attention
        x = x + self.att(self.norm1(x), mask, cos, sin)
        # Pre-norm feedforward  
        x = x + self.ff(self.norm2(x))
        return x

class Gemma3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        # Fix loop variable in blocks creation:
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        self.cfg = cfg

        # Precompute RoPE parameters
        cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _create_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids)
        
        mask = self._create_causal_mask(seq_len, x.device)

        for block in self.blocks:
            x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

# REDUCED CONFIG - Targets ~105M parameters
GEMMA3_CONFIG_110M = {
    "vocab_size": 32000,
    "context_length": 512,
    "emb_dim": 704,        # Changed from 640 to 704
    "hidden_dim": 1832,    # Changed from 1664 to 1832 (704 * 2.6)
    "n_heads": 12,         # Changed to 12
    "head_dim": 58,        # 704 / 12
    "n_layers": 12,        
    "n_kv_groups": 3,      # 12 / 3 = 4 heads per group
    "rope_base": 10_000.0,
    "dtype": torch.bfloat16,
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test reduced model
print("Testing reduced model...")
model = Gemma3Model(GEMMA3_CONFIG_110M)
param_count = count_parameters(model)
print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")

if param_count > 110e6:
    print("Still too large - reducing further...")
elif param_count < 100e6:
    print("Too small - increasing...")
else:
    print("Model parameter count is within target range!")

# ===== BLOCK 4: EFFICIENT BINARY DATA LOADER =====

class BinaryDataLoader:
    def __init__(self, bin_path, block_size=512, buffer_size=100000):
        self.bin_path = bin_path
        self.block_size = block_size
        self.buffer_size = buffer_size
        
        # Load entire binary file into memory (it's only ~5GB, should fit)
        print(f"Loading binary data from {bin_path}...")
        self.data = np.fromfile(bin_path, dtype=np.uint16)
        gc.collect()
        self.total_tokens = len(self.data)
        print(f"Loaded {self.total_tokens:,} tokens ({self.total_tokens/1e9:.2f}B)")
        
        self.current_pos = 0
    
    def get_batch(self, batch_size, device='cpu'):
        sequences = []
        targets = []
        
        for _ in range(batch_size):
            # Get next sequence of block_size tokens
            if self.current_pos + batch_size * self.block_size >= self.total_tokens:
                self.current_pos = 0  # Wrap around to beginning
            
            # Extract input sequence and target (shifted by 1)
            seq = self.data[self.current_pos:self.current_pos + self.block_size]
            target = self.data[self.current_pos + 1:self.current_pos + self.block_size + 1]
            
            sequences.append(seq)
            targets.append(target)
            
            # Move position forward randomly to avoid overfitting to sequence order
            self.current_pos += random.randint(1, self.block_size // 2)
        
        # Convert to tensors
        x = torch.tensor(np.array(sequences), dtype=torch.long)
        y = torch.tensor(np.array(targets), dtype=torch.long)

        if device != 'cpu':
            x = x.to(device)
            y = y.to(device)
        return x, y
    
    def get_progress(self):
        return self.current_pos / self.total_tokens

# Initialize data loaders
print("Initializing data loaders...")
train_loader = BinaryDataLoader(TRAIN_BIN_PATH, block_size=512)
val_loader = BinaryDataLoader(VAL_BIN_PATH, block_size=512)

print(f"Train tokens: {train_loader.total_tokens:,}")
print(f"Val tokens: {val_loader.total_tokens:,}")

# Test batch loading
print("\nTesting batch loading...")
x_test, y_test = train_loader.get_batch(2)
print(f"Batch shape: {x_test.shape}")
print(f"Sample tokens: {x_test[0][:10].tolist()}")

# Test tokenizer decoding
sample_text = tokenizer.decode_ids(x_test[0][:20].tolist())
print(f"Decoded sample: '{sample_text}'")

# ===== BLOCK 5: TRAINING SETUP =====

# Training hyperparameters
TRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 12,
    "gradient_accumulation_steps": 16,  # Effective batch = 192
    "max_steps": 30000,
    "warmup_steps": 1500,
    "eval_interval": 2000,
    "save_interval": 2000,
    "min_lr": 3e-5,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "block_size": 512,
}

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

print(f"Device: {device}")
print(f"Using dtype: {dtype}")
print(f"Effective batch size: {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")

# Initialize model
print("\nInitializing model...")
model = Gemma3Model(GEMMA3_CONFIG_110M)
model = model.to(device)
param_count = count_parameters(model)
print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
torch.cuda.empty_cache()
gc.collect()

# Initialize optimizer and scheduler
from torch.optim.lr_scheduler import OneCycleLR

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=TRAINING_CONFIG["learning_rate"],
    betas=(0.9, 0.95),
    weight_decay=TRAINING_CONFIG["weight_decay"],
    eps=1e-8
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=TRAINING_CONFIG["learning_rate"],
    total_steps=TRAINING_CONFIG["max_steps"],
    pct_start=TRAINING_CONFIG["warmup_steps"] / TRAINING_CONFIG["max_steps"],
    anneal_strategy='cos',
    div_factor=TRAINING_CONFIG["learning_rate"] / TRAINING_CONFIG["min_lr"]
)

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

print("Training setup complete!")
print(f"Will train for {TRAINING_CONFIG['max_steps']:,} steps")
print(f"Evaluation every {TRAINING_CONFIG['eval_interval']:,} steps")
print(f"Saving every {TRAINING_CONFIG['save_interval']:,} steps")

# Add checkpoint saving function that only saves .pt files:
def save_checkpoint(model, optimizer, scheduler, step, loss, is_best=False):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': loss,
    }
    
    # Regular checkpoint every 2000 steps
    checkpoint_path = f'{CHECKPOINT_DIR}/gemma3_step_{step}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Best checkpoint
    if is_best:
        best_path = f'{CHECKPOINT_DIR}/gemma3_best.pt'
        torch.save(checkpoint, best_path)
    
    return checkpoint_path

# ===== BLOCK 7: MAIN TRAINING LOOP =====
def train_model():
    print("=" * 70)
    print("STARTING GEMMA3 TRAINING")
    print("=" * 70)
    
    step = 0
    best_val_loss = float('inf')
    running_loss = 0.0
    current_val_loss = float('inf')  # Track current validation loss
    
    # Training loop
    model.train()
    pbar = tqdm(total=TRAINING_CONFIG["max_steps"], desc="Training")
    
    try:
        while step < TRAINING_CONFIG["max_steps"]:
            # Get batch
            X, Y = get_batch('train', TRAINING_CONFIG["batch_size"])
            
            # Training step
            loss = train_step(X, Y, step)
            running_loss += loss
            
            # Gradient accumulation and optimizer step
            if (step + 1) % TRAINING_CONFIG["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                avg_loss = running_loss / TRAINING_CONFIG["gradient_accumulation_steps"]
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                running_loss = 0.0
            
            # Evaluation
            if step % TRAINING_CONFIG["eval_interval"] == 0 and step > 0:
                print(f"\nStep {step} - Evaluating...")
                losses = estimate_loss()
                current_val_loss = float(losses['val'])  # Ensure it's a Python float
                train_progress = train_loader.get_progress() * 100
                
                print(f"Step {step:,}")
                print(f"Train loss: {losses['train']:.4f}")
                print(f"Val loss: {current_val_loss:.4f}")
                print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
                print(f"Data progress: {train_progress:.2f}%")
                
                # Save best model
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    save_checkpoint_persistent(step, current_val_loss, is_best=True)
            
            # Regular checkpoint saving
            if step % TRAINING_CONFIG["save_interval"] == 0 and step > 0:
                # Use the most recent validation loss, or current training loss if no eval yet
                checkpoint_loss = current_val_loss if current_val_loss != float('inf') else float(loss)
                save_checkpoint_persistent(step, checkpoint_loss, is_best=False)
            
            # Memory cleanup
            if step % 1000 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            step += 1
            pbar.update(1)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final checkpoint - ensure we pass Python float, not tensor
        final_loss = float(best_val_loss) if best_val_loss != float('inf') else float(current_val_loss)
        save_checkpoint_persistent(step, final_loss, is_best=False)
        
        # At the very end of training
        save_final_outputs()
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Total steps: {step:,}")
        pbar.close()

# Simplify checkpoint saving - remove JSON metadata:
def save_checkpoint_persistent(step, loss, is_best=False):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': float(loss),
    }
    
    if is_best:
        model_path = '/kaggle/working/best_gemma3_model.pt'
    else:
        model_path = f'/kaggle/working/checkpoint_step_{step}.pt'
    
    torch.save(checkpoint, model_path)
    print(f"Saved: {model_path}")

print("Ready to start training!")
print("Run: train_model()")
train_model()
