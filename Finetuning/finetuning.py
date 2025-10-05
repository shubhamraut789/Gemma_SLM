import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import sentencepiece as spm
from tqdm import tqdm
import os
import random
import numpy as np
from typing import List, Dict, Tuple
import math
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMMA3_CONFIG_110M = {
    "vocab_size": 32000,
    "context_length": 512,
    "emb_dim": 704,
    "hidden_dim": 1832,
    "n_heads": 12,
    "head_dim": 58,
    "n_layers": 12,
    "n_kv_groups": 3,
    "rope_base": 10_000.0,
    "dtype": torch.bfloat16,
}

# Enhanced training hyperparameters
FINE_TUNE_CONFIG = {
    "learning_rate": 1e-4,  # INCREASED - was too low
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "max_epochs": 15,  # Slightly more epochs
    "warmup_steps": 50,  # Reduced warmup
    "max_grad_norm": 1.0,
    "save_every": 500,
    "eval_every": 800,
    "weight_decay": 0.001,  # Reduced weight decay
    "dropout": 0.05,  # Reduced dropout
    "label_smoothing": 0.0,  # Removed label smoothing
    "early_stopping_patience": 4,
    "min_delta": 0.01,
}

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def analyze_dataset(json_path):
    """Analyze dataset quality and statistics"""
    print("="*50)
    print("DATASET ANALYSIS")
    print("="*50)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total examples: {len(data)}")

    # Check for duplicates
    inputs = [item['input'] for item in data]
    outputs = [item['output'] for item in data]
    unique_inputs = len(set(inputs))
    unique_outputs = len(set(outputs))

    print(f"Unique inputs: {unique_inputs} ({unique_inputs/len(inputs)*100:.1f}%)")
    print(f"Unique outputs: {unique_outputs} ({unique_outputs/len(outputs)*100:.1f}%)")

    # Check lengths
    input_lens = [len(item['input'].split(', ')) for item in data]  # Count words by comma
    output_lens = [len(item['output'].split()) for item in data]

    print(f"Input word count - Min: {min(input_lens)}, Max: {max(input_lens)}, Avg: {sum(input_lens)/len(input_lens):.1f}")
    print(f"Output word count - Min: {min(output_lens)}, Max: {max(output_lens)}, Avg: {sum(output_lens)/len(output_lens):.1f}")

    # Language distribution analysis
    eng_count = 0
    marathi_count = 0
    mixed_count = 0

    for item in data:
        text = item['input'] + ' ' + item['output']
        has_english = any(c.isascii() and c.isalpha() for c in text)
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)

        if has_english and has_devanagari:
            mixed_count += 1
        elif has_devanagari:
            marathi_count += 1
        elif has_english:
            eng_count += 1

    print(f"Language distribution:")
    print(f"  English: {eng_count} ({eng_count/len(data)*100:.1f}%)")
    print(f"  Marathi & Kokani: {marathi_count} ({marathi_count/len(data)*100:.1f}%)")
    print(f"  Mixed: {mixed_count} ({mixed_count/len(data)*100:.1f}%)")

    # Show some examples
    print(f"\nSample examples:")
    for i, item in enumerate(data[:3]):
        print(f"  {i+1}. Input: {item['input']}")
        print(f"     Output: {item['output']}")

    print("="*50)

    return {
        'total': len(data),
        'unique_inputs': unique_inputs,
        'avg_input_len': sum(input_lens)/len(input_lens),
        'avg_output_len': sum(output_lens)/len(output_lens),
        'english_count': eng_count,
        'marathi_count': marathi_count,
        'mixed_count': mixed_count
    }

class RoPEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config["n_heads"]
        self.head_dim = config["head_dim"]
        self.n_kv_groups = config["n_kv_groups"]

        self.q_proj = nn.Linear(config["emb_dim"], self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config["emb_dim"], self.n_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config["emb_dim"], self.n_kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config["emb_dim"], bias=False)

        self.rope_base = config["rope_base"]
        self.dropout = nn.Dropout(FINE_TUNE_CONFIG["dropout"])

    def apply_rotary_pos_emb(self, x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_groups, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_groups, self.head_dim)

        # Create RoPE embeddings
        pos = torch.arange(seq_len, device=x.device).float()
        dim_range = torch.arange(0, self.head_dim, 2, device=x.device).float()
        freqs = 1.0 / (self.rope_base ** (dim_range / self.head_dim))
        angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
        cos, sin = torch.cos(angles), torch.sin(angles)

        # Apply RoPE to queries and keys
        q = self.apply_rotary_pos_emb(q, cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2))
        k = self.apply_rotary_pos_emb(k, cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2))

        # Expand k, v for grouped query attention
        k = k.repeat(1, 1, self.n_heads // self.n_kv_groups, 1)
        v = v.repeat(1, 1, self.n_heads // self.n_kv_groups, 1)

        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed Forward Network"""
    def __init__(self, emb_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate_output = torch.sigmoid(self.gate(x)) * self.gate(x)
        up_output = self.fc1(x)
        hidden = gate_output * up_output
        hidden = self.dropout(hidden)
        output = self.fc3(hidden)
        return self.dropout(output)

class Gemma3Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = RoPEAttention(config)
        self.feed_forward = SwiGLUFFN(config["emb_dim"], config["hidden_dim"], FINE_TUNE_CONFIG["dropout"])
        self.attention_norm = nn.RMSNorm(config["emb_dim"])
        self.ffn_norm = nn.RMSNorm(config["emb_dim"])

    def forward(self, x, mask=None):
        # Pre-norm attention
        normed_x = self.attention_norm(x)
        attn_out = self.attention(normed_x, mask)
        x = x + attn_out

        # Pre-norm feed forward
        normed_x = self.ffn_norm(x)
        ffn_out = self.feed_forward(normed_x)
        x = x + ffn_out

        return x

class Gemma3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.layers = nn.ModuleList([Gemma3Block(config) for _ in range(config["n_layers"])])
        self.norm = nn.RMSNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed_tokens(input_ids)

        # Create causal mask
        seq_len = input_ids.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        combined_mask = causal_mask * attention_mask

        for layer in self.layers:
            x = layer(x, combined_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=FINE_TUNE_CONFIG.get("label_smoothing", 0.0)
            )
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": logits}

class SentenceFormationDataset(Dataset):
    def __init__(self, json_path: str, tokenizer_path: str, max_length: int = 512, split: str = "train"):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.max_length = max_length
        self.split = split

        # Load and process data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

       # Clean and filter data using the clean_dataset_entry function
        cleaned_data = []
        for item in data:
            if isinstance(item, dict) and 'input' in item and 'output' in item:
                cleaned_item = self.clean_dataset_entry(item)  # CALL IT HERE
                if cleaned_item is not None:  # Only add if cleaning succeeded
                    cleaned_data.append(cleaned_item)

        print(f"Original data: {len(data)}, Cleaned data: {len(cleaned_data)}")

        # Shuffle and split data
        random.shuffle(cleaned_data)
        train_size = int(0.85 * len(cleaned_data))
        val_size = int(0.10 * len(cleaned_data))

        if split == "train":
            self.data = cleaned_data[:train_size]
        elif split == "val":
            self.data = cleaned_data[train_size:train_size + val_size]
        else:  # test
            self.data = cleaned_data[train_size + val_size:]

        logger.info(f"Loaded {len(self.data)} examples for {split} split")

    @staticmethod
    def clean_dataset_entry(item):
        """Clean individual dataset entries"""
        input_text = str(item['input']).strip()
        output_text = str(item['output']).strip()

        # Basic cleaning
        input_text = input_text.replace('  ', ' ')
        output_text = output_text.replace('  ', ' ')

        # Count actual words
        if ',' in input_text:
            input_words = [w.strip() for w in input_text.split(',') if w.strip()]
            input_text = ', '.join(input_words)
            word_count = len(input_words)
        else:
            input_words = input_text.split()
            word_count = len(input_words)

        output_words = len(output_text.split())

        # Quality checks
        if word_count < 2 or word_count > 10:  # Reasonable word count
            return None
        if output_words < 3 or output_words > 30:  # Reasonable sentence length
            return None
        if not output_text or not input_text:
            return None

        return {
            'input': input_text,
            'output': output_text
        }

    def create_training_format(self, input_words: str, output_sentence: str) -> str:
        """Create training format: Input: <words> Output: <sentence>"""
        # Clean input - handle both comma-separated and space-separated
        if ',' in input_words:
            words = [w.strip() for w in input_words.split(',')]
            input_clean = ', '.join(words)
        else:
            input_clean = input_words.strip()

        # Format for instruction following
        prompt = f"Words: {input_clean}\nSentence: {output_sentence}"
        return prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_words = item["input"]
        output_sentence = item["output"]

        # Create training format
        text = self.create_training_format(input_words, output_sentence)

        # Tokenize with BOS and EOS tokens
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)

        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + [self.tokenizer.eos_id()]

        # Create labels - only compute loss on the sentence part
        labels = tokens.copy()

        # Find "Sentence: " position and only compute loss after it
        text_str = text
        sentence_pos = text_str.find("Sentence: ")

        if sentence_pos != -1:
            # Encode just the prefix to find where to start loss computation
            prefix = text_str[:sentence_pos + len("Sentence: ")]
            prefix_tokens = self.tokenizer.encode(prefix, add_bos=True, add_eos=False)

            # Mask everything before the actual sentence content
            mask_until = len(prefix_tokens)
            for i in range(min(mask_until, len(labels))):
                labels[i] = -100
        else:
            # Fallback - mask first 30% only
            mask_until = int(len(labels) * 0.3)
            for i in range(mask_until):
                labels[i] = -100

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
            "text": text
        }

def collate_fn(batch):
    """Custom collate function"""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    labels = []
    attention_masks = []

    for item in batch:
        input_id = item["input_ids"]
        label = item["labels"]
        attention_mask = item["attention_mask"]

        # Pad sequences
        pad_length = max_len - len(input_id)
        input_ids.append(F.pad(input_id, (0, pad_length), value=0))
        labels.append(F.pad(label, (0, pad_length), value=-100))
        attention_masks.append(F.pad(attention_mask, (0, pad_length), value=0))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_masks)
    }

def generate_sentence(model, tokenizer, input_words, device, max_new_tokens=50, temperature=0.7, top_p=0.9):
    """Generate sentence from input words"""
    model.eval()

    # Clean input
    if ',' in input_words:
        words = [w.strip() for w in input_words.split(',')]
        input_clean = ', '.join(words)
    else:
        input_clean = input_words.strip()

    # Create prompt in same format as training
    prompt = f"Words: {input_clean}\nSentence:"

    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([tokens], device=device)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"][0, -1, :] / temperature

            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Stop if EOS token
            if next_token.item() == tokenizer.eos_id():
                break

             # Stop if we hit a newline (end of sentence)
            decoded = tokenizer.decode([next_token.item()])
            if '\n' in decoded:
                break

            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Decode only the generated part
    if generated_tokens:
        generated_text = tokenizer.decode(generated_tokens).strip()
        # Clean up any residual special tokens
        generated_text = generated_text.replace("<s>", "").replace("</s>", "").strip()
        return generated_text
    else:
        return "Unable to generate sentence"

def evaluate_model(model, dataloader, device, tokenizer=None, silent=False):
    """Evaluate model with cleaner output"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        # Make evaluation quiet - no progress bar unless specifically requested
        iterator = dataloader if silent else tqdm(dataloader, desc="Evaluating", leave=False, ncols=60)

        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            if loss is not None:
                valid_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

    return avg_loss, perplexity

def test_generation_samples(model, tokenizer, device, epoch_num=None):
    """Test generation with sample inputs"""
    test_cases = [
        "cat, dog, playing, garden",
        "book, reading, knowledge, important",
        "happy, children, school, learning",
        "मुलं, खेळत, बगीचा",  # Marathi if present
        "technology, future, innovation"
    ]

    header = f"EPOCH {epoch_num} GENERATION TEST" if epoch_num is not None else "GENERATION TEST"
    print(f"\n{'='*50}")
    print(header)
    print("="*50)

    model.eval()
    for i, test_input in enumerate(test_cases[:3]):  # Test only first 3 to avoid spam
        try:
            generated = generate_sentence(model, tokenizer, test_input, device, temperature=0.7)
            print(f"{i+1}. {test_input} -> {generated}")
        except Exception as e:
            print(f"{i+1}. {test_input} -> Error: {e}")

    print("="*50)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def fine_tune_model():
    """Enhanced fine-tuning function with cleaner evaluation"""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Analyze dataset
    dataset_stats = analyze_dataset(JSON_DATA_PATH)

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)
    logger.info(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size()}")

    # Create datasets
    train_dataset = SentenceFormationDataset(JSON_DATA_PATH, TOKENIZER_PATH, split="train")
    val_dataset = SentenceFormationDataset(JSON_DATA_PATH, TOKENIZER_PATH, split="val")
    test_dataset = SentenceFormationDataset(JSON_DATA_PATH, TOKENIZER_PATH, split="test")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=FINE_TUNE_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=FINE_TUNE_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Calculate training steps
    total_batches = len(train_loader) * FINE_TUNE_CONFIG["max_epochs"]
    total_steps = total_batches // FINE_TUNE_CONFIG["gradient_accumulation_steps"]

    print(f"\nTraining Configuration:")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Batch size: {FINE_TUNE_CONFIG['batch_size']}")
    print(f"Gradient accumulation: {FINE_TUNE_CONFIG['gradient_accumulation_steps']}")
    print(f"Effective batch size: {FINE_TUNE_CONFIG['batch_size'] * FINE_TUNE_CONFIG['gradient_accumulation_steps']}")
    print(f"Total steps: {total_steps}")

    # Load model
    model = Gemma3Model(GEMMA3_CONFIG_110M)

    # Load pretrained weights if available
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        logger.info("Loading pretrained model...")
        try:
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Load with strict=False to handle architecture differences
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            logger.info(f"Loaded pretrained model. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")

        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")
            logger.info("Starting with random initialization")
    else:
        logger.info("No pretrained model found, starting with random initialization")

    model = model.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINE_TUNE_CONFIG["learning_rate"],
        weight_decay=FINE_TUNE_CONFIG["weight_decay"],
        betas=(0.9, 0.95)
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        FINE_TUNE_CONFIG["warmup_steps"],
        total_steps
    )

    # Training variables
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    # Training loop
    logger.info("Starting training...")

    for epoch in range(FINE_TUNE_CONFIG["max_epochs"]):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{FINE_TUNE_CONFIG['max_epochs']}",
                           leave=True, ncols=100)

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            if loss is not None:
                loss = loss / FINE_TUNE_CONFIG["gradient_accumulation_steps"]
                loss.backward()
                epoch_loss += loss.item()

                if (step + 1) % FINE_TUNE_CONFIG["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), FINE_TUNE_CONFIG["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{current_lr:.2e}",
                        'step': global_step
                    })

                    # Evaluation - less frequent to avoid spam
                    if global_step % FINE_TUNE_CONFIG["eval_every"] == 0:
                        print(f"\n[Step {global_step}] Running evaluation...")
                        val_loss, val_perp = evaluate_model(model, val_loader, device, silent=True)
                        print(f"[Step {global_step}] Val Loss: {val_loss:.4f}, Val Perplexity: {val_perp:.4f}")

                        # Save best model
                        if val_loss < best_val_loss - FINE_TUNE_CONFIG["min_delta"]:
                            best_val_loss = val_loss
                            patience_counter = 0

                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'epoch': epoch,
                                'step': global_step,
                                'val_loss': val_loss,
                                'val_perplexity': val_perp,
                                'config': GEMMA3_CONFIG_110M,
                                'training_config': FINE_TUNE_CONFIG
                            }, FINETUNED_MODEL_PATH.replace('.pt', '_best.pt'))

                            print(f"[Step {global_step}] ✓ New best model saved! Val Loss: {val_loss:.4f}")
                        else:
                            patience_counter += 1

                        if patience_counter >= FINE_TUNE_CONFIG["early_stopping_patience"]:
                            print(f"[Step {global_step}] Early stopping triggered!")
                            break

                        model.train()  # Resume training mode

                    # Save checkpoint
                    if global_step % FINE_TUNE_CONFIG["save_every"] == 0:
                        checkpoint_path = FINETUNED_MODEL_PATH.replace('.pt', f'_step_{global_step}.pt')
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'epoch': epoch,
                            'step': global_step,
                            'config': GEMMA3_CONFIG_110M,
                            'training_config': FINE_TUNE_CONFIG
                        }, checkpoint_path)
                        print(f"\n[Step {global_step}] Checkpoint saved")

        # Check early stopping at epoch level
        if patience_counter >= FINE_TUNE_CONFIG["early_stopping_patience"]:
            break

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n[Epoch {epoch+1}] Completed. Average loss: {avg_epoch_loss:.4f}")

        # End-of-epoch evaluation and testing
        print(f"[Epoch {epoch+1}] Running end-of-epoch evaluation...")
        val_loss, val_perp = evaluate_model(model, val_loader, device, silent=True)
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}, Perplexity: {val_perp:.4f}")

        # Test generation at end of each epoch
        test_generation_samples(model, tokenizer, device, epoch_num=epoch+1)

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    # Load best model
    if os.path.exists(FINETUNED_MODEL_PATH.replace('.pt', '_best.pt')):
        best_checkpoint = torch.load(FINETUNED_MODEL_PATH.replace('.pt', '_best.pt'), map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print("Loaded best model for final evaluation")

    # Final validation
    final_val_loss, final_val_perp = evaluate_model(model, val_loader, device, silent=True)
    print(f"Final Validation - Loss: {final_val_loss:.4f}, Perplexity: {final_val_perp:.4f}")

    # Comprehensive test generation
    comprehensive_test_cases = [
        # English examples
        "cat, running, garden, quickly",
        "book, table, red, interesting",
        "children, playing, park, sunny",
        "technology, future, changing, rapidly",

        # Marathi examples (if present in dataset)
        "मुलं, खेळत, बगीचा, आनंद",
        "पुस्तक, वाचणे, आवडते, मला",

        # Mixed examples
        "love, music, heart, deeply",
        "science, discovery, amazing, always",

        # From actual dataset samples
        "her, licked, all, ate",
        "happy, children, school, learning"
    ]

    print("\n" + "="*60)
    print("COMPREHENSIVE FINAL TEST")
    print("="*60)

    model.eval()
    for i, test_input in enumerate(comprehensive_test_cases):
        print(f"\n{i+1}. Input: {test_input}")
        try:
            # Generate with different temperatures
            generated1 = generate_sentence(model, tokenizer, test_input, device, temperature=0.6)
            generated2 = generate_sentence(model, tokenizer, test_input, device, temperature=0.8)

            print(f"   Output 1 (T=0.6): {generated1}")
            print(f"   Output 2 (T=0.8): {generated2}")
        except Exception as e:
            print(f"   Error: {e}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': GEMMA3_CONFIG_110M,
        'training_config': FINE_TUNE_CONFIG,
        'final_val_loss': final_val_loss,
        'final_val_perplexity': final_val_perp,
        'dataset_stats': dataset_stats,
        'tokenizer_path': TOKENIZER_PATH
    }, FINETUNED_MODEL_PATH)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model: {FINETUNED_MODEL_PATH.replace('.pt', '_best.pt')}")
    print(f"Final model: {FINETUNED_MODEL_PATH}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation perplexity: {final_val_perp:.4f}")
    print("="*60)

    return model, tokenizer

def load_and_test_model(model_path=None, test_cases=None):
    """Load trained model and test"""
    if model_path is None:
        model_path = FINETUNED_MODEL_PATH.replace('.pt', '_best.pt')
        if not os.path.exists(model_path):
            model_path = FINETUNED_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', GEMMA3_CONFIG_110M)

    model = Gemma3Model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {model_path}")
    print(f"Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"Validation Perplexity: {checkpoint.get('val_perplexity', 'N/A')}")

    # Test cases
    if test_cases is None:
        test_cases = [
            "cat, dog, playing, garden",
            "book, reading, knowledge, important",
            "मुलं, शाळा, जाणे, आवडते",  # Marathi
            "happy, children, school, learning",
            "technology, future, innovation, exciting"
        ]

    print("\n" + "="*50)
    print("TESTING FINE-TUNED MODEL")
    print("="*50)

    for i, test_input in enumerate(test_cases):
        print(f"\n{i+1}. Input: {test_input}")

        try:
            # Generate multiple samples with different parameters
            for j, (temp, top_p) in enumerate([(0.6, 0.9), (0.8, 0.85)]):
                generated = generate_sentence(model, tokenizer, test_input, device,
                                           temperature=temp, top_p=top_p)
                print(f"   Sample {j+1} (T={temp}, p={top_p}): {generated}")
        except Exception as e:
            print(f"   Generation error: {e}")

    return model, tokenizer

def interactive_test():
    """Interactive testing function"""
    model, tokenizer = load_and_test_model()

    print("\n" + "="*50)
    print("INTERACTIVE SENTENCE GENERATION")
    print("Enter words separated by commas or 'quit' to exit")
    print("="*50)

    while True:
        try:
            user_input = input("\nEnter words: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            device = next(model.parameters()).device
            generated = generate_sentence(model, tokenizer, user_input, device)
            print(f"Generated sentence: {generated}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")

def export_for_inference(model_path=None, output_dir=f'{DRIVE_PATH}/Finetuning1/Output/'):
    """Export model for inference"""
    if model_path is None:
        model_path = FINETUNED_MODEL_PATH.replace('.pt', '_best.pt')

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')

    # Save just the model state dict and config
    inference_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
        'config': checkpoint.get('config', GEMMA3_CONFIG_110M),
        'val_perplexity': checkpoint.get('val_perplexity', None)
    }

    torch.save(inference_checkpoint, f"{output_dir}/model.pt")

    # Copy tokenizer
    import shutil
    shutil.copy2(TOKENIZER_PATH, f"{output_dir}/tokenizer.model")

    # Save generation script
    generation_script = '''
import torch
import sentencepiece as spm
from model import Gemma3Model, generate_sentence

def load_model(model_dir):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f"{model_dir}/tokenizer.model")

    checkpoint = torch.load(f"{model_dir}/model.pt", map_location='cpu')
    model = Gemma3Model(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer

def generate(words, model_dir=".", temperature=0.7):
    model, tokenizer = load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return generate_sentence(model, tokenizer, words, device, temperature=temperature)
'''

    with open(f"{output_dir}/generate.py", 'w') as f:
        f.write(generation_script)

    print(f"Model exported to {output_dir}/")
    print("Use generate.py for inference")

# Main execution
if __name__ == "__main__":
    print("Starting Sentence Formation Fine-tuning")
    print("="*50)

    try:
        # Run fine-tuning
        model, tokenizer = fine_tune_model()

        print("\nFine-tuning completed successfully!")
        print("You can now test the model with:")
        print("1. load_and_test_model() - Test on predefined examples")
        print("2. interactive_test() - Interactive testing")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

        # Try to load existing model for testing
        if os.path.exists(FINETUNED_MODEL_PATH) or os.path.exists(FINETUNED_MODEL_PATH.replace('.pt', '_best.pt')):
            print("\nTrying to load existing model for testing...")
            try:
                load_and_test_model()
            except Exception as e2:
                logger.error(f"Could not load existing model: {e2}")

print("All functions loaded successfully!")
print("Run fine_tune_model() to start training")
print("Run load_and_test_model() to test existing model")
print("Run interactive_test() for interactive testing")
