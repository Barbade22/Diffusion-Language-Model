import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
import os
from collections import Counter, defaultdict
import pickle
import numpy as np
import re
from tqdm import tqdm
import pickle
import re
from collections import defaultdict
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import gc

class MarathiBPETokenizer:
    def __init__(self, vocab_size=49493):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.vocab = {}
        self.merges = []
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
            '<MASK>': 4
        }
        self.token_to_id = {}
        self.id_to_token = {}

        # HF tokenizer (internal)
        self._hf_tokenizer = None
        self._trained = False

    def _get_word_freqs(self, corpus):
        """Extract word frequencies from corpus (kept for compatibility)"""
        word_freqs = defaultdict(int)
        for text in corpus:
            words = re.findall(r'\S+', text.lower())
            for word in words:
                word_freqs[word] += 1
        return dict(word_freqs)

    def _get_splits(self, word_freqs):
        """Split words into characters initially (kept for compatibility)"""
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word)
        return splits

    def _compute_pair_freqs(self, splits, word_freqs):
        """Compute frequency of adjacent pairs (kept for compatibility)"""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return dict(pair_freqs)

    def _merge_vocab(self, pair, splits):
        """Merge the most frequent pair in vocabulary (kept for compatibility)"""
        new_splits = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word in splits:
            word_str = ' '.join(splits[word])
            new_word = p.sub(''.join(pair), word_str)
            new_splits[word] = new_word.split()
        return new_splits

    def build_vocab(self, corpus):
        """Train BPE tokenizer on corpus - alias for train()"""
        self.train(corpus)

    def train(self, corpus):
        """Train BPE tokenizer on corpus using HF tokenizers (FAST!)"""
        print(f"Training BPE tokenizer on {len(corpus)} texts...")

        # Store word frequencies for compatibility
        self.word_freqs = self._get_word_freqs(corpus)
        print(f"Found {len(self.word_freqs)} unique words")

        # Initialize HF tokenizer
        self._hf_tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

        # Set up pre-tokenization (handle Marathi text properly)
        self._hf_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Prepare special tokens list
        special_tokens_list = list(self.special_tokens.keys())

        # Set up the trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=1,  # Keep all tokens for Marathi
            special_tokens=special_tokens_list,
            show_progress=True
        )

        print(f"Performing BPE training with target vocab size: {self.vocab_size}")

        # Train the tokenizer
        self._hf_tokenizer.train_from_iterator(corpus, trainer)

        # Set up post-processing
        self._hf_tokenizer.post_processor = processors.TemplateProcessing(
            single="<START> $A <END>",
            pair="<START> $A <END> $B <END>",
            special_tokens=[
                ("<START>", self._hf_tokenizer.token_to_id("<START>")),
                ("<END>", self._hf_tokenizer.token_to_id("<END>")),
            ],
        )

        # Set decoder
        self._hf_tokenizer.decoder = decoders.BPEDecoder()

        # Build compatibility mappings
        self._build_compatibility_mappings()

        self._trained = True
        print(f"Training complete! Final vocabulary size: {len(self.vocab)}")

    def _build_compatibility_mappings(self):
        """Build vocab mappings for compatibility with original interface"""
        # Get vocabulary from HF tokenizer
        hf_vocab = self._hf_tokenizer.get_vocab()

        # Create mappings
        self.vocab = hf_vocab.copy()
        self.token_to_id = hf_vocab.copy()
        self.id_to_token = {v: k for k, v in hf_vocab.items()}

        # Extract merges for compatibility (approximation)
        # Note: HF tokenizer doesn't expose merges directly in the same format
        # This is a simplified version for compatibility
        self.merges = []

        # Update special token IDs to match HF tokenizer
        for token, _ in self.special_tokens.items():
            if token in hf_vocab:
                self.special_tokens[token] = hf_vocab[token]

    def _apply_bpe(self, word):
        """Apply BPE merges to a word using HF tokenizer"""
        if not self._trained:
            raise ValueError("Tokenizer not trained!")

        # Use HF tokenizer for actual BPE application
        encoding = self._hf_tokenizer.encode(word, add_special_tokens=False)
        return encoding.tokens

    def encode(self, text):
        """Encode text to token IDs using HF tokenizer"""
        if not self._trained:
            raise ValueError("Tokenizer not trained!")

        # Use HF tokenizer for encoding (much faster)
        encoding = self._hf_tokenizer.encode(text.lower(), add_special_tokens=False)
        return encoding.ids

    def decode(self, ids):
        """Decode token IDs back to text using HF tokenizer"""
        if not self._trained:
            raise ValueError("Tokenizer not trained!")

        if hasattr(ids, 'tolist'):
            ids = ids.tolist()

        # Filter out special tokens we don't want in output
        filtered_ids = []
        for token_id in ids:
            if token_id != self.special_tokens.get('<PAD>', 0):
                filtered_ids.append(token_id)

        # Use HF tokenizer for decoding
        result = self._hf_tokenizer.decode(filtered_ids, skip_special_tokens=True)

        # Post-process for Marathi punctuation
        result = re.sub(r'([।!?;:])', r' \1 ', result)
        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def save(self, path):
        """Save tokenizer with both HF tokenizer and compatibility data"""
        # Save HF tokenizer
        if self._trained:
            hf_path = path + "_hf.json"
            self._hf_tokenizer.save(hf_path)

        # Save compatibility data
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'word_freqs': self.word_freqs,
                'vocab': self.vocab,
                'merges': self.merges,
                'special_tokens': self.special_tokens,
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'trained': self._trained,
                'hf_tokenizer_path': hf_path if self._trained else None
            }, f)

    def load(self, path):
        """Load tokenizer with both HF tokenizer and compatibility data"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vocab_size = data['vocab_size']
            self.word_freqs = data['word_freqs']
            self.vocab = data['vocab']
            self.merges = data['merges']
            self.special_tokens = data['special_tokens']
            self.token_to_id = data['token_to_id']
            self.id_to_token = data['id_to_token']
            self._trained = data.get('trained', False)

            # Load HF tokenizer if available
            hf_path = data.get('hf_tokenizer_path')
            if hf_path and self._trained:
                try:
                    self._hf_tokenizer = Tokenizer.from_file(hf_path)
                except Exception as e:
                    print(f"Warning: Could not load HF tokenizer: {e}")
                    self._trained = False

# Positional Encoding - Memory efficient version
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        # Compute positional encoding on-the-fly to save memory
        seq_len = x.size(1)
        device = x.device

        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(0, seq_len, device=device).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() *
                           -(math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe.unsqueeze(0)

# Memory efficient Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with gradient checkpointing in mind
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x

# Time Embedding for Diffusion
class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, timesteps):
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# Main Diffusion Language Model - Memory Optimized
class DiffusionLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, d_ff=512, max_len=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Time embedding for diffusion
        self.time_embedding = TimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, timesteps):
        batch_size, seq_len = x.shape

        # Token embeddings
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        # Add time information
        time_emb = self.time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb.unsqueeze(1)

        x = self.dropout(x)

        # Transformer layers with gradient checkpointing
        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.norm(x)
        logits = self.output_proj(x)

        return logits

# Memory efficient Dataset class
class MarathiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Pre-tokenize and cache only indices to save memory
        self.tokenized_indices = []
        print("Pre-tokenizing dataset...")
        for i, text in enumerate(tqdm(texts)):
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 4:  # Filter out very short sequences
                self.tokenized_indices.append(i)

    def __len__(self):
        return len(self.tokenized_indices)

    def __getitem__(self, idx):
        text_idx = self.tokenized_indices[idx]
        text = self.texts[text_idx]
        tokens = self.tokenizer.encode(text)

        # Truncate or pad
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens.extend([self.tokenizer.special_tokens['<PAD>']] * (self.max_len - len(tokens)))

        return torch.tensor(tokens, dtype=torch.long)

# Memory management utilities
def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

# Diffusion noise scheduler
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def add_noise(x_start, t, tokenizer, noise_schedule):
    """Add noise by randomly masking tokens"""
    batch_size, seq_len = x_start.shape

    # Calculate noise level based on timestep
    noise_level = noise_schedule[t.cpu()].unsqueeze(1).to(t.device)

    # Create random mask based on noise level
    mask_prob = noise_level.expand_as(x_start.float())
    mask = torch.rand_like(x_start.float()) < mask_prob

    # Apply mask
    x_noisy = x_start.clone()
    x_noisy[mask] = tokenizer.special_tokens['<MASK>']

    return x_noisy

# Optimized training function with centralized configuration
def train_model():
    # ================================
    # CENTRALIZED CONFIGURATION
    # ================================
    CONFIG = {
        # Model Architecture
        'vocab_size': 12000,        # Smaller vocab for efficiency
        'd_model': 320,             # Carefully chosen for target param count
        'n_heads': 8,               # 320/8 = 40 (good head dimension)
        'n_layers': 12,             # Reasonable depth
        'd_ff': 1280,               # 4x d_model
        'dropout': 0.1,
        
        # Training Parameters
        'max_len': 256,             # Keep sequences short for speed
        'batch_size': 256,           # Large batch for efficiency
        'learning_rate': 3e-5,      # Moderate LR for stable long training
        'weight_decay': 0.01,
        'betas': (0.9, 0.95),
        'eps': 1e-8,
        
        # Training Schedule
        'target_token_param_ratio': 20,  # Tokens per parameter target
        'warmup_pct': 0.02,             # 2% warmup
        
        # Diffusion Parameters
        'timesteps': 50,            # Balanced - not too slow, not too simple
        
        # Data Processing
        'min_sentence_length': 20,  # Less filtering since we need all data
        'tokenizer_file': 'tokenizer_optimal.pkl',
        
        # Optimization
        'num_workers': 16,
        'pin_memory': True,
        'persistent_workers': True,
        'gradient_clip': 1.0,
        'memory_clear_freq': 100,
        
        # Checkpointing
        'checkpoint_freq': 5,      # Save checkpoint every N epochs
        'best_model_file': 'marathi_diffusion_best.pt',
        'final_model_file': 'marathi_diffusion_optimal_final.pt',
        'checkpoint_prefix': 'optimal_checkpoint_epoch_',
        # 'model_path': "optimal_checkpoint_epoch_110.pt"
        'model_path': None,  # Set to None if no pre-trained model
    }
    
    # ================================
    # SETUP AND INITIALIZATION
    # ================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Memory utilities
    def get_memory_usage():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def clear_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================================
    # DATA LOADING AND PREPROCESSING
    # ================================
    print("Loading data...")
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Less filtering since we need all the data we can get
    sentences = [s.strip() for s in text.split('\n') 
                if s.strip() and len(s.strip()) >= CONFIG['min_sentence_length']]
    
    print(f"Found {len(sentences)} text chunks")

    # Build tokenizer with configured vocab size
    tokenizer = MarathiBPETokenizer(vocab_size=CONFIG['vocab_size'])
    if os.path.exists(CONFIG['tokenizer_file']):
        tokenizer.load(CONFIG['tokenizer_file'])
    else:
        tokenizer.build_vocab(sentences)
        tokenizer.save(CONFIG['tokenizer_file'])

    # ================================
    # DATASET ANALYSIS AND OPTIMIZATION
    # ================================
    print("Calculating total tokens...")
    total_tokens = 0
    token_lengths = []
    
    for sentence in tqdm(sentences, desc="Counting tokens"):
        tokens = tokenizer.encode(sentence)
        token_count = len(tokens)
        total_tokens += token_count
        token_lengths.append(token_count)
    
    print(f"Total tokens in dataset: {total_tokens:,}")

    # Calculate optimal model size for target token/parameter ratio
    optimal_params = total_tokens // CONFIG['target_token_param_ratio']
    print(f"Optimal model size for {CONFIG['target_token_param_ratio']}:1 ratio: {optimal_params:,} parameters")

    # ================================
    # MODEL CREATION
    # ================================
    model = DiffusionLM(
        vocab_size=CONFIG['vocab_size'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        max_len=CONFIG['max_len'],
        dropout=CONFIG['dropout']
    ).to(device)
    model_path = CONFIG['model_path']
    if model_path:
        model_path = CONFIG['model_path']
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model Loaded Continuing Training")

    actual_params = sum(p.numel() for p in model.parameters())
    actual_ratio = total_tokens / actual_params
    
    # Calculate epochs needed for optimal training
    min_epochs_needed = max(1, int(CONFIG['target_token_param_ratio'] / actual_ratio))
    target_epochs = min_epochs_needed * 2  # Extra buffer for good training
    
    print(f"\n📊 MODEL SIZING ANALYSIS:")
    print(f"  • Target parameters: {optimal_params:,}")
    print(f"  • Actual parameters: {actual_params:,}")
    print(f"  • Single-epoch ratio: {actual_ratio:.1f} tokens/param")
    print(f"  • Minimum epochs needed: {min_epochs_needed}")
    print(f"  • Target epochs: {target_epochs}")
    print(f"  • Effective tokens: {total_tokens * target_epochs:,}")
    print(f"  • Effective ratio: {(total_tokens * target_epochs) / actual_params:.1f} tokens/param")

    # Dataset statistics
    import numpy as np
    token_lengths = np.array(token_lengths)
    truncated_count = np.sum(token_lengths > CONFIG['max_len'])
    truncated_percentage = (truncated_count / len(token_lengths)) * 100
    
    print(f"\n📈 TOKEN LENGTH STATISTICS:")
    print(f"  • Max sequence length: {CONFIG['max_len']}")
    print(f"  • Average tokens per sentence: {total_tokens/len(sentences):.1f}")
    print(f"  • Sentences to be truncated: {truncated_count:,} ({truncated_percentage:.1f}%)")
    
    if truncated_percentage > 5:
        print(f"  • ⚠️  WARNING: {truncated_percentage:.1f}% of data will be truncated!")

    # ================================
    # DATALOADER SETUP
    # ================================
    dataset = MarathiTextDataset(sentences, tokenizer, max_len=CONFIG['max_len'])
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        drop_last=True,
        persistent_workers=CONFIG['persistent_workers']
    )

    # ================================
    # OPTIMIZER AND SCHEDULER SETUP
    # ================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        eps=CONFIG['eps'],
        betas=CONFIG['betas']
    )

    # Learning rate schedule for long training
    total_steps = len(dataloader) * target_epochs
    warmup_steps = int(total_steps * CONFIG['warmup_pct'])
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=CONFIG['warmup_pct'],
        anneal_strategy='cos'
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens['<PAD>'])

    # ================================
    # DIFFUSION SETUP
    # ================================
    noise_schedule = cosine_beta_schedule(CONFIG['timesteps']).to(device)

    # Mixed precision for efficiency
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print(f"\n🎯 OPTIMAL RATIO TRAINING CONFIG:")
    print(f"  • Model parameters: {actual_params:,}")
    print(f"  • Batch size: {CONFIG['batch_size']}")
    print(f"  • Sequence length: {CONFIG['max_len']}")
    print(f"  • Learning rate: {CONFIG['learning_rate']:.1e}")
    print(f"  • Diffusion timesteps: {CONFIG['timesteps']}")
    print(f"  • Training epochs: {target_epochs}")
    print(f"  • Steps per epoch: {len(dataloader)}")
    print(f"  • Total training steps: {total_steps:,}")
    print(f"  • Warmup steps: {warmup_steps:,}")

    # ================================
    # TRAINING LOOP
    # ================================
    model.train()
    best_loss = float('inf')
    los = checkpoint['loss']
    print("Previous Loss:", los)
    
    print(f"\n🚀 Starting optimal ratio training...")
    print(f"Goal: Achieve proper token/parameter ratio through {target_epochs} epochs")
    start_epoch = checkpoint['epoch']
    epoch_pbar = tqdm(range(start_epoch,target_epochs), desc='Optimal Training', position=0, leave=True, ncols=120)

    for epoch in epoch_pbar:
        total_loss = 0
        step_pbar = tqdm(
            dataloader, 
            desc=f'Epoch {epoch+1}/{target_epochs}', 
            position=1, 
            leave=False,
            ncols=120
        )

        for batch_idx, batch in enumerate(step_pbar):
            batch = batch.to(device, non_blocking=True)

            # Sample random timesteps
            t = torch.randint(0, CONFIG['timesteps'], (batch.shape[0],), device=device)
            noisy_batch = add_noise(batch, t, tokenizer, noise_schedule)

            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(noisy_batch, t)
                    loss = criterion(logits.view(-1, CONFIG['vocab_size']), batch.view(-1))
            else:
                logits = model(noisy_batch, t)
                loss = criterion(logits.view(-1, CONFIG['vocab_size']), batch.view(-1))

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            
            step_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.1e}',
                'mem': f'{get_memory_usage():.1f}GB',
                'progress': f'{((epoch * len(dataloader) + batch_idx + 1) / total_steps) * 100:.1f}%'
            })

            # Memory management
            if batch_idx % CONFIG['memory_clear_freq'] == 0:
                clear_memory()

        step_pbar.close()
        avg_loss = total_loss / len(dataloader)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'avg_loss': f'{avg_loss:.4f}',
            'best_loss': f'{best_loss:.4f}',
            'lr': f'{current_lr:.1e}',
            'mem': f'{get_memory_usage():.1f}GB'
        })

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'vocab_size': CONFIG['vocab_size'],
                    'd_model': CONFIG['d_model'],
                    'n_heads': CONFIG['n_heads'],
                    'n_layers': CONFIG['n_layers'],
                    'd_ff': CONFIG['d_ff'],
                    'max_len': CONFIG['max_len']
                },
                'training_config': CONFIG,
                'epoch': epoch,
                'loss': avg_loss,
                'timesteps': CONFIG['timesteps'],
                'noise_schedule': noise_schedule,
                'total_params': actual_params,
                'token_param_ratio': (total_tokens * (epoch + 1)) / actual_params
            }, CONFIG['best_model_file'])

        # Save checkpoint periodically
        if (epoch + 1) % CONFIG['checkpoint_freq'] == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'model_config': {
                    'vocab_size': CONFIG['vocab_size'],
                    'd_model': CONFIG['d_model'],
                    'n_heads': CONFIG['n_heads'],
                    'n_layers': CONFIG['n_layers'],
                    'd_ff': CONFIG['d_ff'],
                    'max_len': CONFIG['max_len']
                },
                'training_config': CONFIG,
                'epoch': epoch,
                'loss': avg_loss,
                'timesteps': CONFIG['timesteps'],
                'noise_schedule': noise_schedule
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()

            torch.save(checkpoint, f"{CONFIG['checkpoint_prefix']}{epoch+1:03d}.pt")
            
            current_ratio = (total_tokens * (epoch + 1)) / actual_params
            print(f"\n📈 Checkpoint {epoch+1}: Current token/param ratio: {current_ratio:.1f}")

    epoch_pbar.close()

    # ================================
    # FINAL MODEL SAVE
    # ================================
    final_ratio = (total_tokens * target_epochs) / actual_params
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': CONFIG['vocab_size'],
            'd_model': CONFIG['d_model'],
            'n_heads': CONFIG['n_heads'],
            'n_layers': CONFIG['n_layers'],
            'd_ff': CONFIG['d_ff'],
            'max_len': CONFIG['max_len']
        },
        'training_config': CONFIG,
        'timesteps': CONFIG['timesteps'],
        'noise_schedule': noise_schedule,
        'training_stats': {
            'total_epochs': target_epochs,
            'total_params': actual_params,
            'total_tokens': total_tokens,
            'effective_tokens': total_tokens * target_epochs,
            'final_token_param_ratio': final_ratio,
            'final_loss': avg_loss
        }
    }, CONFIG['final_model_file'])

    print(f"\n✅ OPTIMAL RATIO TRAINING COMPLETED!")
    print(f"📊 Final Statistics:")
    print(f"  • Model parameters: {actual_params:,}")
    print(f"  • Training epochs: {target_epochs}")
    print(f"  • Effective tokens seen: {total_tokens * target_epochs:,}")
    print(f"  • Final token/param ratio: {final_ratio:.1f}")
    print(f"  • Best loss achieved: {best_loss:.4f}")
    print(f"📁 Models saved:")
    print(f"  • Best model: {CONFIG['best_model_file']}")
    print(f"  • Final model: {CONFIG['final_model_file']}")
    
    if final_ratio >= CONFIG['target_token_param_ratio']:
        print(f"🎯 SUCCESS! Achieved optimal token/parameter ratio!")
    else:
        print(f"⚠️  Note: Ratio is {final_ratio:.1f}, ideally should be {CONFIG['target_token_param_ratio']}+")
        print(f"   Consider training for more epochs or getting more data")

    return model
if __name__ == "__main__":
    train_model()