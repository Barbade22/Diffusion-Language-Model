import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
import numpy as np
import time
import os
import re
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from tokenizers import Tokenizer
from tqdm import tqdm

# Set page config
st.set_page_config(
    page_title="🔮 Enhanced Marathi Diffusion Generator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .step-container {
        border: 2px solid #4ECDC4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .token-masked {
        background-color: #FF6B6B;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        font-weight: bold;
    }
    
    .token-early {
        background-color: #FFA500;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        font-weight: bold;
    }
    
    .token-mid {
        background-color: #4ECDC4;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        font-weight: bold;
    }
    
    .token-late {
        background-color: #45B7D1;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        font-weight: bold;
    }
    
    .token-final {
        background-color: #96CEB4;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        font-weight: bold;
    }
    
    .token-input {
        background-color: #9B59B6;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        font-weight: bold;
        border: 2px solid #8E44AD;
    }
    
    .input-area {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Your model classes (from training code)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device

        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(0, seq_len, device=device).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() *
                           -(math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe.unsqueeze(0)

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
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x

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

class DiffusionLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, d_ff=512, max_len=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.time_embedding = TimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, timesteps):
        batch_size, seq_len = x.shape

        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        time_emb = self.time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb.unsqueeze(1)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.output_proj(x)

        return logits

# Your tokenizer class with UI-friendly methods
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
        self._hf_tokenizer = None
        self._trained = False

    def _fallback_encode(self, text):
        """Fallback encoding using vocab mapping when HF tokenizer fails"""
        words = re.findall(r'\S+', text.lower())
        tokens = []
        for word in words:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                char_tokens = []
                for char in word:
                    if char in self.token_to_id:
                        char_tokens.append(self.token_to_id[char])
                    else:
                        char_tokens.append(self.special_tokens['<UNK>'])
                tokens.extend(char_tokens)
        return tokens

    def _fallback_decode(self, ids):
        """Fallback decoding using id_to_token mapping"""
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        
        tokens = []
        for token_id in ids:
            if token_id in self.id_to_token and token_id != self.special_tokens.get('<PAD>', 0):
                token = self.id_to_token[token_id]
                if token not in ['<START>', '<END>', '<MASK>', '<UNK>', '<PAD>']:
                    tokens.append(token)
        
        result = ' '.join(tokens)
        result = re.sub(r'([।!?;:])', r' \1 ', result)
        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def encode(self, text):
        if not self._trained:
            raise ValueError("Tokenizer not trained!")
        
        if self._hf_tokenizer is not None:
            try:
                encoding = self._hf_tokenizer.encode(text.lower(), add_special_tokens=False)
                return encoding.ids
            except Exception:
                pass
        
        return self._fallback_encode(text)

    def decode(self, ids):
        if not self._trained:
            raise ValueError("Tokenizer not trained!")

        if self._hf_tokenizer is not None:
            try:
                if hasattr(ids, 'tolist'):
                    ids = ids.tolist()

                filtered_ids = []
                for token_id in ids:
                    if token_id != self.special_tokens.get('<PAD>', 0):
                        filtered_ids.append(token_id)

                result = self._hf_tokenizer.decode(filtered_ids, skip_special_tokens=True)
                result = re.sub(r'([।!?;:])', r' \1 ', result)
                result = re.sub(r'\s+', ' ', result).strip()
                return result
            except Exception:
                pass
        
        return self._fallback_decode(ids)

    def get_mask_token_id(self):
        return self.special_tokens.get('<MASK>', 4)
    
    def get_pad_token_id(self):
        return self.special_tokens.get('<PAD>', 0)

    def load(self, path):
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

            hf_path = data.get('hf_tokenizer_path')
            if hf_path and self._trained:
                try:
                    if os.path.exists(hf_path):
                        self._hf_tokenizer = Tokenizer.from_file(hf_path)
                except Exception:
                    self._hf_tokenizer = None

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def prepare_input_sequence(input_text, tokenizer, max_len, generation_mode):
    """Prepare input sequence based on generation mode"""
    mask_token_id = tokenizer.get_mask_token_id()
    
    if generation_mode == "Unconditional (Random Generation)":
        x = torch.full((1, max_len), mask_token_id, dtype=torch.long)
        fixed_positions = set()
        
    elif generation_mode == "Text Completion":
        input_ids = tokenizer.encode(input_text)
        input_len = min(len(input_ids), max_len - 10)
        
        x = torch.full((1, max_len), mask_token_id, dtype=torch.long)
        
        for i, token_id in enumerate(input_ids[:input_len]):
            x[0, i] = token_id
        
        fixed_positions = set(range(input_len))
        
    elif generation_mode == "Fill in the Blanks":
        input_text_processed = input_text.replace('[MASK]', '<MASK>')
        
        input_ids = []
        i = 0
        while i < len(input_text_processed):
            if input_text_processed[i:].startswith('<MASK>'):
                input_ids.append(mask_token_id)
                i += 6
            else:
                char_id = tokenizer.token_to_id.get(input_text_processed[i], tokenizer.special_tokens.get('<UNK>', 1))
                input_ids.append(char_id)
                i += 1
        
        input_len = min(len(input_ids), max_len)
        x = torch.full((1, max_len), mask_token_id, dtype=torch.long)
        
        for i, token_id in enumerate(input_ids[:input_len]):
            x[0, i] = token_id
        
        fixed_positions = set()
        for i in range(input_len):
            if x[0, i] != mask_token_id:
                fixed_positions.add(i)
    
    else:  # Conditioned Generation
        input_ids = tokenizer.encode(input_text)
        input_len = min(len(input_ids), max_len // 2)
        
        x = torch.full((1, max_len), mask_token_id, dtype=torch.long)
        
        for i, token_id in enumerate(input_ids[:input_len]):
            x[0, i] = token_id
        
        fixed_positions = set(range(input_len))
    
    return x, fixed_positions

def get_token_color_class(token, step, total_steps, is_fixed=False):
    """Get CSS class for token based on denoising progress"""
    if is_fixed:
        return 'token-input'
    elif token == '<MASK>':
        return 'token-masked'
    else:
        progress = 1 - (step / total_steps)
        if progress > 0.8:
            return 'token-early'
        elif progress > 0.6:
            return 'token-mid'
        elif progress > 0.4:
            return 'token-late'
        else:
            return 'token-final'

def get_confidence_color_class(confidence, is_fixed=False):
    """Get CSS class based on confidence level"""
    if is_fixed:
        return 'token-input'
    elif confidence > 0.8:
        return 'token-final'
    elif confidence > 0.6:
        return 'token-late'
    elif confidence > 0.4:
        return 'token-mid'
    elif confidence > 0.2:
        return 'token-early'
    else:
        return 'token-masked'

def create_enhanced_progress_chart(steps_data):
    """Create enhanced progress chart with phases"""
    if not steps_data:
        return None
        
    df = pd.DataFrame(steps_data)
    
    fig = go.Figure()
    
    # Separate data by phase
    phase1_data = df[df['phase'] == 'Initial Diffusion']
    phase2_data = df[df['phase'] == 'Confidence Refinement']
    
    # Phase 1 data
    if not phase1_data.empty:
        fig.add_trace(go.Scatter(
            x=phase1_data['step'],
            y=phase1_data['masked_percentage'],
            mode='lines+markers',
            name='Phase 1: Masked %',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=phase1_data['step'],
            y=phase1_data['avg_confidence'],
            mode='lines+markers',
            name='Phase 1: Confidence',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
    
    # Phase 2 data
    if not phase2_data.empty:
        fig.add_trace(go.Scatter(
            x=phase2_data['step'],
            y=phase2_data['masked_percentage'],
            mode='lines+markers',
            name='Phase 2: Re-masked %',
            line=dict(color='#9B59B6', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=phase2_data['step'],
            y=phase2_data['avg_confidence'],
            mode='lines+markers',
            name='Phase 2: Confidence',
            line=dict(color='#E67E22', width=3, dash='dash'),
            marker=dict(size=8),
            yaxis='y2'
        ))
    
    fig.update_layout(
        title="🔮 Enhanced Diffusion with Confidence-Based Re-masking",
        xaxis_title="Generation Step",
        yaxis_title="Masked/Re-masked Tokens (%)",
        yaxis2=dict(
            title="Average Confidence",
            overlaying='y',
            side='right'
        ),
        template="plotly_dark",
        height=500
    )
    
    return fig

def create_progress_chart(steps_data):
    """Create interactive progress chart"""
    if not steps_data:
        return None
        
    df = pd.DataFrame(steps_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['masked_percentage'],
        mode='lines+markers',
        name='Masked Tokens %',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['avg_confidence'],
        mode='lines+markers',
        name='Avg Confidence',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="🔮 Diffusion Denoising Progress",
        xaxis_title="Denoising Step",
        yaxis_title="Masked Tokens (%)",
        yaxis2=dict(
            title="Average Confidence",
            overlaying='y',
            side='right'
        ),
        template="plotly_dark",
        height=400
    )
    
    return fig

@torch.no_grad()
def generate_with_confidence_remasking(model, tokenizer, device, input_text="", generation_mode="Unconditional (Random Generation)", 
                                     max_len=64, timesteps=50, temperature=1.0, top_k=50, top_p=0.9,
                                     confidence_threshold=0.7, remask_ratio=0.2, max_refinement_steps=20, 
                                     min_confidence_improvement=0.02):
    """
    Enhanced generation with confidence-based re-masking for iterative quality improvement
    
    Args:
        confidence_threshold: Minimum confidence to keep a token (default: 0.7)
        remask_ratio: Fraction of low-confidence tokens to re-mask each iteration (default: 0.2)
        max_refinement_steps: Maximum number of refinement iterations (default: 20)
        min_confidence_improvement: Minimum improvement needed to continue iterations (default: 0.02)
    """
    model.eval()
    
    # Initialize containers
    step_container = st.empty()
    chart_container = st.empty()
    confidence_container = st.empty()
    stats_container = st.empty()
    
    # Prepare input sequence
    x, fixed_positions = prepare_input_sequence(input_text, tokenizer, max_len, generation_mode)
    x = x.to(device)
    
    steps_data = []
    progress_bar = st.progress(0)
    mask_token_id = tokenizer.get_mask_token_id()
    
    # Show initial state
    st.markdown("### 🎬 Initial State")
    html_tokens = []
    
    for i, token_id in enumerate(x[0]):
        token = tokenizer.id_to_token.get(token_id.item(), '<UNK>')
        if token == '<PAD>':
            continue
            
        if i in fixed_positions:
            html_tokens.append(f'<span class="token-input">{token}</span>')
        elif token_id.item() == mask_token_id:
            html_tokens.append(f'<span class="token-masked">[MASK]</span>')
        else:
            html_tokens.append(f'<span class="token-final">{token}</span>')
    
    st.markdown(' '.join(html_tokens), unsafe_allow_html=True)
    time.sleep(1)
    
    # Phase 1: Initial reverse diffusion process
    st.markdown("### 🔄 Phase 1: Initial Diffusion Denoising")
    total_steps = 0
    
    for t in reversed(range(timesteps)):
        timestep = torch.tensor([t], device=device)
        
        with torch.no_grad():
            logits = model(x, timestep)
        
        logits = logits / temperature
        
        # Store confidence scores for each position
        position_confidences = {}
        
        # Apply top-k and top-p sampling for each position
        for pos in range(max_len):
            if x[0, pos] == mask_token_id and pos not in fixed_positions:
                pos_logits = logits[0, pos].clone()
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(pos_logits, min(top_k, pos_logits.size(-1)))
                    pos_logits = torch.full_like(pos_logits, float('-inf'))
                    pos_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(pos_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    pos_logits[indices_to_remove] = float('-inf')
                
                # Calculate confidence before sampling
                probs = torch.softmax(pos_logits, dim=-1)
                max_prob = torch.max(probs).item()
                position_confidences[pos] = max_prob
                
                # Adaptive unmasking probability based on timestep and confidence
                unmask_prob = 0.15 + 0.25 * (1 - t / timesteps)  # Increase probability as we progress
                if max_prob > 0.8:  # High confidence - more likely to unmask
                    unmask_prob *= 1.5
                elif max_prob < 0.3:  # Low confidence - less likely to unmask
                    unmask_prob *= 0.5
                
                if np.random.random() < unmask_prob:
                    next_token = torch.multinomial(probs, 1)
                    x[0, pos] = next_token
        
        # Calculate statistics
        mask_count = sum(1 for i, t_id in enumerate(x[0]) if t_id == mask_token_id and i not in fixed_positions)
        total_tokens = len([t_id for t_id in x[0] if t_id != tokenizer.get_pad_token_id()])
        masked_percentage = (mask_count / max(total_tokens, 1)) * 100
        
        # Calculate average confidence
        if position_confidences:
            avg_confidence = np.mean(list(position_confidences.values())) * 100
        else:
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            avg_confidence = max_probs.mean().item() * 100
        
        total_steps += 1
        steps_data.append({
            'step': total_steps,
            'masked_percentage': masked_percentage,
            'avg_confidence': avg_confidence,
            'phase': 'Initial Diffusion'
        })
        
        progress = (timesteps - t) / timesteps * 0.6  # Phase 1 takes 60% of progress
        progress_bar.progress(progress)
        
        # Visualize current step
        if t % max(1, timesteps // 8) == 0 or t < 5:
            html_tokens = []
            for i, token_id in enumerate(x[0]):
                token = tokenizer.id_to_token.get(token_id.item(), '<UNK>')
                if token == '<PAD>':
                    continue
                
                is_fixed = i in fixed_positions
                css_class = get_token_color_class(token, t, timesteps, is_fixed=is_fixed)
                
                if token_id.item() == mask_token_id:
                    html_tokens.append(f'<span class="{css_class}">[MASK]</span>')
                else:
                    html_tokens.append(f'<span class="{css_class}">{token}</span>')
            
            with step_container.container():
                st.markdown(f"### 🔄 Phase 1 - Step {timesteps - t}/{timesteps}")
                st.markdown(f"**Progress:** {progress*100:.1f}% | **Masked:** {mask_count} | **Avg Confidence:** {avg_confidence:.1f}%")
                st.markdown(' '.join(html_tokens), unsafe_allow_html=True)
            
            if len(steps_data) > 1:
                fig = create_enhanced_progress_chart(steps_data)
                if fig:
                    chart_container.plotly_chart(fig, use_container_width=True)
            
            time.sleep(0.2)
    
    # Phase 2: Confidence-based refinement
    st.markdown("### 🎯 Phase 2: Confidence-Based Refinement")
    
    prev_avg_confidence = 0
    refinement_step = 0
    
    for refinement_iter in range(max_refinement_steps):
        # Get current state confidence scores
        timestep = torch.tensor([0], device=device)  # Use timestep 0 for refinement
        
        with torch.no_grad():
            logits = model(x, timestep)
        
        logits = logits / temperature
        
        # Calculate confidence for all positions
        position_confidences = {}
        low_confidence_positions = []
        
        for pos in range(max_len):
            if pos not in fixed_positions and x[0, pos] != mask_token_id:
                pos_logits = logits[0, pos]
                probs = torch.softmax(pos_logits, dim=-1)
                
                # Get confidence for current token
                current_token_id = x[0, pos].item()
                if current_token_id < len(probs):
                    confidence = probs[current_token_id].item()
                else:
                    confidence = 0.0
                
                position_confidences[pos] = confidence
                
                # Mark low confidence positions for potential re-masking
                if confidence < confidence_threshold:
                    low_confidence_positions.append((pos, confidence))
        
        # Calculate current average confidence
        if position_confidences:
            current_avg_confidence = np.mean(list(position_confidences.values())) * 100
        else:
            current_avg_confidence = 100.0  # No tokens to evaluate
        
        # Check if we should continue refinement
        confidence_improvement = current_avg_confidence - prev_avg_confidence
        
        if (confidence_improvement < min_confidence_improvement and refinement_iter > 2) or not low_confidence_positions:
            st.info(f"🎉 Refinement converged after {refinement_iter} steps! Confidence improvement: {confidence_improvement:.3f}%")
            break
        
        # Re-mask lowest confidence tokens
        if low_confidence_positions:
            # Sort by confidence (lowest first)
            low_confidence_positions.sort(key=lambda x: x[1])
            
            # Re-mask a fraction of lowest confidence tokens
            num_to_remask = max(1, int(len(low_confidence_positions) * remask_ratio))
            positions_to_remask = [pos for pos, conf in low_confidence_positions[:num_to_remask]]
            
            # Re-mask selected positions
            for pos in positions_to_remask:
                x[0, pos] = mask_token_id
            
            # Now regenerate the re-masked positions
            for pos in positions_to_remask:
                pos_logits = logits[0, pos].clone()
                
                # Apply sampling constraints
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(pos_logits, min(top_k, pos_logits.size(-1)))
                    pos_logits = torch.full_like(pos_logits, float('-inf'))
                    pos_logits.scatter_(0, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(pos_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    pos_logits[indices_to_remove] = float('-inf')
                
                # Sample new token with slightly lower temperature for refinement
                probs = torch.softmax(pos_logits / (temperature * 0.8), dim=-1)
                next_token = torch.multinomial(probs, 1)
                x[0, pos] = next_token
        else:
            positions_to_remask = []
        
        # Update statistics
        mask_count = sum(1 for i, t_id in enumerate(x[0]) if t_id == mask_token_id and i not in fixed_positions)
        masked_percentage = (mask_count / max(total_tokens, 1)) * 100
        
        total_steps += 1
        refinement_step += 1
        steps_data.append({
            'step': total_steps,
            'masked_percentage': masked_percentage,
            'avg_confidence': current_avg_confidence,
            'phase': 'Confidence Refinement'
        })
        
        progress = 0.6 + (refinement_iter / max_refinement_steps) * 0.4
        progress_bar.progress(progress)
        
        # Visualize refinement step
        html_tokens = []
        for i, token_id in enumerate(x[0]):
            token = tokenizer.id_to_token.get(token_id.item(), '<UNK>')
            if token == '<PAD>':
                continue
            
            is_fixed = i in fixed_positions
            
            # Color based on confidence
            if is_fixed:
                css_class = 'token-input'
            elif token_id.item() == mask_token_id:
                css_class = 'token-masked'
            else:
                confidence = position_confidences.get(i, 1.0)
                css_class = get_confidence_color_class(confidence, is_fixed)
            
            if token_id.item() == mask_token_id:
                html_tokens.append(f'<span class="{css_class}">[MASK]</span>')
            else:
                confidence_val = position_confidences.get(i, 1.0)
                html_tokens.append(f'<span class="{css_class}" title="Confidence: {confidence_val:.3f}">{token}</span>')
        
        with step_container.container():
            st.markdown(f"### 🎯 Phase 2 - Refinement Step {refinement_step}/{max_refinement_steps}")
            st.markdown(f"**Avg Confidence:** {current_avg_confidence:.2f}% | **Improvement:** +{confidence_improvement:.3f}% | **Re-masked:** {len(positions_to_remask)}")
            st.markdown(' '.join(html_tokens), unsafe_allow_html=True)
        
        # Update confidence display
        with confidence_container.container():
            st.markdown("### 📊 Confidence Analysis")
            if position_confidences:
                conf_values = list(position_confidences.values())
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Confidence", f"{current_avg_confidence:.2f}%")
                with col2:
                    st.metric("Min Confidence", f"{min(conf_values)*100:.2f}%")
                with col3:
                    st.metric("Max Confidence", f"{max(conf_values)*100:.2f}%")
                with col4:
                    low_conf_count = sum(1 for c in conf_values if c < confidence_threshold)
                    st.metric("Low Confidence Tokens", low_conf_count)
        
        # Update chart
        if len(steps_data) > 1:
            fig = create_enhanced_progress_chart(steps_data)
            if fig:
                chart_container.plotly_chart(fig, use_container_width=True)
        
        prev_avg_confidence = current_avg_confidence
        time.sleep(0.4)
    
    # Final statistics
    with stats_container.container():
        st.markdown("### 📈 Final Generation Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Steps", total_steps)
        with col2:
            st.metric("Refinement Steps", refinement_step)
        with col3:
            final_confidence = steps_data[-1]['avg_confidence'] if steps_data else 0
            st.metric("Final Confidence", f"{final_confidence:.2f}%")
        with col4:
            generation_time = total_steps * 0.3
            st.metric("Generation Time", f"{generation_time:.1f}s")
    
    # Generate final result
    final_text = tokenizer.decode(x[0].cpu().tolist())
    
    return final_text, steps_data

@torch.no_grad()
def generate_with_visualization(model, tokenizer, device, input_text="", generation_mode="Unconditional (Random Generation)", 
                              max_len=64, timesteps=50, temperature=1.0, top_k=50, top_p=0.9):
    """Original generation with step-by-step visualization"""
    model.eval()
    
    # Initialize containers
    step_container = st.empty()
    chart_container = st.empty()
    
    # Prepare input sequence
    x, fixed_positions = prepare_input_sequence(input_text, tokenizer, max_len, generation_mode)
    x = x.to(device)
    
    steps_data = []
    progress_bar = st.progress(0)
    
    # Show initial state
    st.markdown("### 🎬 Initial State")
    html_tokens = []
    mask_token_id = tokenizer.get_mask_token_id()
    
    for i, token_id in enumerate(x[0]):
        token = tokenizer.id_to_token.get(token_id.item(), '<UNK>')
        if token == '<PAD>':
            continue
            
        if i in fixed_positions:
            html_tokens.append(f'<span class="token-input">{token}</span>')
        elif token_id.item() == mask_token_id:
            html_tokens.append(f'<span class="token-masked">[MASK]</span>')
        else:
            html_tokens.append(f'<span class="token-final">{token}</span>')
    
    st.markdown(' '.join(html_tokens), unsafe_allow_html=True)
    time.sleep(1)
    
            # Reverse diffusion process
    for t in reversed(range(timesteps)):
        timestep = torch.tensor([t], device=device)
        
        with torch.no_grad():
            logits = model(x, timestep)
        
        logits = logits / temperature
        
        # Apply top-k and top-p sampling for each position
        for pos in range(max_len):
            if x[0, pos] == mask_token_id and pos not in fixed_positions:
                pos_logits = logits[0, pos]
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(pos_logits, min(top_k, pos_logits.size(-1)))
                    pos_logits = torch.full_like(pos_logits, float('-inf'))
                    pos_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(pos_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    pos_logits[indices_to_remove] = float('-inf')
                
                # Sample with some probability of unmasking
                if np.random.random() < 0.3:  # 30% chance to unmask at each step
                    probs = torch.softmax(pos_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    x[0, pos] = next_token
        
        # Calculate statistics
        mask_count = sum(1 for i, t_id in enumerate(x[0]) if t_id == mask_token_id and i not in fixed_positions)
        total_tokens = len([t_id for t_id in x[0] if t_id != tokenizer.get_pad_token_id()])
        masked_percentage = (mask_count / max(total_tokens, 1)) * 100
        
        # Calculate average confidence (approximation)
        probs = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        avg_confidence = max_probs.mean().item() * 100
        
        steps_data.append({
            'step': timesteps - t,
            'masked_percentage': masked_percentage,
            'avg_confidence': avg_confidence
        })
        
        progress = (timesteps - t) / timesteps
        progress_bar.progress(progress)
        
        # Visualize current step
        if t % max(1, timesteps // 10) == 0 or t < 5:
            html_tokens = []
            for i, token_id in enumerate(x[0]):
                token = tokenizer.id_to_token.get(token_id.item(), '<UNK>')
                if token == '<PAD>':
                    continue
                
                is_fixed = i in fixed_positions
                css_class = get_token_color_class(token, t, timesteps, is_fixed=is_fixed)
                
                if token_id.item() == mask_token_id:
                    html_tokens.append(f'<span class="{css_class}">[MASK]</span>')
                else:
                    html_tokens.append(f'<span class="{css_class}">{token}</span>')
            
            with step_container.container():
                st.markdown(f"### 🔄 Step {timesteps - t}/{timesteps}")
                st.markdown(f"**Progress:** {progress*100:.1f}% | **Masked:** {mask_count} | **Fixed:** {len(fixed_positions)}")
                st.markdown(' '.join(html_tokens), unsafe_allow_html=True)
            
            if len(steps_data) > 1:
                fig = create_progress_chart(steps_data)
                if fig:
                    chart_container.plotly_chart(fig, use_container_width=True)
            
            time.sleep(0.3)
    
    # Generate final result
    final_text = tokenizer.decode(x[0].cpu().tolist())
    
    return final_text, steps_data

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer (cached)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check for model files
    model_files = [
        # 'marathi_diffusion_best.pt',
        # 'marathi_diffusion_optimal_final.pt',
        # 'optimal_checkpoint_epoch_095.pt',
        'optimal_checkpoint_epoch_135.pt',
    ]
    
    tokenizer_file = 'tokenizer_optimal.pkl'
    
    model_path = None
    for file in model_files:
        if os.path.exists(file):
            model_path = file
            break
    
    if not model_path or not os.path.exists(tokenizer_file):
        return None, None, None
    
    # Load tokenizer
    tokenizer = MarathiBPETokenizer()
    tokenizer.load(tokenizer_file)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Fallback configuration
        model_config = {
            'vocab_size': tokenizer.vocab_size,
            'd_model': 320,
            'n_heads': 8,
            'n_layers': 12,
            'd_ff': 1280,
            'max_len': 256
        }
    
    model = DiffusionLM(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    timesteps = checkpoint.get('timesteps', 50)
    
    return model, tokenizer, timesteps

def main():
    # Header
    st.markdown('<h1 class="main-header">🔮Diffusion Language Model (Prototype)</h1>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer, default_timesteps = load_model_and_tokenizer()
    
    if model is None:
        st.error("❌ Model files not found! Please ensure you have the trained model files.")
        st.info("""
        Required files:
        - `marathi_diffusion_best.pt` OR `marathi_diffusion_optimal_final.pt`
        - `tokenizer_optimal.pkl`
        - `tokenizer_optimal_hf.json` (optional but recommended)
        """)
        return
    
    # Input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    st.markdown("### 📝 Text Input & Generation Mode")
    
    # Generation mode selection
    generation_mode = st.selectbox(
        "🎯 Select Generation Mode",
        [
            "Unconditional (Random Generation)",
            "Text Completion", 
            "Fill in the Blanks",
            "Conditioned Generation"
        ],
        help="Choose how you want to use your input text"
    )
    
    # Input text area
    if generation_mode == "Unconditional (Random Generation)":
        st.info("💡 In this mode, the model generates completely random Marathi text. Input text will be ignored.")
        input_text = ""
    elif generation_mode == "Text Completion":
        input_text = st.text_area(
            "📝 Enter text to complete:",
            placeholder="मला वाटतं की...",
            help="The model will complete your text"
        )
    elif generation_mode == "Fill in the Blanks":
        input_text = st.text_area(
            "📝 Enter text with [MASK] tokens:",
            placeholder="मी [MASK] जाणार आहे आणि [MASK] करणार आहे।",
            help="Use [MASK] to indicate positions where the model should fill in words"
        )
    else:  # Conditioned Generation
        input_text = st.text_area(
            "📝 Enter conditioning text:",
            placeholder="प्रेम",
            help="The model will generate text related to or continuing from your input"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Generation Controls")
    
    # Model info
    st.sidebar.markdown("### 📊 Model Info")
    device = next(model.parameters()).device
    total_params = sum(p.numel() for p in model.parameters())
    st.sidebar.info(f"**Device:** {device}")
    st.sidebar.info(f"**Vocab Size:** {tokenizer.vocab_size:,}")
    st.sidebar.info(f"**Model Parameters:** {total_params:,}")
    
    # Generation parameters
    st.sidebar.markdown("### ⚙️ Parameters")
    
    max_length = st.sidebar.slider(
        "🔢 Text Length", 
        min_value=16, 
        max_value=256, 
        value=128, 
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "🌡️ Temperature", 
        min_value=0.1, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="Controls randomness: Lower = more focused, Higher = more creative"
    )
    
    top_k = st.sidebar.slider(
        "🎯 Top-k", 
        min_value=0, 
        max_value=100, 
        value=50, 
        help="Limit vocabulary to top k tokens (0 = disabled)"
    )
    
    top_p = st.sidebar.slider(
        "🎪 Top-p", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.9, 
        step=0.05,
        help="Nucleus sampling threshold"
    )
    
    timesteps = st.sidebar.slider(
        "⏰ Diffusion Steps", 
        min_value=10, 
        max_value=100, 
        value=default_timesteps, 
        step=5,
        help="Number of denoising steps: More steps = better quality but slower"
    )
    
    # Enhanced sidebar controls for confidence refinement
    st.sidebar.markdown("### 🎯 Confidence Refinement")
    
    enable_refinement = st.sidebar.checkbox(
        "Enable Confidence-Based Refinement", 
        value=True,
        help="Re-mask and improve low-confidence tokens after initial generation"
    )
    
    if enable_refinement:
        confidence_threshold = st.sidebar.slider(
            "🎯 Confidence Threshold", 
            min_value=0.3, 
            max_value=0.9, 
            value=0.7, 
            step=0.05,
            help="Minimum confidence to keep a token (lower = more aggressive re-masking)"
        )
        
        remask_ratio = st.sidebar.slider(
            "🔄 Re-mask Ratio", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.2, 
            step=0.05,
            help="Fraction of low-confidence tokens to re-mask each iteration"
        )
        
        max_refinement_steps = st.sidebar.slider(
            "🔢 Max Refinement Steps", 
            min_value=5, 
            max_value=30, 
            value=15, 
            step=5,
            help="Maximum refinement iterations"
        )
        
        min_improvement = st.sidebar.slider(
            "📈 Min Improvement", 
            min_value=0.001, 
            max_value=0.05, 
            value=0.02, 
            step=0.005,
            format="%.3f",
            help="Minimum confidence improvement to continue refinement"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Live Generation")
        
        # Generate button
        if st.button("🚀 Generate Text", type="primary", use_container_width=True):
            with st.spinner("🔄 Initializing enhanced diffusion process..."):
                
                # Show input info
                if input_text and generation_mode != "Unconditional (Random Generation)":
                    st.markdown("### 📄 Input Text")
                    st.info(f"**Mode:** {generation_mode}")
                    st.code(input_text, language="text")
                
                # Generate text with enhanced method
                if enable_refinement:
                    final_text, steps_data = generate_with_confidence_remasking(
                        model, tokenizer, device, input_text, generation_mode, 
                        max_length, timesteps, temperature, top_k, top_p,
                        confidence_threshold, remask_ratio, max_refinement_steps, min_improvement
                    )
                    
                    st.markdown("### ✨ Final Enhanced Generated Text")
                    st.success(f"**{final_text}**")
                    
                    # Show enhancement info
                    phase2_steps = [s for s in steps_data if s['phase'] == 'Confidence Refinement']
                    if phase2_steps:
                        st.info(f"🎯 **Enhancement Applied:** {len(phase2_steps)} refinement steps improved the text quality!")
                else:
                    # Use original method
                    final_text, steps_data = generate_with_visualization(
                        model, tokenizer, device, input_text, generation_mode, 
                        max_length, timesteps, temperature, top_k, top_p
                    )
                    
                    st.markdown("### ✨ Final Generated Text")
                    st.success(f"**{final_text}**")
    
    with col2:
        st.markdown("### 🎨 Enhanced Color Legend")
        st.markdown("""
        <div style="padding: 1rem; border-radius: 10px; background: #f0f2f6;">
            <p><strong>Phase 1 - Initial Diffusion:</strong></p>
            <p><span class="token-input">Input</span> - Your input text (fixed)</p>
            <p><span class="token-masked">[MASK]</span> - Masked tokens</p>
            <p><span class="token-early">Token</span> - Early denoising</p>
            <p><span class="token-mid">Token</span> - Mid denoising</p>
            <p><span class="token-late">Token</span> - Late denoising</p>
            <p><span class="token-final">Token</span> - Final tokens</p>
            
            <p><strong>Phase 2 - Confidence Refinement:</strong></p>
            <p>Colors represent confidence levels:</p>
            <p><span class="token-final">High</span> - Confident (80%+)</p>
            <p><span class="token-late">Good</span> - Acceptable (60-80%)</p>
            <p><span class="token-mid">Medium</span> - Moderate (40-60%)</p>
            <p><span class="token-early">Low</span> - Uncertain (20-40%)</p>
            <p><span class="token-masked">Very Low</span> - Re-mask candidate (<20%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Generation Modes")
        st.markdown("""
        **🎲 Unconditional:** Random text generation
        
        **📝 Text Completion:** Complete your partial text
        
        **🎭 Fill in Blanks:** Use [MASK] tokens to specify what to fill
        
        **🎪 Conditioned:** Generate text related to your input
        """)
        
        st.markdown("### 🔬 How Enhanced Model Works")
        st.markdown("""
        **Phase 1 - Initial Diffusion:**
        1. 🎲 Start with masked tokens (except input)
        2. 🧠 Model predicts what each position should be
        3. 🎯 Sample from predictions using temperature/top-k/top-p
        4. 🔄 Repeat denoising for specified timesteps
        
        **Phase 2 - Confidence Refinement:**
        1. 📊 Calculate confidence for each generated token
        2. 🎯 Identify low-confidence tokens (below threshold)
        3. 🔄 Re-mask a fraction of low-confidence tokens
        4. 🧠 Re-generate with refined sampling
        5. 📈 Repeat until convergence or max steps
        6. ✨ Final high-quality Marathi text!
        
        **Key Enhancements:**
        - **Iterative Quality Improvement**: Continuously refines output
        - **Confidence-based Selection**: Targets uncertain predictions
        - **Adaptive Re-masking**: Smart token replacement strategy
        - **Convergence Detection**: Stops when no further improvement
        """)
        
        # Model architecture details
        if model and tokenizer:
            st.markdown("### 🏗️ Model Architecture")
            config_info = f"""
            - **Vocab Size:** {tokenizer.vocab_size:,}
            - **Model Dim:** {model.d_model}
            - **Attention Heads:** {model.layers[0].attention.num_heads}
            - **Layers:** {len(model.layers)}
            - **Max Length:** {model.max_len}
            - **Total Parameters:** {sum(p.numel() for p in model.parameters()):,}
            """
            st.markdown(config_info)

    # Footer with enhanced information
    st.markdown("---")
    
    # Quick test section with enhanced options
    with st.expander("🧪 Quick Test Generation"):
        st.markdown("Test your enhanced model with some quick examples:")
        
        col_test1, col_test2, col_test3 = st.columns(3)
        
        with col_test1:
            if st.button("🎲 Random Generation", use_container_width=True):
                with st.spinner("Generating random text..."):
                    if enable_refinement:
                        quick_result, _ = generate_with_confidence_remasking(
                            model, tokenizer, device, 
                            "", "Unconditional (Random Generation)", 
                            64, 25, 1.0, 50, 0.9, 0.7, 0.2, 10, 0.02
                        )
                    else:
                        quick_result, _ = generate_with_visualization(
                            model, tokenizer, device, 
                            "", "Unconditional (Random Generation)", 
                            64, 25, 1.0, 50, 0.9
                        )
                    st.write(f"**Result:** {quick_result}")
        
        with col_test2:
            quick_prompt = st.text_input("Quick completion test:", placeholder="मराठी भाषा")
            if st.button("⚡ Quick Complete", use_container_width=True) and quick_prompt:
                with st.spinner("Completing text..."):
                    if enable_refinement:
                        quick_result, _ = generate_with_confidence_remasking(
                            model, tokenizer, device, 
                            quick_prompt, "Text Completion", 
                            64, 25, 0.8, 50, 0.9, 0.7, 0.2, 10, 0.02
                        )
                    else:
                        quick_result, _ = generate_with_visualization(
                            model, tokenizer, device, 
                            quick_prompt, "Text Completion", 
                            64, 25, 0.8, 50, 0.9
                        )
                    st.write(f"**Result:** {quick_result}")
        
        with col_test3:
            if st.button("🎭 Fill in Blanks Test", use_container_width=True):
                test_text = "मी [MASK] करतो आणि [MASK] आवडतं"
                with st.spinner("Filling blanks..."):
                    if enable_refinement:
                        quick_result, _ = generate_with_confidence_remasking(
                            model, tokenizer, device, 
                            test_text, "Fill in the Blanks", 
                            64, 20, 0.9, 30, 0.8, 0.6, 0.3, 8, 0.03
                        )
                    else:
                        quick_result, _ = generate_with_visualization(
                            model, tokenizer, device, 
                            test_text, "Fill in the Blanks", 
                            64, 20, 0.9, 30, 0.8
                        )
                    st.write(f"**Result:** {quick_result}")
    
    # Advanced settings with refinement options
    with st.expander("⚙️ Advanced Settings & Enhanced Diagnostics"):
        st.markdown("### 🔍 Model Diagnostics")
        
        col_diag1, col_diag2 = st.columns(2)
        
        with col_diag1:
            if st.button("🧠 Test Model Forward Pass"):
                with st.spinner("Testing model..."):
                    try:
                        # Test forward pass
                        test_input = torch.randint(0, tokenizer.vocab_size, (1, 32)).to(device)
                        test_timesteps = torch.randint(0, 50, (1,)).to(device)
                        
                        with torch.no_grad():
                            test_output = model(test_input, test_timesteps)
                        
                        st.success(f"✅ Model working! Output shape: {test_output.shape}")
                        st.info(f"Input shape: {test_input.shape}, Timesteps: {test_timesteps.item()}")
                        
                    except Exception as e:
                        st.error(f"❌ Model test failed: {e}")
        
        with col_diag2:
            if st.button("🔤 Test Tokenizer"):
                with st.spinner("Testing tokenizer..."):
                    try:
                        test_text = "मराठी भाषा"
                        encoded = tokenizer.encode(test_text)
                        decoded = tokenizer.decode(encoded)
                        
                        st.success("✅ Tokenizer working!")
                        st.info(f"Original: {test_text}")
                        st.info(f"Encoded: {encoded}")
                        st.info(f"Decoded: {decoded}")
                        
                    except Exception as e:
                        st.error(f"❌ Tokenizer test failed: {e}")
        
        st.markdown("### 🎛️ Enhanced Generation Strategy")
        
        strategy_info = f"""
        **Current Configuration:**
        - Temperature: {temperature} (randomness control)
        - Top-k: {top_k} (vocabulary filtering) 
        - Top-p: {top_p} (nucleus sampling)
        - Timesteps: {timesteps} (denoising steps)
        - Max Length: {max_length} tokens
        
        **Confidence Refinement Settings:**
        - Enabled: {enable_refinement}
        {f"- Confidence Threshold: {confidence_threshold}" if enable_refinement else ""}
        {f"- Re-mask Ratio: {remask_ratio}" if enable_refinement else ""}
        {f"- Max Refinement Steps: {max_refinement_steps}" if enable_refinement else ""}
        {f"- Min Improvement: {min_improvement}" if enable_refinement else ""}
        
        **Enhanced Generation Tips:**
        - Lower confidence threshold = more aggressive refinement
        - Higher re-mask ratio = more tokens reconsidered per iteration
        - More refinement steps = potentially higher quality but slower
        - Confidence refinement works best with temperature 0.7-1.2
        """
        
        st.markdown(strategy_info)
        
        # Memory usage info
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            st.markdown(f"""
            **GPU Memory Usage:**
            - Allocated: {memory_allocated:.2f} GB
            - Reserved: {memory_reserved:.2f} GB
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🔮 <strong>Enhanced Marathi Diffusion Language Model</strong><br>
        Built with PyTorch • Streamlit • Advanced Confidence Refinement • ❤️<br>
        <em>Bringing iterative quality improvement to Marathi text generation</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()