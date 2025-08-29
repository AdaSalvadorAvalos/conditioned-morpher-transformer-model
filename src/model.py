"""
Model Module
==============

This module implements neural network layers and conditioning mechanisms
for music/audio modeling. It provides reusable building blocks such as
positional encodings, FiLM-based conditioning, and embeddings for style
and BPM features.

Features
--------
- PositionalEncoding: Standard sinusoidal position encoding for sequences.
- FiLMLayer: Feature-wise Linear Modulation (FiLM) for conditioning on auxiliary information.
- StyleConditioningEmbedding: Projects style probability vectors into embedding space.
- BPMConditioningEmbedding: Encodes tempo (BPM) into embeddings for model conditioning.

Global Parameters
-----------------
This module does not define global constants; all configuration is passed 
through class initializers.

Dependencies
------------
- Python standard library: math
- Third-party:
  - torch
  - torch.nn

Usage
-----
Import and use components in model architectures:

    from model import PositionalEncoding, FiLMLayer, StyleConditioningEmbedding

Example:

    d_model = 256
    seq_len = 100
    batch_size = 8

    pos_enc = PositionalEncoding(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    x_with_pe = pos_enc(x)

    film = FiLMLayer(feature_dim=256, condition_dim=64)
    condition = torch.randn(batch_size, 64)
    x_film = film(x_with_pe, condition)

"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning"""
    def __init__(self, feature_dim, condition_dim):
        super(FiLMLayer, self).__init__()
        # Generate scale (gamma) and shift (beta) parameters from conditions
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim),
            nn.ReLU(), 
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x, condition):
        """
        Apply FiLM conditioning: x_out = gamma * x + beta
        Args:
            x: [batch_size, seq_len, feature_dim] - input features
            condition: [batch_size, condition_dim] - conditioning vector
        """
        gamma = self.gamma_net(condition)  # [batch_size, feature_dim]
        beta = self.beta_net(condition)    # [batch_size, feature_dim]
        
        # Expand to match sequence dimension
        gamma = gamma.unsqueeze(1)  # [batch_size, 1, feature_dim]
        beta = beta.unsqueeze(1)    # [batch_size, 1, feature_dim]
        
        return gamma * x + beta

class StyleConditioningEmbedding(nn.Module):
    def __init__(self, style_dim, embed_dim):
        super(StyleConditioningEmbedding, self).__init__()
        self.style_embedding = nn.Linear(style_dim, embed_dim)

    def forward(self, style_prob):
        return self.style_embedding(style_prob)

class BPMConditioningEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(BPMConditioningEmbedding, self).__init__()
        self.bpm_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, bpm):
        return self.bpm_embedding(bpm)

class FiLMTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, style_dim, bpm_dim):
        super(FiLMTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Separate FiLM layers for style and BPM
        self.style_film1 = FiLMLayer(d_model, style_dim)
        self.style_film2 = FiLMLayer(d_model, style_dim)
        self.bpm_film1 = FiLMLayer(d_model, bpm_dim)
        self.bpm_film2 = FiLMLayer(d_model, bpm_dim)
        
        self.activation = nn.ReLU()

    def apply_combined_film(self, x, style_condition, bpm_condition, film_style, film_bpm):
        """Apply both FiLM layers and combine additively"""
        # Apply both FiLM layers to the original input
        style_modulated = film_style(x, style_condition)
        bpm_modulated = film_bpm(x, bpm_condition)
        
        # Combine: take the additive effect of both modulations
        # This preserves both conditioning signals
        combined = x + (style_modulated - x) + (bpm_modulated - x)
        return combined

    def forward(self, src, style_condition, bpm_condition, src_mask=None, src_key_padding_mask=None):
        # Self attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Apply combined FiLM conditioning
        src = self.apply_combined_film(src, style_condition, bpm_condition, 
                                     self.style_film1, self.bpm_film1)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # Apply combined FiLM conditioning
        src = self.apply_combined_film(src, style_condition, bpm_condition,
                                     self.style_film2, self.bpm_film2)
        
        return src

class FiLMTransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with separate FiLM conditioning for style and BPM"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, style_dim, bpm_dim):
        super(FiLMTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Separate FiLM layers for style and BPM conditioning
        self.style_film1 = FiLMLayer(d_model, style_dim)
        self.style_film2 = FiLMLayer(d_model, style_dim)
        self.style_film3 = FiLMLayer(d_model, style_dim)
        self.bpm_film1 = FiLMLayer(d_model, bpm_dim)
        self.bpm_film2 = FiLMLayer(d_model, bpm_dim)
        self.bpm_film3 = FiLMLayer(d_model, bpm_dim)

        self.activation = nn.ReLU()

    def apply_combined_film(self, x, style_condition, bpm_condition, film_style, film_bpm):
        """Apply both FiLM layers and combine additively"""
        # Apply both FiLM layers to the original input
        style_modulated = film_style(x, style_condition)
        bpm_modulated = film_bpm(x, bpm_condition)
        
        # Combine: take the additive effect of both modulations
        # This preserves both conditioning signals
        combined = x + (style_modulated - x) + (bpm_modulated - x)
        return combined

    def forward(self, tgt, memory, style_condition, bpm_condition, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Apply combined FiLM conditioning after self attention
        tgt = self.apply_combined_film(tgt, style_condition, bpm_condition, 
                                     self.style_film1, self.bpm_film1)
        
        # Cross attention
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Apply combined FiLM conditioning after cross attention
        tgt = self.apply_combined_film(tgt, style_condition, bpm_condition,
                                     self.style_film2, self.bpm_film2)
        
        # Feed forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        # Apply combined FiLM conditioning after feed forward
        tgt = self.apply_combined_film(tgt, style_condition, bpm_condition,
                                     self.style_film3, self.bpm_film3)
        
        return tgt

class SimpleDACMorpher(nn.Module):
    def __init__(self,
                 num_codebooks=9,
                 codebook_size=1024,
                 d_model=128,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=512,
                 dropout=0.2,
                 style_dim=400,
                 max_seq_len=3879):
        super().__init__()
        
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        
        # Token embeddings for each codebook
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, d_model) 
            for _ in range(num_codebooks)
        ])
        
        # Conditioning embeddings
        self.style_embedding = StyleConditioningEmbedding(style_dim, d_model)
        self.bpm_embedding = BPMConditioningEmbedding(d_model)
        
        # REDUNDANT: Store embedding dimensions for FiLM layers
        self.style_embed_dim = d_model
        self.bpm_embed_dim = d_model
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer Encoder with separate FiLM conditioning
        self.encoder_layers = nn.ModuleList([
            FiLMTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                style_dim=self.style_embed_dim,
                bpm_dim=self.bpm_embed_dim
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Latent projector
        self.latent_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer Decoder with separate FiLM conditioning
        self.decoder_layers = nn.ModuleList([
            FiLMTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                style_dim=self.style_embed_dim,
                bpm_dim=self.bpm_embed_dim
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Output heads for each codebook
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(num_codebooks)
        ])

    def prepare_conditioning(self, style_prob, bpm):
        """Prepare separate conditioning vectors for style and BPM"""
        # Shape fixes
        if len(style_prob.shape) == 3:
            style_prob = style_prob.squeeze(1)
        elif len(style_prob.shape) == 1:
            style_prob = style_prob.unsqueeze(0)
        
        if len(bpm.shape) == 3:
            bpm = bpm.squeeze(1)
        elif len(bpm.shape) == 1:
            bpm = bpm.unsqueeze(1)
        elif len(bpm.shape) == 0:
            bpm = bpm.unsqueeze(0).unsqueeze(1)
        
        # Get separate conditioning embeddings
        style_embed = self.style_embedding(style_prob)  # [batch_size, d_model]
        bpm_embed = self.bpm_embedding(bpm)             # [batch_size, d_model]
        
        return style_embed, bpm_embed

    def encode_sequence(self, codes, style_prob, bpm, seq_lengths=None):
        batch_size, num_codebooks, seq_len = codes.shape
        device = codes.device
        
        # Embed each codebook tokens
        all_embeds = []
        for i in range(num_codebooks):
            codebook_codes = codes[:, i, :].long()
            code_embeds = self.token_embeddings[i](codebook_codes)
            all_embeds.append(code_embeds)
        
        # Average embeddings across codebooks
        x = torch.stack(all_embeds, dim=0).mean(dim=0)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Prepare separate conditioning vectors
        style_condition, bpm_condition = self.prepare_conditioning(style_prob, bpm)
        
        # Create padding mask
        src_key_padding_mask = None
        if seq_lengths is not None:
            positions = torch.arange(0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
            src_key_padding_mask = ~(positions < seq_lengths.to(positions.device).unsqueeze(1))

        # Apply encoder layers with separate FiLM conditioning
        for layer in self.encoder_layers:
            x = layer(x, style_condition, bpm_condition, src_key_padding_mask=src_key_padding_mask)
        
        # Project to latent space
        latent = self.latent_projector(x)
        
        return latent, style_condition, bpm_condition

    def interpolate_latents(self, source_latent, target_latent, morph_ratio):
        # Handle different input shapes for morph_ratio
        if isinstance(morph_ratio, (int, float)):
            morph_ratio = torch.tensor(morph_ratio, device=source_latent.device, dtype=torch.float32)
        
        # Ensure proper tensor dimensions for broadcasting
        if len(morph_ratio.shape) == 0:  # scalar tensor
            morph_ratio = morph_ratio.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1]
        elif len(morph_ratio.shape) == 1:  # [B]
            morph_ratio = morph_ratio.unsqueeze(1).unsqueeze(1)  # [B, 1, 1]
        elif len(morph_ratio.shape) == 2:  # [B, 1]
            morph_ratio = morph_ratio.unsqueeze(2)  # [B, 1, 1]

        morphed_latent = (1 - morph_ratio) * source_latent + morph_ratio * target_latent
        return morphed_latent

    # def interpolate_conditions(self, source_style, source_bpm, target_style, target_bpm, morph_ratio):
    #     """Interpolate separate conditioning vectors"""
    #     if isinstance(morph_ratio, (int, float)):
    #         morph_ratio = torch.tensor(morph_ratio, device=source_style.device, dtype=torch.float32)
        
    #     # Ensure proper tensor dimensions for broadcasting
    #     if len(morph_ratio.shape) == 0:  # scalar tensor
    #         morph_ratio = morph_ratio.unsqueeze(0).unsqueeze(0)  # [1, 1]
    #     elif len(morph_ratio.shape) == 1:  # [B]
    #         morph_ratio = morph_ratio.unsqueeze(1)  # [B, 1]

    #     morphed_style = (1 - morph_ratio) * source_style + morph_ratio * target_style
    #     morphed_bpm = (1 - morph_ratio) * source_bpm + morph_ratio * target_bpm
    #     return morphed_style, morphed_bpm


    def decode_sequence(self, latent_representation, style_condition, bpm_condition, target_length):
        """
        Decode sequence with separate style and BPM conditioning
        
        Args:
            latent_representation: Encoded latent features [batch_size, seq_len, d_model]
            style_condition: Style conditioning vector [batch_size, d_model] 
            bpm_condition: BPM conditioning vector [batch_size, d_model]
            target_length: Length of target sequence to generate
        """
        batch_size, seq_len, _ = latent_representation.shape
        
        # Decoder input: zeros + positional encoding
        decoder_input = torch.zeros(batch_size, target_length, self.d_model, device=latent_representation.device)
        decoder_input = self.pos_encoder(decoder_input)
        
        # Apply decoder layers with separate FiLM conditioning
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input, latent_representation, style_condition, bpm_condition)
        
        # Output logits per codebook
        outputs = []
        for i in range(self.num_codebooks):
            logits = self.output_heads[i](decoder_input)
            outputs.append(logits)
        
        return outputs

    def forward(self, 
        source_codes, source_style, source_bpm,
        target_codes, target_style, target_bpm,
        morph_ratio=0.5,
        custom_style=None, 
        custom_bpm=None,
        source_seq_lengths=None, target_seq_lengths=None):
        """
        Forward pass with separate FiLM conditioning for style and BPM
        
        Args:
            morph_ratio: Interpolation ratio between source and target (0=source, 1=target)
            custom_style/custom_bpm: Optional custom conditioning (replaces target if provided)
        """
        
        # Encode source
        source_latent, source_style_condition, source_bpm_condition = self.encode_sequence(
            source_codes, source_style, source_bpm, source_seq_lengths
        )
        
        # Use custom values as target conditions if provided
        effective_target_style = custom_style if custom_style is not None else target_style
        effective_target_bpm = custom_bpm if custom_bpm is not None else target_bpm
        
        # Encode target with potentially custom conditions
        target_latent, target_style_condition, target_bpm_condition = self.encode_sequence(
            target_codes, effective_target_style, effective_target_bpm, target_seq_lengths
        )
        
        # Interpolate only the latents
        morphed_latent = self.interpolate_latents(source_latent, target_latent, morph_ratio)

        # Use effective conditioning directly (skip interpolation)
        # seq_len = source_latent.shape[1]
        morphed_style_condition = target_style_condition if custom_style is None else self.style_embedding(custom_style)
        morphed_bpm_condition = target_bpm_condition if custom_bpm is None else self.bpm_embedding(custom_bpm)

        
        # Decode with morphed latent and separate conditions
        target_length = target_codes.shape[-1]
        outputs = self.decode_sequence(morphed_latent, morphed_style_condition, morphed_bpm_condition, target_length)
        
        return outputs
