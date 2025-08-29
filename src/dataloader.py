"""
Dataloader Module
====================

This module provides dataset and batching utilities for training
the DAC-based loop morphing model. It handles embeddings,
style probabilities, BPM information, and morph ratios, ensuring
compatibility with variable-length audio embeddings.

Features
--------
- custom_collate_fn:
  - Handles batching of loop embeddings, style probabilities, BPM, and morph ratios.
  - Automatically pads variable-length DAC embeddings to the longest sequence in the batch.
  - Stores sequence lengths for masking in the model.
- ImprovedDACLoopMorphingDataset:
  - Loads source and target embeddings with associated style and BPM metadata.
  - Provides robust file loading and optional sequence length restriction.
  - Produces training samples suitable for morphing tasks (source → target).

Dependencies
------------
- Python standard library: os
- Third-party:
  - torch
  - numpy

Usage
-----
Example for dataset and dataloader:

    from torch.utils.data import DataLoader
    from dataloader import ImprovedDACLoopMorphingDataset, custom_collate_fn

    dataset = ImprovedDACLoopMorphingDataset(
        embeddings_dir="./data/embeddings",
        prob_dir="./data/style_probs",
        max_sequence_length=2048
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    for batch in dataloader:
        loop_embedding = batch["loop_embedding"]
        style_prob = batch["style_prob"]
        bpm = batch["bpm"]
        # ... training logic ...

"""

import torch
import numpy as np
import os
from torch.utils.data import Dataset

def custom_collate_fn(batch):
    """Enhanced collate function for the improved morphing model"""
    batch_data = {}

    # Handle all the standard fields
    for key in ['loop_embedding', 'style_prob', 'bpm', 'target_embedding', 'target_style_prob', 'target_bpm', 'morph_ratio']:
        if key not in batch[0]:
            continue
            
        tensors = [sample[key] for sample in batch]
        
        # For DAC embeddings, handle variable lengths
        if key in ['loop_embedding', 'target_embedding']:
            # Find max sequence length
            max_seq_len = max(t.shape[-1] for t in tensors)
            
            # Store sequence lengths for masking
            seq_len_key = 'seq_lengths' if key == 'loop_embedding' else 'target_seq_lengths'
            seq_lengths = torch.tensor([t.shape[-1] for t in tensors], dtype=torch.long)
            batch_data[seq_len_key] = seq_lengths

            # Pad all tensors to max length
            padded_tensors = []
            for t in tensors:
                if t.shape[-1] < max_seq_len:
                    # Pad the time dimension (last dimension)
                    padding = torch.zeros((t.shape[0], t.shape[1], max_seq_len - t.shape[-1]), dtype=t.dtype)
                    padded_t = torch.cat([t, padding], dim=-1)
                    padded_tensors.append(padded_t)
                else:
                    padded_tensors.append(t)
            batch_data[key] = torch.stack(padded_tensors)
        else:
            # For other tensors, stack if possible
            if all(t.shape == tensors[0].shape for t in tensors):
                batch_data[key] = torch.stack(tensors)
            else:
                batch_data[key] = tensors

    return batch_data


class ImprovedDACLoopMorphingDataset(Dataset):
    def __init__(self, embeddings_dir, prob_dir, max_sequence_length=None, robust_loading=True):
        """
        Enhanced dataset that provides both source and target sequences for proper morphing
        """
        self.embeddings_dir = embeddings_dir
        self.prob_dir = prob_dir
        self.max_sequence_length = max_sequence_length
        self.robust_loading = robust_loading
        
        # Get all DAC files
        self.dac_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_processed.dac')]
        self.base_names = [f.replace('_processed.dac', '') for f in self.dac_files]
        
        # Filter to ensure all required files exist
        if robust_loading:
            print("Verifying all files exist for each sample...")
            self.valid_files = []
            for name in self.base_names:
                dac_path = os.path.join(embeddings_dir, f"{name}_processed.dac")
                bpm_path = os.path.join(embeddings_dir, f"{name}_bpm.pt")
                prob_path = os.path.join(prob_dir, f"{name}_style_probs.pt")
                
                if os.path.exists(dac_path) and os.path.exists(bpm_path) and os.path.exists(prob_path):
                    try:
                        if os.path.getsize(dac_path) > 0:
                            self.valid_files.append(name)
                    except:
                        print(f"Skipping {name}: Cannot verify DAC file integrity")
        else:
            self.valid_files = []
            for name in self.base_names:
                dac_path = os.path.join(embeddings_dir, f"{name}_processed.dac")
                bpm_path = os.path.join(embeddings_dir, f"{name}_bpm.pt")
                prob_path = os.path.join(prob_dir, f"{name}_style_probs.pt")
                
                if os.path.exists(dac_path) and os.path.exists(bpm_path) and os.path.exists(prob_path):
                    self.valid_files.append(name)
        
        print(f"Found {len(self.valid_files)} valid files")
        # We'll create pairs of different files for morphing
        self.num_combinations = len(self.valid_files) * (len(self.valid_files) - 1)
        print(f"Total possible source-target combinations: {self.num_combinations}")

    def __len__(self):
        return min(self.num_combinations, 50000)  # Reasonable limit
    
    def load_file_data(self, name):
        """Load all data for a given file name with robust error handling"""
        try:
            # Load DAC codes
            dac_path = os.path.join(self.embeddings_dir, f"{name}_processed.dac")
            dac_data = np.load(dac_path, allow_pickle=True).item()
            
            if isinstance(dac_data, dict) and 'codes' in dac_data:
                codes = dac_data['codes']
                if codes.dtype == np.uint16:
                    codes = codes.astype(np.int64)
                
                # Remove batch dimension if present
                if len(codes.shape) == 3 and codes.shape[0] == 1:
                    codes = codes[0]
                
                codes = torch.from_numpy(codes).clone()
            else:
                raise ValueError("Invalid DAC format")
            
            # Load BPM
            bpm_path = os.path.join(self.embeddings_dir, f"{name}_bpm.pt")
            bpm = torch.load(bpm_path, map_location='cpu')
            if not isinstance(bpm, torch.Tensor):
                bpm = torch.tensor([bpm], dtype=torch.float32)
            else:
                bpm = bpm.clone().detach()
            
            # Ensure BPM shape is [1, 1]
            if len(bpm.shape) == 0:
                bpm = bpm.reshape(1, 1)
            elif len(bpm.shape) == 1:
                bpm = bpm.reshape(-1, 1)
            
            # Load style probabilities
            prob_path = os.path.join(self.prob_dir, f"{name}_style_probs.pt")
            style_prob = torch.load(prob_path, map_location='cpu')
            if not isinstance(style_prob, torch.Tensor):
                style_prob = torch.tensor(style_prob, dtype=torch.float32)
            else:
                style_prob = style_prob.clone().detach()
            
            # Ensure style_prob has proper shape
            if len(style_prob.shape) == 1:
                style_prob = style_prob.unsqueeze(0)
            
            return codes, bpm, style_prob
            
        except Exception as e:
            print(f"Error loading data for {name}: {str(e)}")
            return None, None, None

    def __getitem__(self, idx):
        # Convert idx to source and target indices
        n = len(self.valid_files)
        source_idx = idx // (n - 1)
        target_idx = idx % (n - 1)
        
        # Ensure target_idx != source_idx
        if target_idx >= source_idx:
            target_idx += 1
        
        # Ensure indices are within bounds
        source_idx = source_idx % n
        target_idx = target_idx % n
        
        source_name = self.valid_files[source_idx]
        target_name = self.valid_files[target_idx]
        
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Load source data
                source_codes, source_bpm, source_style = self.load_file_data(source_name)
                if source_codes is None:
                    source_idx = (source_idx + 1) % n
                    source_name = self.valid_files[source_idx]
                    continue
                
                # Load target data
                target_codes, target_bpm, target_style = self.load_file_data(target_name)
                if target_codes is None:
                    target_idx = (target_idx + 1) % n
                    if target_idx == source_idx:
                        target_idx = (target_idx + 1) % n
                    target_name = self.valid_files[target_idx]
                    continue
                
                # Verify shapes
                assert len(source_codes.shape) == 2, f"Expected 2D tensor for source_codes, got {source_codes.shape}"
                assert len(target_codes.shape) == 2, f"Expected 2D tensor for target_codes, got {target_codes.shape}"
                assert source_codes.shape[0] == 9, f"Expected 9 codebooks in source, got {source_codes.shape[0]}"
                assert target_codes.shape[0] == 9, f"Expected 9 codebooks in target, got {target_codes.shape[0]}"
                
                # NOTE: I use the curriculum during training, no need for this here 
                # Generate morph ratio with curriculum-friendly distribution
                # if random.random() < 0.4:
                #     # 40% extreme values for easier learning
                #     morph_ratio = torch.tensor(random.choice([0.0, 1.0]), dtype=torch.float32).reshape(1)
                # else:
                #     # 60% distributed values
                #     morph_ratio = torch.tensor(random.random(), dtype=torch.float32).reshape(1)
                morph_ratio = torch.tensor([0.5], dtype=torch.float32)  # Will be overridden in training

                
                return {
                    'loop_embedding': source_codes,
                    'style_prob': source_style,
                    'bpm': source_bpm,
                    'target_embedding': target_codes,
                    'target_style_prob': target_style,
                    'target_bpm': target_bpm,
                    'morph_ratio': morph_ratio # Will be overridden in training
                }
                
            except Exception as e:
                print(f"Error with data for {source_name}, {target_name} (attempt {attempt+1}): {str(e)}")
                source_idx = (source_idx + 1) % n
                target_idx = (target_idx + 1) % n
                if target_idx == source_idx:
                    target_idx = (target_idx + 1) % n
                source_name = self.valid_files[source_idx]
                target_name = self.valid_files[target_idx]
                continue
        
        # Fallback dummy data
        print(f"Failed to load valid data after {max_attempts} attempts")
        dummy_codes = torch.zeros((9, 100))
        dummy_style = torch.zeros((1, 400))
        dummy_bpm = torch.tensor([[120.0]])
        dummy_morph = torch.tensor([0.5]).reshape(1)
        
        return {
            'loop_embedding': dummy_codes,
            'style_prob': dummy_style,
            'bpm': dummy_bpm,
            'target_embedding': dummy_codes,
            'target_style_prob': dummy_style,
            'target_bpm': dummy_bpm,
            'morph_ratio': dummy_morph
        }