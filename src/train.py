"""
Training Module
===================

This module provides training utilities for the **SimpleDACMorpher** model, a Transformer-based 
morphing network with FiLM conditioning on style and BPM. It defines the full training loop, 
validation logic, curriculum learning strategy, and model checkpointing.

Features
--------
- Curriculum-based morph ratio scheduling (`get_curriculum_morph_ratio`)
- Custom morphing loss across multiple codebooks with optional sequence length masking
- Training loop with gradient accumulation and TensorBoard logging
- Validation with multiple morph ratios to ensure robustness
- Automatic best-model checkpoint saving
- Fault tolerance for bad batches during training/validation

Global Parameters
-----------------
- 'num_epochs' (int): Number of training epochs (default 100)
- 'learning_rate' (float): Optimizer learning rate (default 1e-4)
- 'device' (str): Training device, defaults to CUDA if available
- 'save_dir' (str): Directory to store checkpoints (default: "checkpoints")
- 'accumulation_steps' (int): Steps for gradient accumulation (default 1)
- 'start_epoch' (int): Starting epoch (useful for resuming training)

Dependencies
------------
- Python standard library: os, json, random
- PyTorch: torch, torch.nn, torch.utils.data (DataLoader, random_split)
- TensorBoard: torch.utils.tensorboard.SummaryWriter
- Local modules:
  - 'dataloader.ImprovedDACLoopMorphingDataset', 'dataloader.custom_collate_fn'
  - 'model.SimpleDACMorpher'

Usage
-----
Example training script:

from dataloader import ImprovedDACLoopMorphingDataset, custom_collate_fn
from model import SimpleDACMorpher
from train import train_simplified_dac_morpher
from torch.utils.data import DataLoader, random_split

# Prepare dataset
dataset = ImprovedDACLoopMorphingDataset(json_path="data/loops.json")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

# Initialize model
model = SimpleDACMorpher(...)

# Train
train_simplified_dac_morpher(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=50,
    learning_rate=1e-4,
    device="cuda"
)
"""


import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import json
import random
from dataloader import ImprovedDACLoopMorphingDataset, custom_collate_fn
from model import SimpleDACMorpher

def get_curriculum_morph_ratio(epoch, max_epochs, num_values=1):
    """
    Get curriculum-based morph ratio(s).
    
    Args:
        epoch: Current training epoch
        max_epochs: Total number of epochs
        num_values: Number of independent values to generate (default: 1)
    
    Returns:
        Single float if num_values=1, list of floats if num_values>1
    """
    progress = epoch / max_epochs
    
    def get_single_ratio():
        if progress < 0.3:
            return random.choice([0.0, 1.0])  # Extreme values first
        elif progress < 0.6:
            return random.choice([0.0, 0.25, 0.75, 1.0])  # Add some intermediate
        else:
            return random.uniform(0.0, 1.0)  # Full range
    
    if num_values == 1:
        return get_single_ratio()
    else:
        return [get_single_ratio() for _ in range(num_values)]

def train_simplified_dac_morpher(model, train_dataloader, val_dataloader,
                                 num_epochs=100, learning_rate=0.0001, start_epoch=0,
                                 device='cuda' if torch.cuda.is_available() else 'cpu',
                                 save_dir='checkpoints', accumulation_steps=1):
    """
    Simplified training using only unified morph_ratio parameter.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        pct_start=0.1
    )
    
    def morphing_loss(predictions, targets, seq_lengths=None):
        total_loss = 0
        ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        
        batch_size, num_codebooks, seq_len = targets.shape
        
        for i in range(len(predictions)):
            pred_logits = predictions[i]
            target_tokens = targets[:, i, :].long()
            
            if seq_lengths is not None:
                # Move seq_lengths to the same device as targets
                seq_lengths = seq_lengths.to(targets.device)
                mask = torch.arange(seq_len, device=targets.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
                valid_pred = pred_logits[mask]
                valid_targets = target_tokens[mask]
                
                if valid_pred.numel() > 0:
                    codebook_loss = ce_loss(valid_pred, valid_targets)
                    total_loss += codebook_loss
            else:
                pred_flat = pred_logits.view(-1, pred_logits.size(-1))
                target_flat = target_tokens.view(-1)
                codebook_loss = ce_loss(pred_flat, target_flat)
                total_loss += codebook_loss
        
        return total_loss / len(predictions)
    
    # Training metrics
    best_val_loss = float('inf')
    no_improve_epochs = 0
    early_stop_patience = 15
    
    print(f"Training with unified morph_ratio only")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                # Generate single morph ratio using curriculum learning
                batch_size = batch['morph_ratio'].shape[0]
                morph_ratio = get_curriculum_morph_ratio(epoch, num_epochs)
                morph_ratio_tensor = torch.full((batch_size, 1), morph_ratio, dtype=torch.float32).to(device)
                
                # Move batch to device
                source_codes = batch['loop_embedding'].to(device)
                source_style = batch['style_prob'].to(device)
                source_bpm = batch['bpm'].to(device)
                target_codes = batch['target_embedding'].to(device)
                target_style = batch['target_style_prob'].to(device)
                target_bpm = batch['target_bpm'].to(device)
                
                # Forward pass with unified morph_ratio
                outputs = model(
                    source_codes=source_codes,
                    source_style=source_style,
                    source_bpm=source_bpm,
                    target_codes=target_codes,
                    target_style=target_style,
                    target_bpm=target_bpm,
                    morph_ratio=morph_ratio_tensor,
                    source_seq_lengths=batch.get('seq_lengths'),
                    target_seq_lengths=batch.get('target_seq_lengths')
                )
                
                # Calculate loss
                loss = morphing_loss(
                    outputs, 
                    target_codes, 
                    batch.get('target_seq_lengths')
                )
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
                num_batches += 1
                
                if batch_idx % 20 == 0:
                    ...
                    # print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item() * accumulation_steps:.4f} | Morph Ratio: {morph_ratio:.3f}")
            
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_train_loss = train_loss / max(num_batches, 1)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation with different morph ratios
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            # Test multiple morph ratios during validation
            for morph_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                val_loss = 0.0
                val_batches = 0
                
                for batch in val_dataloader:
                    try:
                        batch_size = batch['morph_ratio'].shape[0]
                        morph_ratio_tensor = torch.full((batch_size, 1), morph_val, dtype=torch.float32).to(device)
                        
                        outputs = model(
                            source_codes=batch['loop_embedding'].to(device),
                            source_style=batch['style_prob'].to(device),
                            source_bpm=batch['bpm'].to(device),
                            target_codes=batch['target_embedding'].to(device),
                            target_style=batch['target_style_prob'].to(device),
                            target_bpm=batch['target_bpm'].to(device),
                            morph_ratio=morph_ratio_tensor,
                            source_seq_lengths=batch.get('seq_lengths'),
                            target_seq_lengths=batch.get('target_seq_lengths')
                        )
                        
                        loss = morphing_loss(
                            outputs,
                            batch['target_embedding'].to(device),
                            batch.get('target_seq_lengths')
                        )
                        
                        val_loss += loss.item()
                        val_batches += 1
                    
                    except Exception as e:
                        print(f"Error processing validation batch: {str(e)}")
                        continue
                
                if val_batches > 0:
                    val_losses.append(val_loss / val_batches)
                    writer.add_scalar(f'Loss/val_morph_{morph_val}', val_loss / val_batches, epoch)
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        writer.add_scalar('Loss/val_avg', avg_val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            
            checkpoint_path = os.path.join(save_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            
            print(f"Best model saved to {checkpoint_path}")
        else:
            no_improve_epochs += 1
        
        # Early stopping
        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
    
    writer.close()
    print("Training completed!")
    return model


def evaluate_unified_morphing(model, test_dataloader, device):
    """Evaluate unified morphing capabilities across different ratios"""
    model.eval()
    results = {}
    
    print("Evaluating unified morphing...")
    
    # Test different morph ratios
    test_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for ratio in test_ratios:
        losses = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                try:
                    batch_size = batch['morph_ratio'].shape[0]
                    morph_ratio_tensor = torch.full((batch_size, 1), ratio, dtype=torch.float32).to(device)
                    
                    outputs = model(
                        source_codes=batch['loop_embedding'].to(device),
                        source_style=batch['style_prob'].to(device),
                        source_bpm=batch['bpm'].to(device),
                        target_codes=batch['target_embedding'].to(device),
                        target_style=batch['target_style_prob'].to(device),
                        target_bpm=batch['target_bpm'].to(device),
                        morph_ratio=morph_ratio_tensor
                    )
                    
                    # Calculate loss
                    ce_loss = nn.CrossEntropyLoss()
                    batch_loss = 0
                    for i, output in enumerate(outputs):
                        logits = output.view(-1, model.codebook_size)
                        targets = batch['target_embedding'][:, i, :].reshape(-1).long().to(device)
                        batch_loss += ce_loss(logits, targets)
                    
                    losses.append(batch_loss.item() / len(outputs))
                
                except Exception as e:
                    continue
        
        if losses:
            avg_loss = sum(losses) / len(losses)
            results[f"morph_ratio_{ratio}"] = avg_loss
            print(f"Morph ratio {ratio:.1f}: Loss = {avg_loss:.4f}")
    
    return results

def main():
    """Main training function with improved architecture"""
    # Configuration
    embeddings_dir = "../output_preprocess/encoded"
    prob_dir = "../output_preprocess/style_probs"
    save_dir = "checkpoints_CMT"
    os.makedirs(save_dir, exist_ok=True)
    
    # Model parameters - optimized for morphing
    model_params = {
        'num_codebooks': 9,
        'codebook_size': 1024,
        'd_model': 64,  # Increased for better representation
        'nhead': 8,
        'num_encoder_layers': 6,  # More layers for better encoding
        'num_decoder_layers': 6,
        'dim_feedforward': 512,
        'dropout': 0.2,  # Reduced dropout for better morphing
        'style_dim': 400,
        'max_seq_len': 3879
    }
    
    # Training parameters
    train_params = {
        'batch_size': 4,  # Small batch for memory efficiency
        'learning_rate': 1e-4,  # Lower learning rate for stability
        'num_epochs': 300,
        'accumulation_steps': 8  # Effective batch size of 16
    }
    
    # Create improved dataset
    dataset = ImprovedDACLoopMorphingDataset(embeddings_dir, prob_dir, robust_loading=True)
    
    # Split into train/val sets
    train_size = int(0.85 * len(dataset))  # More training data
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True  # For consistent batch sizes
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_params['batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create improved model
    model = SimpleDACMorpher(**model_params).to(device)

    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} trainable parameters")
    
    # Check for existing checkpoints
    starting_epoch = 0
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))[-1]
        checkpoint_path = os.path.join(save_dir, latest_checkpoint)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        
        # Adjust epochs to ensure sufficient additional training
        train_params['num_epochs'] = max(train_params['num_epochs'], starting_epoch + 20)
    
    # Train the improved model
    trained_model = train_simplified_dac_morpher(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs=train_params['num_epochs'],
        learning_rate=train_params['learning_rate'],
        start_epoch=starting_epoch,
        device=device,
        save_dir=save_dir,
        accumulation_steps=train_params['accumulation_steps']
    )
    
    # Save final model
    final_path = os.path.join(save_dir, 'final_FILMmodel2.pth')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_params': model_params,
        'training_params': train_params
    }, final_path)
    print(f"Final improved model saved to {final_path}")
    
    # Evaluate morphing quality
    print("Evaluating morphing quality...")
    morphing_results = evaluate_unified_morphing(trained_model, val_dataloader, device)
    
    print("Morphing Quality Results:")
    for key, loss in morphing_results.items():
        print(f"{key}: Loss {loss:.4f}")
    
    # Save evaluation results
    eval_path = os.path.join(save_dir, 'morphing_evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(morphing_results, f, indent=2)
    print(f"Evaluation results saved to {eval_path}")


if __name__ == "__main__":
    main()
