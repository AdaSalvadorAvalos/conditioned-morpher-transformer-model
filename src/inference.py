"""
Inference Module
==================

This module provides utilities for **encoding, decoding, and transforming audio**
using a Discrete Audio Codec (DAC) model and a morphing architecture ('SimpleDACMorpher').
It integrates preprocessing functions and supports inference workflows for music/audio
analysis and transformation.

Features
--------
- Encode audio into DAC representation with optional compression.
- Decode audio back from encoded representation.
- Perform morphing and transformation using `SimpleDACMorpher`.
- Preprocess audio: cleaning filenames, predicting labels, detecting BPM.
- Command-line support for batch processing and custom configurations.

Global Parameters
-----------------
- device (torch.device): Default is set to CPU. Controls where the model is executed.

Dependencies
------------
- Python standard libraries: os, sys, argparse, subprocess, typing
- Third-party libraries: 
  - torch (PyTorch) for deep learning and model inference
  - numpy for numerical operations
- Local modules:
  - model.SimpleDACMorpher : Morphing model for DAC-encoded audio
  - preprocess.process_wav : Preprocessing for audio files
  - preprocess.clean_filename : Utility for standardized file naming
  - preprocess.predict_labels_MAEST : Label prediction for audio
  - preprocess.detect_bpm : Beat-per-minute detection

Usage
-----
As a library:
    >>> from inference import encode_audio, decode_audio
    >>> success, message, mapping = encode_audio("input.wav", output_path="encoded.dac")

From command line:
    $ python inference.py --input input.wav --output output.dac --model_size 44k

Notes
-----
- Ensure DAC is installed or specify its directory via `codec_dir`.
- Model size options include "8k", "24k", "44k" depending on available checkpoints.
- The script assumes audio files are in standard formats supported by the DAC encoder.
"""

import torch
import numpy as np
import os
import argparse
from typing import Dict, Any, Tuple, Optional, List
import sys
import subprocess
# Import model architecture from original file
# Add the current directory to the path to allow importing from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model architecture
# from transformer4_DAC_morphdis import SimpleDACMorpher
from model import SimpleDACMorpher
from preprocess import process_wav, clean_filename, predict_labels_MAEST,detect_bpm


device = torch.device('cpu')

def encode_audio(input_path, output_path=None, codec_dir=None, model_size="44k", compression_level=None):
    """
    Encode audio using the DAC (Discrete Audio Codec) encoder.
    
    Args:
        input_path (str): Path to the input audio file or directory
        output_path (str, optional): Path where the encoded file should be saved.
                                    If None, saves in the same directory with '_encoded' suffix.
        codec_dir (str, optional): Directory where the DAC module is installed.
                                  If None, assumes DAC is installed globally.
        model_size (str, optional): Size of the DAC model to use. Default is "44k".
                                   Available options typically include "8k", "24k", "44k", etc.
        compression_level (int, optional): Compression level to use. If None, uses DAC default.
    
    Returns:
        tuple: (success, message, mapping)
            - success (bool): Whether the encoding was successful
            - message (str): stdout if successful, stderr if failed
            - mapping (dict): Mapping from input files to encoded files (if input was directory)
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = "encoded"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build command
    cmd = []
    
    # Use python module directly if no codec_dir specified
    if codec_dir:
        cmd.extend([os.path.join(codec_dir, "python")])
    else:
        cmd.extend(["python"])
    
    cmd.extend(["-m", "dac", "encode","--device", "cpu", input_path, "--output", output_path])
    # NOTE: when using cpu add this:
    #  "--device", "cpu",
    
    # Add optional model size
    if model_size:
        cmd.extend(["--model_size", model_size])
    
    # Add optional compression level
    if compression_level is not None:
        cmd.extend(["--compression_level", str(compression_level)])
    
    # print(f"Running encode command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # print(f"Encoding successful: {result.stdout}")
        
        # Create a mapping between input and output files
        mapping = {}
        if os.path.isdir(input_path):
            # If input is a directory, create a mapping between processed and encoded files
            input_files = [f for f in os.listdir(input_path) if f.lower().endswith(".wav")]
            for input_file in input_files:
                input_full_path = os.path.join(input_path, input_file)
                # The encoded file will have .dac extension
                encoded_file = os.path.splitext(input_file)[0] + ".dac"
                encoded_full_path = os.path.join(output_path, encoded_file)
                if os.path.exists(encoded_full_path):
                    mapping[input_full_path] = encoded_full_path
        
        return True, result.stdout, mapping
    except subprocess.CalledProcessError as e:
        print(f"Encoding failed: {e.stderr}")
        return False, e.stderr, {}

def decode_audio(encoded_path, output_path=None, codec_dir=None, sample_rate=44100):
    """
    Decode audio using the DAC (Discrete Audio Codec) decoder.
    
    Args:
        encoded_path (str): Path to the encoded DAC file or directory
        output_path (str, optional): Path where the decoded audio should be saved.
                                    If None, saves in the same directory with '_decoded' suffix.
        codec_dir (str, optional): Directory where the DAC module is installed.
                                  If None, assumes DAC is installed globally.
        sample_rate (int, optional): Sample rate for the decoded audio. Default is 44100.
    
    Returns:
        tuple: (success, message, mapping)
            - success (bool): Whether the decoding was successful
            - message (str): stdout if successful, stderr if failed
            - mapping (dict): Mapping from encoded files to decoded files (if input was directory)
    """
    # Set default output path if not provided
    if output_path is None:
        base_dir = os.path.dirname(encoded_path)
        output_path = os.path.join(base_dir, "decoded")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build command
    cmd = []
    
    # Use python module directly if no codec_dir specified
    if codec_dir:
        cmd.extend([os.path.join(codec_dir, "python")])
    else:
        cmd.extend(["python"])
    
    cmd.extend(["-m", "dac", "decode", encoded_path, "--device", "cpu", "--output", output_path])
    
    # Add sample rate
    if sample_rate is not None:
        cmd.extend(["--sample_rate", str(sample_rate)])
    
    print(f"Running decode command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Decoding successful: {result.stdout}")
        
        # Create a mapping between encoded and decoded files
        mapping = {}
        if os.path.isdir(encoded_path):
            # If input is a directory, create a mapping between encoded and decoded files
            encoded_files = [f for f in os.listdir(encoded_path) if f.lower().endswith(".dac")]
            for encoded_file in encoded_files:
                encoded_full_path = os.path.join(encoded_path, encoded_file)
                # The decoded file will have .wav extension
                decoded_file = os.path.splitext(encoded_file)[0] + ".wav"
                decoded_full_path = os.path.join(output_path, decoded_file)
                if os.path.exists(decoded_full_path):
                    mapping[encoded_full_path] = decoded_full_path
        
        return True, result.stdout, mapping
    except subprocess.CalledProcessError as e:
        print(f"Decoding failed: {e.stderr}")
        return False, e.stderr, {}
    
def load_model(
    checkpoint_path: str, 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[SimpleDACMorpher, Dict[str, Any]]:
    """
    Load a trained DAC morpher model from a checkpoint
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        model: Loaded model
        model_params: Model parameters dictionary
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from checkpoint
    if 'model_params' in checkpoint:
        model_params = checkpoint['model_params']
    else:
        # Default parameters if not found in checkpoint
        model_params = {
            'num_codebooks': 9,
            'codebook_size': 1024,
            'd_model': 64,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 512,
            'dropout': 0.2,
            'style_dim': 400,
            'max_seq_len': 3879
        }
        print(f"Warning: No model parameters found in checkpoint. Using default parameters.")
    
    # Create model instance
    model = SimpleDACMorpher(**model_params)
    
    # Load state dictionary
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model, model_params


def load_dac_file(file_path: str) -> torch.Tensor:
    """
    Load a DAC file and convert to PyTorch tensor
    
    Args:
        file_path: Path to the DAC file
        
    Returns:
        DAC codes as tensor [1, num_codebooks, seq_len]
    """
    try:
        # Load DAC file
        dac_data = np.load(file_path, allow_pickle=True).item()
        
        # Extract codes
        if isinstance(dac_data, dict) and 'codes' in dac_data:
            codes = dac_data['codes']
            
            # Convert uint16 to int64 for PyTorch compatibility
            if codes.dtype == np.uint16:
                codes = codes.astype(np.int64)
            
            # Remove batch dimension if present
            if len(codes.shape) == 3 and codes.shape[0] == 1:
                codes = codes[0]  # Shape: [num_codebooks, seq_len]
            
            # Convert to torch tensor
            codes = torch.from_numpy(codes).clone()
            
            # Ensure shape is [1, num_codebooks, seq_len]
            if len(codes.shape) == 2:
                codes = codes.unsqueeze(0)
                
            return codes
        else:
            raise ValueError(f"Invalid DAC format in {file_path}, missing 'codes' key")
    except Exception as e:
        raise RuntimeError(f"Error loading DAC file {file_path}: {str(e)}")


def load_style_prob(file_path: str) -> torch.Tensor:
    """
    Load a style probability file
    
    Args:
        file_path: Path to the style probability file
        
    Returns:
        Style probability tensor [1, style_dim]
    """
    try:
        # Load style probabilities
        style_prob = torch.load(file_path, map_location='cpu')
        
        # Convert to tensor if not already
        if not isinstance(style_prob, torch.Tensor):
            style_prob = torch.tensor(style_prob, dtype=torch.float32)
        
        # Ensure shape is [1, style_dim]
        if len(style_prob.shape) == 1:
            style_prob = style_prob.unsqueeze(0)
        
        return style_prob
    except Exception as e:
        raise RuntimeError(f"Error loading style probability file {file_path}: {str(e)}")


def load_bpm(file_path: str) -> torch.Tensor:
    """
    Load a BPM file
    
    Args:
        file_path: Path to the BPM file
        
    Returns:
        BPM tensor [1, 1]
    """
    try:
        # Load BPM
        bpm = torch.load(file_path, map_location='cpu')
        
        # Convert to tensor if not already
        if not isinstance(bpm, torch.Tensor):
            bpm = torch.tensor(bpm, dtype=torch.float32)
        
        # Ensure shape is [1, 1]
        if len(bpm.shape) == 0:  # scalar
            bpm = bpm.reshape(1, 1)
        elif len(bpm.shape) == 1:  # vector
            bpm = bpm.reshape(-1, 1)
        
        return bpm
    except Exception as e:
        raise RuntimeError(f"Error loading BPM file {file_path}: {str(e)}")


def save_dac_output(output_codes: torch.Tensor, output_path: str, source_metadata: dict = None) -> None:
    """
    Save generated DAC codes to a file with metadata
    
    Args:
        output_codes: Generated DAC codes [1, num_codebooks, seq_len]
        output_path: Path to save the output DAC file
        source_metadata: Optional metadata to include (from source file)
    """
    try:
        # Convert to numpy if needed
        if isinstance(output_codes, torch.Tensor):
            output_codes = output_codes.cpu().numpy()
        
        # Create output dictionary with codes
        output_dict = {'codes': output_codes}
        
        # Add metadata
        if source_metadata:
            # Use source metadata with any necessary modifications
            output_dict['metadata'] = source_metadata.copy()
        else:
            # Create default metadata if none provided
            output_dict['metadata'] = {
                'input_db': [-20.0],  # Default value
                'original_length': output_codes.shape[1] * 416,  # Estimate based on codes length
                'sample_rate': 44100,  # Standard sample rate
                'chunk_length': 416,   # Standard chunk length for DAC
                'channels': 1,         # Mono audio
                'padding': False,      # No padding
                'dac_version': '1.0.0' # DAC version
            }
        
        # If output_path is a directory, create a filename within it
        if os.path.isdir(output_path):
            output_file = os.path.join(output_path, "morphed_output.dac")
        else:
            # Ensure the filename ends with .dac
            if not output_path.lower().endswith('.dac'):
                output_path += '.dac'
            output_file = output_path
        
        # Save as numpy file but with .dac extension
        np.save(output_file, output_dict)
        
        # If numpy added .npy extension, rename the file
        if os.path.exists(output_file + '.npy'):
            os.rename(output_file + '.npy', output_file)
            
        print(f"Saved morphed DAC to {output_file} with metadata")
    except Exception as e:
        print(f"Error saving DAC output: {str(e)}")



def generate_advanced_morph_gradient(
    model: SimpleDACMorpher,
    source_dac: torch.Tensor,
    source_style: torch.Tensor,
    source_bpm: torch.Tensor,
    target_dac: torch.Tensor,
    target_style: torch.Tensor,
    target_bpm: torch.Tensor,
    num_steps: int = 10,
    morph_type: str = "overall",  # "overall", "loop", "style", "bpm"
    custom_style: Optional[torch.Tensor] = None,
    custom_bpm: Optional[torch.Tensor] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[float, torch.Tensor]:
    """
    Generate a series of morphed loops with gradually changing morph ratios for specific aspects
    
    Args:
        model: Trained DAC loop morpher model
        source_dac: Source DAC codes [1, num_codebooks, seq_len]
        source_style: Source style probability [1, style_dim]
        source_bpm: Source BPM [1, 1]
        target_dac: Target DAC codes [1, num_codebooks, seq_len]
        target_style: Target style probability [1, style_dim]
        target_bpm: Target BPM [1, 1]
        num_steps: Number of morph steps to generate
        morph_type: Type of morphing gradient ("overall")
        custom_style: Optional custom style to morph towards
        custom_bpm: Optional custom BPM to morph towards
        device: Device to run on
    
    Returns:
        Dictionary mapping morph ratios to morphed DAC codes
    """
    # Create a series of morph ratios
    morph_ratios = np.linspace(0.1, 0.9, num_steps)
    
    # Generate morphs for each ratio
    morphs = {}
    for ratio in morph_ratios:
        print(f"Generating {morph_type} morph with ratio {ratio:.2f}")
        
        # Set up ratio parameters based on morph type
        overall_ratio = None
        
        if morph_type == "overall":
            overall_ratio = ratio
   
        morphed_dac = generate_advanced_morph(
            model, source_dac, source_style, source_bpm,
            target_dac, target_style, target_bpm,
            morph_ratio=overall_ratio,
            custom_style=custom_style,
            custom_bpm=custom_bpm,
            device=device
        )
        morphs[float(ratio)] = morphed_dac
    
    return morphs


def load_source_metadata(file_path: str) -> dict:
    """
    Load metadata from a DAC file
    
    Args:
        file_path: Path to the DAC file
        
    Returns:
        Metadata dictionary or default metadata if not found
    """
    try:
        # Load DAC file
        dac_data = np.load(file_path, allow_pickle=True).item()
        
        # Extract metadata if available
        if isinstance(dac_data, dict) and 'metadata' in dac_data:
            return dac_data['metadata']
        else:
            print(f"Warning: No metadata found in {file_path}, using default metadata")
            return {
                'input_db': [-20.0],
                'sample_rate': 44100,
                'chunk_length': 416,
                'channels': 1,
                'padding': False,
                'dac_version': '1.0.0'
            }
    except Exception as e:
        print(f"Error loading metadata from {file_path}: {str(e)}")
        return {
            'input_db': [-20.0],
            'sample_rate': 44100,
            'chunk_length': 416,
            'channels': 1,
            'padding': False,
            'dac_version': '1.0.0'
        }

def collect_all_labels(file_path):
    """
    Collects all unique genre labels from a file
    to ensure consistent one-hot encoding dimensions
    """
    _, _, all_labels = predict_labels_MAEST(file_path)
    return all_labels


def create_genre_style_prob(all_labels, genre_names, base_probability=0.3):
    """
    Create a style probability distribution from a list of genre names.
    Supports parent genres (e.g., "Rock") which will distribute probability
    across all matching subgenres (e.g., "Rock---AOR", "Rock---Acid Rock").
    
    Args:
        all_labels: List of all possible genre labels
        genre_names: List of genre names (can include parent genres)
        base_probability: Base probability value to assign (default 0.3, typical model output)
    Returns:
        Style probability tensor (sigmoid-compatible, not normalized)
    """
    if all_labels is None:
        raise ValueError("Genre labels not loaded")
        
    # Create empty distribution (all zeros for sigmoid)
    style_prob = torch.zeros(len(all_labels), dtype=torch.float32).to(device)
    
    # Process each requested genre
    for genre in genre_names:
        # Check if it's an exact match first
        if genre in all_labels:
            idx = all_labels.index(genre)
            style_prob[idx] = base_probability
            print(f"Found exact match for '{genre}' -> probability {base_probability}")
        else:
            # Check if it's a parent genre (e.g., "Rock")
            parent_matches = [label for label in all_labels if label.startswith(f"{genre}---")]
            
            if parent_matches:
                # Assign the same base probability to ALL subgenres
                # This maintains the sigmoid scale while covering the genre family
                for match in parent_matches:
                    idx = all_labels.index(match)
                    style_prob[idx] = base_probability
                
                print(f"Assigned probability {base_probability} to {len(parent_matches)} subgenres of '{genre}':")
                for match in parent_matches[:5]:  # Show first 5 matches
                    print(f" - {match}")
                if len(parent_matches) > 5:
                    print(f" - ... and {len(parent_matches) - 5} more")
            else:
                print(f"Warning: Genre '{genre}' not found as exact match or parent genre")
    
    # NO NORMALIZATION - sigmoid doesn't require it!
    # Each probability is independent, so keep the original scale
    print(f"Final distribution: {(style_prob > 0).sum().item()} genres with probability {base_probability}")
    print(f"Total 'probability mass': {style_prob.sum().item():.2f} (this is fine for sigmoid)")
            
    return style_prob

def print_all_parent_genres(all_labels):
    """
    Print all unique parent genre names from the all_labels list.
    """
    parent_genres = set()
    for label in all_labels:
        if '---' in label:
            parent = label.split('---')[0]
            parent_genres.add(parent)
    print(f"Found {len(parent_genres)} parent genres:")
    for genre in sorted(parent_genres):
        print(f"- {genre}")

def create_custom_bpm_tensor(bpm_value: float) -> torch.Tensor:
    """
    Create a custom BPM tensor from a BPM value
    
    Args:
        bpm_value: BPM value to use
        
    Returns:
        BPM tensor [1, 1]
    """
    return torch.tensor([[bpm_value]], dtype=torch.float32)


def generate_advanced_morph(
    model: SimpleDACMorpher,
    source_dac: torch.Tensor,
    source_style: torch.Tensor,
    source_bpm: torch.Tensor,
    target_dac: torch.Tensor,
    target_style: torch.Tensor,
    target_bpm: torch.Tensor,
    morph_ratio: Optional[float] = None,
    custom_style: Optional[torch.Tensor] = None,
    custom_bpm: Optional[torch.Tensor] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    Generate an advanced morphed loop with granular control over different aspects
    
    Args:
        model: Trained DAC loop morpher model
        source_dac: Source DAC codes [1, num_codebooks, seq_len]
        source_style: Source style probability [1, style_dim]
        source_bpm: Source BPM [1, 1]
        target_dac: Target DAC codes [1, num_codebooks, seq_len]
        target_style: Target style probability [1, style_dim]
        target_bpm: Target BPM [1, 1]
        morph_ratio: Overall morph ratio (used if specific ratios not provided)
        custom_style: Custom style embedding to use instead of target style
        custom_bpm: Custom BPM value to use instead of target BPM
        use_custom_as_target: Whether to use custom values as morph targets
        device: Device to run on
    
    Returns:
        Morphed DAC codes [1, num_codebooks, seq_len]
    """
    model.eval()
    
    # Ensure all inputs are on the proper device
    source_dac = source_dac.to(device)
    source_style = source_style.to(device)
    source_bpm = source_bpm.to(device)
    target_dac = target_dac.to(device)
    target_style = target_style.to(device)
    target_bpm = target_bpm.to(device)
    
    # Convert ratios to tensors and move to device
    morph_ratio_tensor = None
    if morph_ratio is not None:
        morph_ratio_tensor = torch.tensor([[morph_ratio]], dtype=torch.float32).to(device)
    
    # Handle custom conditioning
    custom_style_tensor = None
    if custom_style is not None:
        custom_style_tensor = custom_style.unsqueeze(0).to(device)
    
    custom_bpm_tensor = None
    if custom_bpm is not None:
        custom_bpm_tensor = custom_bpm.to(device)
    
    with torch.no_grad():
        # Use the model's forward method with all the advanced parameters
        output_logits = model.forward(
            source_codes=source_dac,
            source_style=source_style,
            source_bpm=source_bpm,
            target_codes=target_dac,
            target_style=target_style,
            target_bpm=target_bpm,
            morph_ratio=morph_ratio_tensor,
            custom_style=custom_style_tensor,
            custom_bpm=custom_bpm_tensor
        )
        
        # Convert logits to predictions for each codebook
        morphed_codes = []
        for i in range(model.num_codebooks):
            logits = output_logits[i]  # [1, seq_len, codebook_size]
            preds = torch.argmax(logits, dim=2)  # [1, seq_len]
            morphed_codes.append(preds)
        
        # Stack predictions to get full DAC codes
        morphed_dac = torch.stack(morphed_codes, dim=1)  # [1, num_codebooks, seq_len]
        
    return morphed_dac

def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description="DAC Loop Morpher Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--source_file", type=str, required=True, help="Path to source wav file")
    parser.add_argument("--target_file", type=str, required=True, help="Path to target wav file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output DAC file")
    parser.add_argument("--morph_ratio", type=float, default=None, help="Morph ratio (0.0-1.0)")
    parser.add_argument("--target_style", type=str, nargs='+', help="Optional: target genre labels")
    parser.add_argument("--target_style_file", type=str, help="Path to target style wav file")
    parser.add_argument("--target_bpm", type=float, help="Optional: target BPM value")
    parser.add_argument("--gradient_mode", action="store_true", help="Generate gradient of morphs")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of gradient steps (if gradient_mode)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()
    # Assign output_dir from args
    output_dir = args.output_dir

    # Create the base output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define subdirectory for output DAC
    outputDAC_dir = os.path.join(output_dir, "outputDAC")
    os.makedirs(outputDAC_dir, exist_ok=True)
    
    # Set device
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        file_mappings = {}
        # Load model
        print(f"Loading model from {args.checkpoint}")
        model, model_params = load_model(args.checkpoint, device)
        print(f"Model loaded successfully with parameters: {model_params}")
        
        # Process source file
        print(f"Processing source file: {args.source_file}")
        source_base_name = clean_filename(os.path.basename(args.source_file))
        source_processed_path = os.path.join(args.output_dir, f"{source_base_name}_processed.wav")
        
        # Process the source WAV file
        source_processed_path, source_original_bpm = process_wav(
            args.source_file, 
            source_processed_path, 
            preserve_bpm=True
        )

        file_mappings[args.source_file] = {"processed": source_processed_path}
        
        # Process target file
        print(f"Processing target file: {args.target_file}")
        target_base_name = clean_filename(os.path.basename(args.target_file))
        target_processed_path = os.path.join(args.output_dir, f"{target_base_name}_processed.wav")
        
        # Process the target WAV file
        target_processed_path, target_original_bpm = process_wav(
            args.target_file, 
            target_processed_path, 
            preserve_bpm=True
        )

        file_mappings[args.target_file] = {"processed": target_processed_path}
        
        # Generate Mel spectrograms and encode both source and target audio
        _, _, encode_mappings = encode_audio(args.output_dir, output_path=args.output_dir)

        # Match encoded files to original files
        source_encoded_path = None
        target_encoded_path = None
        for processed_path, encoded_path in encode_mappings.items():
            # Find the original file that maps to this processed file
            for orig_path, paths in file_mappings.items():
                if paths["processed"] == processed_path:
                    file_mappings[orig_path]["encoded"] = encoded_path
                    if orig_path == args.source_file:
                        source_encoded_path = encoded_path
                    elif orig_path == args.target_file:
                        target_encoded_path = encoded_path
                    break
        
        # Load source metadata
        source_metadata = load_source_metadata(source_encoded_path)
        print(f"Loaded source metadata: {source_metadata}")
        
        # Save source BPM as a feature
        source_bpm_feature = torch.tensor([source_original_bpm], dtype=torch.float32)
        source_bpm_path = os.path.join(args.output_dir, f"{source_base_name}_bpm.pt")
        torch.save(source_bpm_feature.cpu(), source_bpm_path)
        
        # Save target BPM as a feature
        target_bpm_feature = torch.tensor([target_original_bpm], dtype=torch.float32)
        target_bpm_path = os.path.join(args.output_dir, f"{target_base_name}_bpm.pt")
        torch.save(target_bpm_feature.cpu(), target_bpm_path)
        
        # Predict source labels and get full probability distribution
        top_n = 5
        source_labels, source_probabilities, source_all_maest_labels, source_full_probs = predict_labels_MAEST(
            source_processed_path, top_n=top_n, return_full_probs=True
        )
        
        # Save the source probability distribution
        source_probs_path = os.path.join(args.output_dir, f"{source_base_name}_style_probs.pt")
        torch.save(source_full_probs.cpu(), source_probs_path)

        # Predict target labels and get full probability distribution
        target_labels, target_probabilities, target_all_maest_labels, target_full_probs = predict_labels_MAEST(
            target_processed_path, top_n=top_n, return_full_probs=True
        )
        
        # Save the target probability distribution
        target_probs_path = os.path.join(args.output_dir, f"{target_base_name}_style_probs.pt")
        torch.save(target_full_probs.cpu(), target_probs_path)

        # Load source data
        print(f"Loading source DAC from {source_encoded_path}")
        source_dac = load_dac_file(source_encoded_path)
        print(f"Source DAC shape: {source_dac.shape}")
        
        print(f"Loading source style from {source_probs_path}")
        source_style = load_style_prob(source_probs_path)
        print(f"Source style shape: {source_style.shape}")
        
        print(f"Loading source BPM from {source_bpm_path}")
        source_bpm = load_bpm(source_bpm_path)
        print(f"Source BPM: {source_bpm.item():.1f}")
        
        # Load target data
        print(f"Loading target DAC from {target_encoded_path}")
        target_dac = load_dac_file(target_encoded_path)
        print(f"Target DAC shape: {target_dac.shape}")
        
        print(f"Loading target style from {target_probs_path}")
        target_style = load_style_prob(target_probs_path)
        print(f"Target style shape: {target_style.shape}")
        
        print(f"Loading target BPM from {target_bpm_path}")
        target_bpm = load_bpm(target_bpm_path)
        print(f"Target BPM: {target_bpm.item():.1f}")
        
        # Print all parent genres for debugging
        # all_labels = collect_all_labels(source_processed_path)
        # print_all_parent_genres(all_labels)
        # print(f"All available genres: {all_labels}")
        # Handle custom style if provided
        custom_style = None
        if args.target_style:
            print(f"Creating custom style for: {args.target_style}")
            all_labels = collect_all_labels(source_processed_path)  # Get all available labels
            custom_style = create_genre_style_prob(all_labels, args.target_style)  # Remove the extra brackets

        elif args.target_style_file:
            print(f"Processing target file: {args.target_style_file}")
            target_style_file_name = clean_filename(os.path.basename(args.target_style_file))
            target_style_processed_path = os.path.join(args.output_dir, f"{target_style_file_name}_processed.wav")
            
            # Process the target WAV file
            target_style_processed_path, target_original_bpm = process_wav(
                args.target_style_file, 
                target_style_processed_path, 
                preserve_bpm=True
            )

            # Predict source labels and get full probability distribution
            top_n = 5
            source_labels, source_probabilities, source_all_maest_labels, target_style_full_probs = predict_labels_MAEST(
                target_style_processed_path, top_n=top_n, return_full_probs=True
            )
            
            # Save the source probability distribution
            target_style_probs_path = os.path.join(args.output_dir, f"{target_style_file_name}_style_probs.pt")
            torch.save(target_style_full_probs.cpu(), target_style_probs_path)
            print(f"Loading target style from {target_style_probs_path}")
            target_style = load_style_prob(target_style_probs_path)

        # Handle custom BPM if provided
        custom_bpm = None
        if args.target_bpm:
            print(f"Using custom BPM: {args.target_bpm}")
            custom_bpm = create_custom_bpm_tensor(args.target_bpm)
        
        # Generate morph
        if args.gradient_mode:
            print(f"Generating gradient of morphs with {args.num_steps} steps")
            # Create descriptive name for the morphed file
            morph_base_name = f"{source_base_name}_to_{target_base_name}"
            
            morphs = generate_advanced_morph_gradient(
                model, source_dac, source_style, source_bpm,
                target_dac, target_style, target_bpm, 
                num_steps=args.num_steps,
                morph_type="overall",
                custom_style=custom_style,
                custom_bpm=custom_bpm,
                device=device
            )
            
            # Save each morph
            for ratio, morphed_dac in morphs.items():
                # Create filename with ratio
                output_file = os.path.join(outputDAC_dir, f"{morph_base_name}_morph_{ratio:.2f}.dac")
                # Update original_length in metadata based on morphed output
                updated_metadata = source_metadata.copy()
                if 'original_length' in updated_metadata:
                    # Calculate new length based on the shape of the morphed DAC codes
                    seq_len = morphed_dac.shape[2] if len(morphed_dac.shape) == 3 else morphed_dac.shape[1]
                    updated_metadata['original_length'] = seq_len * updated_metadata.get('chunk_length', 416)
                
                save_dac_output(morphed_dac, output_file, updated_metadata)
                
                # Decode each gradient step
                decoded_dir = os.path.join(output_dir, "decoded")
                os.makedirs(decoded_dir, exist_ok=True)
                decode_success, decode_message, _ = decode_audio(output_file, decoded_dir)
                if decode_success:
                    print(f"Successfully decoded morph {ratio:.2f} to {decoded_dir}")
                else:
                    print(f"Failed to decode morph {ratio:.2f}: {decode_message}")
        else:
            print(f"Generating single morph with ratios - overall: {args.morph_ratio}")
            
            # Create descriptive name for the morphed file based on which ratios are provided
            morph_parts = []
            if args.morph_ratio is not None:
                morph_parts.append(f"overall_{args.morph_ratio:.2f}")
            # if args.loop_morph_ratio is not None:
            #     morph_parts.append(f"loop_{args.loop_morph_ratio:.2f}")
            # if args.style_morph_ratio is not None:
            #     morph_parts.append(f"style_{args.style_morph_ratio:.2f}")
            # if args.bpm_morph_ratio is not None:
            #     morph_parts.append(f"bpm_{args.bpm_morph_ratio:.2f}")
            
            # If no ratios provided, use default naming
            if not morph_parts:
                morph_suffix = "default"
            else:
                morph_suffix = "_".join(morph_parts)
            
            morph_file_name = f"{source_base_name}_to_{target_base_name}_morph_{morph_suffix}.dac"
            output_file = os.path.join(outputDAC_dir, morph_file_name)
            
            morphed_dac = generate_advanced_morph(
                model, source_dac, source_style, source_bpm,
                target_dac, target_style, target_bpm,
                morph_ratio=args.morph_ratio,
                custom_style=custom_style,
                custom_bpm=custom_bpm,
                device=device
            )
            
            # Update original_length in metadata based on morphed output
            updated_metadata = source_metadata.copy()
            if 'original_length' in updated_metadata:
                # Calculate new length based on the shape of the morphed DAC codes
                seq_len = morphed_dac.shape[2] if len(morphed_dac.shape) == 3 else morphed_dac.shape[1]
                updated_metadata['original_length'] = seq_len * updated_metadata.get('chunk_length', 416)
            
            # Save result with proper file name and metadata
            save_dac_output(morphed_dac, output_file, updated_metadata)
            
            # Decode the DAC file
            decoded_dir = os.path.join(output_dir, "decoded")
            os.makedirs(decoded_dir, exist_ok=True)
            decode_success, decode_message, _ = decode_audio(output_file, decoded_dir)
            if decode_success:
                print(f"Successfully decoded morphed audio to {decoded_dir}")
            else:
                print(f"Failed to decode morphed audio: {decode_message}")
            
        print("Inference complete!")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())