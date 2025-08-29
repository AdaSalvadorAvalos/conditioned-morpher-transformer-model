"""
 Preprocessing Module
=============================

This module provides functions to preprocess audio files for machine learning
and music information retrieval tasks. It integrates TorchAudio, Librosa,
Essentia, and FFmpeg for robust handling of audio data, including resampling,
BPM detection, metadata extraction, and format conversion.

Features
--------
- Load and parse metadata (JSON-based).
- Extract file IDs from filenames.
- Detect BPM using:
  - Metadata annotations (if available).
  - Essentia's RhythmExtractor2013 (fallback).
- Normalize audio sample rate and duration.
- Run FFmpeg commands with error handling.
- Utilities for dataset preprocessing (shuffle, conversion, etc.).

Global Parameters
-----------------
- TARGET_SAMPLE_RATE (int): Desired sample rate for all audio (default: 44100 Hz).
- TARGET_DURATION (int): Target clip length in seconds (default: 5s).
- TARGET_BPM (int): Default tempo for normalization when no BPM is found (default: 120 BPM).

Dependencies
------------
- Python standard library: os, re, json, random, subprocess, shutil, argparse
- Third-party:
  - torch
  - torchaudio
  - librosa
  - numpy
  - tqdm
  - essentia (with standard algorithms)
  - maest (custom module: provides `get_maest`)

Usage
-----
Run as a script to preprocess a dataset:

    python preprocess.py --input_dir ./raw_audio --output_dir ./processed_audio --metadata ./metadata.json

Or import specific functions in your own pipeline:

    from preprocess import detect_bpm, load_metadata, run_ffmpeg_command

"""

import torch
import torchaudio
import re
import librosa
import numpy as np
import os
import json
import random  # Add this import for shuffling
from maest import get_maest
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm
import argparse
import shutil
import essentia
from essentia.standard import MonoLoader, RhythmExtractor2013, KeyExtractor
import subprocess


TARGET_SAMPLE_RATE = 44100
TARGET_DURATION = 5
TARGET_BPM = 120 

def run_ffmpeg_command(cmd):
    """Run FFmpeg command with error handling."""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    
def load_metadata(metadata_path):
    """Load BPM information from metadata.json"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def get_file_id_from_filename(filename):
    """Extract file ID from filename like FSL10K/audio/wav/425998_9497060.wav.wav"""
    # Extract just the base filename without path
    base_name = os.path.basename(filename)
    # Extract the ID part (before the first underscore)
    file_id = base_name.split('_')[0]
    return file_id

def get_bpm_from_metadata(file_path, metadata):
    """Get the BPM for a file from metadata"""
    file_id = get_file_id_from_filename(file_path)
    
    # Look up the file in metadata
    if file_id in metadata:
        if "annotations" in metadata[file_id] and "bpm" in metadata[file_id]["annotations"]:
            return float(metadata[file_id]["annotations"]["bpm"])
    
    # Return None if BPM not found, we'll detect it algorithmically later
    return None

def detect_bpm(file_path):
    """Detect BPM using Essentia's RhythmExtractor2013."""
    try:
        audio = MonoLoader(filename=file_path)()
        rhythm_extractor = RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm_extractor(audio)
        
        # Apply sanity check
        if 30 <= bpm <= 300:
            return float(bpm)
        else:
            print(f"Detected unreasonable BPM ({bpm}), defaulting to 120.0")
            return 120.0
    except Exception as e:
        print(f"Error detecting BPM: {e}")
        return 120.0
    

def clean_filename(filename):
    base_name = re.sub(r'(\.\w+)+$', '', filename)
    base_name = base_name.replace("_processed", "")
    base_name = base_name.replace("_", "")
    return base_name


def tempo_convert_ffmpeg(input_file, target_bpm, original_bpm, target_sample_rate):
    """
    Convert tempo using FFmpeg with both atempo and asetrate methods,
    then determine which result is closer to the target BPM.
    """
    # Calculate speed ratio
    speed_ratio = target_bpm / original_bpm
    # print(f"Speed ratio for conversion: {speed_ratio} (Original: {original_bpm} BPM → Target: {target_bpm} BPM)")
    
    # File paths for the two methods
    output_atempo = f"temp_atempo_{os.path.basename(input_file)}"
    output_asetrate = f"temp_asetrate_{os.path.basename(input_file)}"
    
    # Method 1: FFmpeg atempo
    atempo_chain = []
    remaining = speed_ratio
    
    # Handle speed ratio limits in FFmpeg
    if speed_ratio > 2.0:
        # Chain multiple atempo filters (each with value <= 2.0)
        while remaining > 1.0:
            factor = min(2.0, remaining)
            atempo_chain.append(f"atempo={factor}")
            remaining /= factor
    elif speed_ratio < 0.5:
        # Chain multiple atempo filters (each with value >= 0.5)
        while remaining < 1.0:
            factor = max(0.5, remaining)
            atempo_chain.append(f"atempo={factor}")
            remaining /= factor
    else:
        atempo_chain.append(f"atempo={speed_ratio}")
    
    # Run atempo method
    cmd_atempo = [
        "ffmpeg", "-y", "-i", input_file,
        "-filter:a", ",".join(atempo_chain),
        "-ar", str(target_sample_rate),
        output_atempo
    ]
    
    # print("Running FFmpeg atempo method...")
    success_atempo, _ = run_ffmpeg_command(cmd_atempo)
    
    # Method 2: FFmpeg asetrate
    # Calculate the asetrate value
    original_sr = torchaudio.info(input_file).sample_rate
    asetrate = int(original_sr * speed_ratio)
    
    # Run asetrate method
    cmd_asetrate = [
        "ffmpeg", "-y", "-i", input_file,
        "-filter:a", f"asetrate={asetrate},aresample={target_sample_rate}",
        output_asetrate
    ]
    
    # print("Running FFmpeg asetrate method...")
    success_asetrate, _ = run_ffmpeg_command(cmd_asetrate)
    
    # Determine which method produced the better result
    best_method = None
    best_file = None
    best_error = float('inf')
    
    if success_atempo and os.path.exists(output_atempo):
        bpm_atempo = detect_bpm(output_atempo)
        error_atempo = abs(bpm_atempo - target_bpm)
       #  print(f"atempo method result: {bpm_atempo} BPM (error: {error_atempo:.2f})")
        
        if error_atempo < best_error:
            best_method = "atempo"
            best_file = output_atempo
            best_error = error_atempo
    
    if success_asetrate and os.path.exists(output_asetrate):
        bpm_asetrate = detect_bpm(output_asetrate)
        error_asetrate = abs(bpm_asetrate - target_bpm)
        # print(f"asetrate method result: {bpm_asetrate} BPM (error: {error_asetrate:.2f})")
        
        if error_asetrate < best_error:
            best_method = "asetrate"
            best_file = output_asetrate
            best_error = error_asetrate
    
    if best_method:
        # print(f"Selected {best_method} method (closer to target BPM)")
        return best_file
    else:
        # print("Both tempo conversion methods failed!")
        return input_file  # Return original if both methods fail


def process_wav(file_path, processed_save_path, metadata=None, target_duration=TARGET_DURATION, 
                target_sample_rate=TARGET_SAMPLE_RATE, target_bpm=TARGET_BPM, preserve_bpm=False):
    """
    Process a WAV file to have consistent sample rate and duration.
    If preserve_bpm is False, will convert to target_bpm using the best FFmpeg method.
    """
    if not os.path.exists(file_path):
        print(f"Error: Input file {file_path} does not exist")
        return None, None
    
    # Step 1: Detect original BPM
    original_bpm = get_file_bpm(file_path, metadata)
    
    # print(f"Original file: {file_path}")
    # print(f"Original BPM: {original_bpm}")
    
    # Create a temp copy to work with
    temp_file = f"temp_{os.path.basename(file_path)}"
    
    # Step 2: Adjust tempo if needed
    if not preserve_bpm and abs(original_bpm - target_bpm) > 1.0:
        # print(f"Converting tempo from {original_bpm} to {target_bpm} BPM...")
        tempo_adjusted_file = tempo_convert_ffmpeg(
            file_path, target_bpm, original_bpm, target_sample_rate
        )
        working_file = tempo_adjusted_file
        final_bpm = target_bpm
    else:
        # If preserving BPM or BPMs are already close, just use original
        working_file = file_path
        final_bpm = original_bpm
        
        # Just convert sample rate if needed
        if torchaudio.info(file_path).sample_rate != target_sample_rate:
            wav, sr = torchaudio.load(file_path)
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)(wav)
            torchaudio.save(temp_file, wav, target_sample_rate)
            working_file = temp_file
    
    # Step 3: Adjust duration
    wav, sr = torchaudio.load(working_file)
    target_samples = target_duration * target_sample_rate
    
    # Handle mono/stereo
    if wav.shape[0] > 1:
        # Average stereo to mono
        wav = wav.mean(dim=0, keepdim=True)
    
    # Adjust length through padding or trimming
    if wav.shape[1] < target_samples:
        # Loop the audio if needed
        repeats_needed = int(np.ceil(target_samples / wav.shape[1]))
        repeated_audio = torch.tile(wav, (1, repeats_needed))
        wav = repeated_audio[:, :target_samples]
    else:
        # Trim to desired length
        wav = wav[:, :target_samples]
    
    # Step 4: Save the final processed file
    torchaudio.save(processed_save_path, wav, target_sample_rate)
    
    # Clean up temp files
    temp_files = [
        f"temp_atempo_{os.path.basename(file_path)}",
        f"temp_asetrate_{os.path.basename(file_path)}",
        temp_file
    ]
    for tf in temp_files:
        if os.path.exists(tf) and tf != processed_save_path:
            try:
                os.remove(tf)
            except:
                pass
    
    # print(f"Processed file saved to: {processed_save_path}")
    # print(f"Final BPM: {final_bpm}")
    
    return processed_save_path, final_bpm


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
    
    cmd.extend(["-m", "dac", "encode", input_path,"--device", "cpu", "--output", output_path])
    # NOTE: when using cpu add this:
    #"--device", "cpu",
    
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

def predict_labels_MAEST(file_path, top_n=5, return_full_probs=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #NOTE: arch="discogs-maest-5s-pw-129e"  USED: arch="discogs-maest-30s-pw-129e-519l"
    model = get_maest(arch="discogs-maest-5s-pw-129e").to(device)
    model.eval()

    # Load and preprocess audio
    wav, sr = torchaudio.load(file_path)
    if sr != TARGET_SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.to(device)  # Move to GPU
    # Log-mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=1024,
        hop_length=320,
        n_mels=80
    ).to(device)
    logmel = mel_transform(wav)
    logmel = torch.log1p(logmel)  # shape: [1, 128, time]

    # Add batch and channel dims: [1, 1, 128, T]
    logmel = logmel.unsqueeze(0)

    # Resize to model input shape: [1, 1, 96, 1875]
    # size=(96, 1875) FOR arch="discogs-maest-30s-pw-129e-519l" (96*312) FOR arch="discogs-maest-5s-pw-129e"
    logmel_resized = F.interpolate(logmel, size=(96, 312), mode='bilinear', align_corners=False).to(device)

    # Predict
    with torch.no_grad():
        activations, labels = model.predict_labels(logmel_resized)

    # Convert activations to a PyTorch tensor if they're in numpy format
    if isinstance(activations, np.ndarray):
        activations = torch.tensor(activations).to(device)

    # Convert activations to probabilities (e.g., applying softmax)
    # probabilities = F.softmax(activations, dim=-1).squeeze(0).cpu()
    probabilities = activations.squeeze(0).cpu()  # No need for additional activation function
    
    # Get the top N labels based on highest probability for backward compatibility
    top_n_indices = torch.topk(probabilities, top_n).indices
    top_n_probabilities = probabilities[top_n_indices].tolist()
    top_n_labels = [labels[i] for i in top_n_indices]
    
    if return_full_probs:
        # Return both the top N results and the full probability distribution
        return top_n_labels, top_n_probabilities, labels, probabilities
    else:
        # Return just the top N results for backward compatibility
        return top_n_labels, top_n_probabilities, labels

def create_one_hot_encoding(top_labels, all_labels):
    """
    Create a one-hot encoding for the top labels
    Args:
        top_labels: List of top predicted labels
        all_labels: Complete list of all possible labels
    Returns:
        Dictionary mapping each label to 1 if it's in top_labels, 0 otherwise
    """
    one_hot = {label: 1 if label in top_labels else 0 for label in all_labels}
    return one_hot

def collect_all_labels(input_dir, num_files=1):
    """
    Collects all unique genre labels from a sample of files
    to ensure consistent one-hot encoding dimensions
    """
    all_labels_set = set()
    
    # Process a sample of files to get labels
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.WAV'))][:num_files]
    
    # Use the first file to get the complete label set
    if wav_files:
        _, _, all_labels = predict_labels_MAEST(os.path.join(input_dir, wav_files[0]))
        return all_labels
    else:
        raise ValueError("No WAV files found in the specified directory")

def get_file_bpm(file_path, metadata=None):
    """
    Get the BPM for a file without processing it
    Returns:
        float: BPM value or None if couldn't be determined
    """
    # Try to get BPM from metadata first
    if metadata:
        bpm = get_bpm_from_metadata(file_path, metadata)
        if bpm is not None:
            return bpm
    
    # If not in metadata, try to detect it
    try:
        # wav, sr = torchaudio.load(file_path)
        # audio_np = wav[0].numpy() if wav.shape[0] > 1 else wav[0].numpy()
        tempo = detect_bpm(file_path)
        # tempo, _ = librosa.beat.beat_track(y=audio_np, sr=sr)
        
        if isinstance(tempo, np.ndarray) and tempo.size > 0:
            return float(tempo.item())
        else:
            return float(tempo)
    except:
        return None

def filter_files_by_bpm(input_dir, metadata=None, min_bpm=120, max_bpm=130, max_files=None):
    """
    Filter files by BPM range
    Args:
        input_dir: Directory containing WAV files
        metadata: Metadata dictionary
        min_bpm: Minimum BPM value to include
        max_bpm: Maximum BPM value to include
        max_files: Maximum number of files to return (None for all)
    Returns:
        List of filenames that fall within the BPM range
    """
    all_wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.WAV'))]
    
    filtered_files = []
    print(f"Filtering files with BPM between {min_bpm} and {max_bpm}...")
    
    for wav_file in tqdm(all_wav_files, desc="Checking BPM"):
        file_path = os.path.join(input_dir, wav_file)
        bpm = get_file_bpm(file_path, metadata)
        
        if bpm is not None and min_bpm <= bpm <= max_bpm:
            filtered_files.append(wav_file)
            print(f"  Found file with BPM {bpm}: {wav_file}")
            
            if max_files is not None and len(filtered_files) >= max_files:
                break
    
    return filtered_files

def process_dataset(input_dir, output_dir, all_labels, file_list=None, num_files=None, 
                  metadata=None, preserve_bpm=True, shuffle=False):
    """
    Process a set of files and save their embeddings and labels with one-hot encoding 
    and full probability distributions.
    
    Args:
        input_dir: Directory containing WAV files
        output_dir: Directory to save processed files
        all_labels: List of all possible labels for one-hot encoding
        file_list: List of specific files to process (optional)
        num_files: Number of files to process (optional)
        metadata: Metadata dictionary (optional)
        preserve_bpm: Whether to preserve original BPM
        shuffle: Whether to shuffle the files before processing
    """
    
    # Create necessary directories
    processed_dir = os.path.join(output_dir, "processed")
    embed_dir = os.path.join(output_dir, "encoded")
    labels_dir = os.path.join(output_dir, "labels")
    onehot_dir = os.path.join(output_dir, "onehot")
    probs_dir = os.path.join(output_dir, "style_probs")
    
    # Create all directories
    for directory in [processed_dir, embed_dir, labels_dir, onehot_dir, probs_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # If file_list is provided, use it; otherwise get all WAV files from the input directory
    if file_list:
        wav_files = file_list
    else:
        wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.WAV'))]
    
    # Shuffle files if requested
    if shuffle:
        print(f"Shuffling {len(wav_files)} files randomly...")
        random.shuffle(wav_files)
        
        # Save the shuffled file list for reproducibility
        shuffled_list_path = os.path.join(output_dir, "shuffled_files_list.json")
        with open(shuffled_list_path, 'w') as f:
            json.dump(wav_files, f, indent=2)
        print(f"Saved shuffled file list to {shuffled_list_path}")
    
    # Limit to the specified number of files if needed
    if num_files is not None:
        wav_files = wav_files[:num_files]
    
    # Process each original file
    original_results = []
    file_mappings = {}
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        file_path = os.path.join(input_dir, wav_file)
        
        try:
            # Clean the filename for saving
            base_name = clean_filename(os.path.basename(wav_file))
            processed_path = os.path.join(processed_dir, f"{base_name}_processed.wav")
            
            # Process the WAV file
            processed_path, original_bpm = process_wav(
                file_path, 
                processed_path, 
                metadata=metadata,
                preserve_bpm=preserve_bpm
            )

            file_mappings[file_path] = {"processed": processed_path}
            
            # Generate encoded audio
            _,_ , encode_mappings = encode_audio(processed_dir, output_path=embed_dir)

            for processed_path, encoded_path in encode_mappings.items():
            # Find the original file that maps to this processed file
                for orig_path, paths in file_mappings.items():
                    if paths["processed"] == processed_path:
                        file_mappings[orig_path]["encoded"] = encoded_path
                        break
            
            # Save BPM as a feature
            bpm_feature = torch.tensor([original_bpm], dtype=torch.float32)
            bpm_path = os.path.join(embed_dir, f"{base_name}_bpm.pt")
            torch.save(bpm_feature.cpu(), bpm_path)
            
            # Predict labels and get full probability distribution
            top_n = 5
            labels, probabilities, all_maest_labels, full_probs = predict_labels_MAEST(
                processed_path, top_n=top_n, return_full_probs=True
            )
            
            # Create one-hot encoding
            one_hot_encoding = create_one_hot_encoding(labels, all_labels)
            
            # Convert one-hot encoding to an array
            one_hot_array = np.array([one_hot_encoding[label] for label in all_labels], dtype=np.float32)
            
            # Save one-hot encoding as numpy array
            onehot_path = os.path.join(onehot_dir, f"{base_name}_onehot.npy")
            np.save(onehot_path, one_hot_array)
            
            # Save the full probability distribution
            probs_path = os.path.join(probs_dir, f"{base_name}_style_probs.pt")
            torch.save(full_probs.cpu(), probs_path)
            
            # For human readability, also save top probabilities and their corresponding labels
            top_probs_dict = {label: float(prob) for label, prob in zip(labels, probabilities)}
            
            # Also save as JSON for human readability
            onehot_json_path = os.path.join(onehot_dir, f"{base_name}_onehot.json")
            with open(onehot_json_path, 'w') as f:
                json.dump(one_hot_encoding, f, indent=2)
            
            # Save labels and probabilities as JSON
            label_info = {
                "file": wav_file,
                "original_bpm": float(original_bpm),
                "labels": labels,
                "top_probabilities": probabilities,
                "one_hot_encoding": one_hot_encoding,
                "top_probs_dict": top_probs_dict
            }
            
            label_path = os.path.join(labels_dir, f"{base_name}_labels.json")
            with open(label_path, 'w') as f:
                json.dump(label_info, f, indent=2)
            
            # Store results for the summary
            original_results.append({
                "file": wav_file,
                "processed": processed_path,
                "encoded_path": encoded_path,
                "bpm": float(original_bpm),
                "bmp_feature_path": bpm_path,
                "labels": labels,
                "probabilities": probabilities,
                "one_hot_path": onehot_path,
                "style_probs_path": probs_path
            })
                
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
    
    # Save a summary of all processed files
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(original_results, f, indent=2)
    
    # Also save BPM statistics
    bpm_values = [result["bpm"] for result in original_results]
    bpm_stats = {
        "min_bpm": min(bpm_values) if bpm_values else None,
        "max_bpm": max(bpm_values) if bpm_values else None,
        "mean_bpm": sum(bpm_values) / len(bpm_values) if bpm_values else None,
        "median_bpm": sorted(bpm_values)[len(bpm_values)//2] if bpm_values else None,
        "bpm_count": {bpm: bpm_values.count(bpm) for bpm in set(bpm_values)} if bpm_values else {}
    }
    bpm_stats_path = os.path.join(output_dir, "bpm_stats.json")
    with open(bpm_stats_path, 'w') as f:
        json.dump(bpm_stats, f, indent=2)
    
    return original_results

if __name__ == "__main__":
    # At the beginning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Process audio files with DAC encoding and one-hot encoded labels")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing WAV files")
    parser.add_argument("--output_dir", type=str, default="DAC_output", help="Output directory for processed files")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata.json file")
    parser.add_argument("--num_files", type=int, default=None, help="Number of files to process (None for all)")
    parser.add_argument("--preserve_bpm", action="store_true", help="Preserve original BPM instead of normalizing to reference BPM")
    parser.add_argument("--min_bpm", type=float, default=120, help="Minimum BPM value to include")
    parser.add_argument("--max_bpm", type=float, default=130, help="Maximum BPM value to include")
    parser.add_argument("--filter_by_bpm", action="store_true", help="Filter files by BPM range")
    parser.add_argument("--shuffle", action="store_true", help="Randomly shuffle files before processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    
    args = parser.parse_args() 
    # Set random seed for reproducibility
    if args.shuffle:
        random.seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    print(f"BPM preservation is {'enabled' if args.preserve_bpm else 'disabled'}")
    print(f"BPM filtering is {'enabled' if args.filter_by_bpm else 'disabled'}")
    print(f"Random shuffling is {'enabled' if args.shuffle else 'disabled'}")
    
    # Load metadata
    print(f"Loading metadata from {args.metadata}...")
    metadata = load_metadata(args.metadata)
    print(f"Loaded metadata for {len(metadata)} files")
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filter files by BPM range if requested
    files_to_process = None
    
    if args.filter_by_bpm:
        print(f"Filtering files by BPM range {args.min_bpm}-{args.max_bpm}...")
        files_to_process = filter_files_by_bpm(
            args.input_dir, 
            metadata=metadata, 
            min_bpm=args.min_bpm, 
            max_bpm=args.max_bpm,
            max_files=args.num_files
        )
        print(f"Found {len(files_to_process)} files in BPM range")
        
        # Save the filtered file list
        files_list_path = os.path.join(args.output_dir, f"files_bpm_{args.min_bpm}_{args.max_bpm}.json")
        with open(files_list_path, 'w') as f:
            json.dump(files_to_process, f, indent=2)
    
    # First, collect all possible labels for consistent one-hot encoding
    print("Collecting all possible genre labels...")
    all_labels = collect_all_labels(args.input_dir, 1)
    print(f"Found {len(all_labels)} unique genre labels for one-hot encoding")
    
    # Save the complete list of labels
    labels_master_path = os.path.join(args.output_dir, "all_labels_master.json")
    with open(labels_master_path, 'w') as f:
        json.dump(all_labels, f, indent=2)
    
    # Process all files
    print("Processing files...")
    original_results, mixed_results = process_dataset(
        args.input_dir, 
        args.output_dir, 
        all_labels, 
        file_list=files_to_process,
        num_files=args.num_files,
        metadata=metadata,
        preserve_bpm=args.preserve_bpm,
        shuffle=args.shuffle
    )

    print(f"Completed processing {len(original_results)} original files")
    
    print(f"All processing complete. Results saved to {args.output_dir}")