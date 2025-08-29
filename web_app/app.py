import os
import tempfile
import subprocess
import glob
from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains and routes
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add static file serving route
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files from the static directory"""
    return send_from_directory('static', filename)

@app.route('/morph', methods=['POST'])
def morph():
    try:
        # Validate required files
        if 'source' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'Both source and target files are required'}), 400
        
        source_file = request.files['source']
        target_file = request.files['target']
        
        if source_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Get form parameters with defaults
        checkpoint_path = request.form.get('checkpointPath', '')
        if not checkpoint_path:
            return jsonify({'error': 'Checkpoint path is required'}), 400

        # Get morph parameters - these should be passed as strings from form data
        morph_ratio = request.form.get('morphRatio')
        target_style = request.form.get('TargetStyle')
        target_bpm = request.form.get('TargetBPM')
        gradient_mode = request.form.get('GradientMode')
        num_steps = request.form.get('NumSteps')


        # Save uploaded files
        src_path = os.path.join(UPLOAD_FOLDER, secure_filename(source_file.filename))
        tgt_path = os.path.join(UPLOAD_FOLDER, secure_filename(target_file.filename))
        out_dir = os.path.join(UPLOAD_FOLDER, 'morphed_output')
        shutil.rmtree(out_dir, ignore_errors=True)

        source_file.save(src_path)
        target_file.save(tgt_path)

        # Build command arguments (filter out None values)
        cmd_args = [
            'python', 'inference.py',
            '--source_file', src_path,
            '--target_file', tgt_path,
            '--checkpoint', checkpoint_path,
            '--output_dir', out_dir,
            '--cpu'
        ]
        
        # Add optional parameters only if they're provided and not empty
        if morph_ratio is not None and morph_ratio.strip():
            cmd_args.extend(['--morph_ratio', morph_ratio])
        if target_style is not None and target_style.strip():
            cmd_args.extend(['--target_style', target_style])
        if target_bpm is not None and target_bpm.strip():
            cmd_args.extend(['--target_bpm', target_bpm])
        if gradient_mode is not None:
            cmd_args.append('--gradient_mode')
        if num_steps is not None and num_steps.strip():
            cmd_args.extend(['--num_steps', num_steps])

        print(f"Running command: {' '.join(cmd_args)}")  # Debug logging
        
        # Run the inference script
        result = subprocess.run(cmd_args, capture_output=True, text=True, check=True)
        
        # Find the generated audio file in the output directory
        # First check the main output directory
        audio_files = []

        decoded_dir = os.path.join(out_dir, "decoded")
        if os.path.exists(decoded_dir):
            for ext in ['*.wav', '*.mp3', '*.flac']:
                audio_files.extend(glob.glob(os.path.join(decoded_dir, ext)))
        
        if not audio_files:
            # List all files in output directory for debugging
            all_files = []
            for root, dirs, files in os.walk(out_dir):
                for file in files:
                    all_files.append(os.path.join(root, file))
            
            return jsonify({
                'error': 'No audio output file was generated',
                'debug_info': f'Files found in {out_dir}: {all_files}'
            }), 500
        
        # If multiple files, try to find the main output or use the first one
        output_file = audio_files[0]
        if len(audio_files) > 1:
            # Look for common output naming patterns
            for file in audio_files:
                filename = os.path.basename(file).lower()
                if any(pattern in filename for pattern in ['output', 'morph', 'result']):
                    output_file = file
                    break
        
        print(f"Sending output file: {output_file}")  # Debug logging
        
        # Generate appropriate filename for download
        original_name = os.path.basename(output_file)
        download_name = f'morphed_{morph_ratio}_{original_name}'
        
        return send_file(output_file, as_attachment=True, download_name=download_name)

    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return jsonify({
            'error': 'Inference failed', 
            'details': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }), 500
    except Exception as e:
        print(f"General error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup uploaded files
        try:
            if 'src_path' in locals() and os.path.exists(src_path):
                os.remove(src_path)
            if 'tgt_path' in locals() and os.path.exists(tgt_path):
                os.remove(tgt_path)
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'DAC Morpher API is running'})

@app.route('/')
def serve_html():
    """Serve the main HTML file"""
    return send_from_directory('.', 'maindark.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)