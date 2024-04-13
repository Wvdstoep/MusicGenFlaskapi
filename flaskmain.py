from flask import Flask, request, jsonify, send_file
import logging
from audiocraft.models import musicgen, MusicGen
import torch
import torchaudio
import os
import uuid
from queue import Queue
from threading import Thread
import sqlite3

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
task_queue = Queue()


def get_db_connection():
    try:
        conn = sqlite3.connect('tasks.db')
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise


def save_audio(samples: torch.Tensor, sample_rate: int, filename: str):
    # Detach samples from computation graph and move to CPU
    samples = samples.detach().cpu()

    # Check if the tensor is 3D (e.g., [Batch, Channels, Samples])
    # and reduce it to 2D (e.g., [Channels, Samples]) if necessary
    if samples.dim() == 3 and samples.size(0) == 1:
        # Assuming batch size is 1, squeeze the batch dimension
        samples = samples.squeeze(0)

    # Calculate duration
    duration = calculate_duration(samples, sample_rate)

    # Define the directory based on the duration rounded to nearest second
    duration_folder = f"duration_{int(round(duration))}"

    # Ensure the directory exists
    os.makedirs(duration_folder, exist_ok=True)

    # Construct the full file path
    filepath = os.path.join(duration_folder, filename)

    # Save the audio file using torchaudio
    torchaudio.save(filepath, samples, sample_rate)
    logging.info(f"Audio saved as {filepath}.")

    return filepath, duration


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    try:
        logging.info(f"Attempting to download file: {filename}")
        base_path = 'C:/Users/carla/PycharmProjects/pythonProject'

        # Search for the file in all duration folders
        for root, dirs, files in os.walk(base_path):
            logging.info(f"Searching in directory: {root}")
            for dir_name in dirs:
                if dir_name.startswith('duration_'):
                    logging.info(f"Found duration folder: {dir_name}")
                    full_path = os.path.join(base_path, dir_name, filename)
                    if os.path.exists(full_path):
                        # Get database connection and update task status
                        task_id = extract_task_id_from_filename(filename)
                        if task_id:
                            conn = get_db_connection()
                            conn.execute("UPDATE tasks SET status = 'downloaded' WHERE id = ?", (task_id,))
                            conn.commit()
                            conn.close()
                            logging.info(f"Status for task {task_id} updated to downloaded in the database.")
                        else:
                            logging.error("Task ID could not be extracted from filename.")

                        # Send the file
                        return send_file(full_path, as_attachment=True)
                    else:
                        logging.info(f"File not found in folder: {dir_name}")

        # If the file is not found in any duration folder
        logging.error(f"File not found: {filename}")
        return jsonify({"error": "File not found."}), 404

    except Exception as e:
        logging.error(f"Error during the download process for {filename}: {e}")
        return jsonify({"error": "Internal server error."}), 500


def calculate_duration(samples, sample_rate):
    # Assuming `samples` is a tensor where the first dimension is the time dimension
    num_samples = samples.shape[-1]  # Last dimension in a 2D tensor (channel, samples)
    duration = num_samples / sample_rate
    return duration


def generate_music(task):
    task_id = task['task_id']
    try:
        model_name = task['model_name']
        duration = task['duration']
        if task['type'] == 'generate':
            model = musicgen.MusicGen.get_pretrained(model_name, device='cpu')
            model.set_generation_params(duration=duration)
            prompt = [task['prompt']]
            res = model.generate(prompt, progress=True)
            # Use task_id for the filename
            unique_filename = f"{task_id}_generated_audio.wav"
            save_audio(res, 32000, filename=unique_filename)
            logging.info("Music generated successfully.")

        elif task['type'] in ['generate_continuation', 'generate_followup']:
            temp_filename = task['file_path']
            model = MusicGen.get_pretrained(model_name, device='cpu')
            model.set_generation_params(duration=duration)
            melody, sr = torchaudio.load(temp_filename)
            if task['type'] == 'generate_continuation':
                res = model.generate_with_chroma([task['descriptions']], melody[None].expand(1, -1, -1), sr)
            else:
                res = model.generate_continuation(melody.unsqueeze(0), sr, descriptions=[task['descriptions']])[0]
            # Use task_id for the filename, differentiating type in the name
            unique_filename = f"{task_id}_{task['type']}.wav"
            save_audio(res.cpu(), sr, filename=unique_filename)
            os.remove(temp_filename)

        download_url = f"{task['url_root']}download/{unique_filename}"
        conn = get_db_connection()
        conn.execute("UPDATE tasks SET status = 'completed', download_url = ? WHERE id = ?", (download_url, task_id))
        conn.commit()
        conn.close()
        logging.info(f"Task completed: {task['type']}, available at {download_url}")

    except Exception as e:
        logging.error(f"Error generating music for task {task_id}: {e}")
        conn = get_db_connection()
        conn.execute("UPDATE tasks SET status = 'failed', error = ? WHERE id = ?", (str(e), task_id))
        conn.commit()
        conn.close()


def task_worker():
    while True:
        task = task_queue.get()
        try:
            generate_music(task)
        except Exception as e:
            logging.error(f"Unhandled exception in task worker: {e}")
        finally:
            task_queue.task_done()


Thread(target=task_worker, daemon=True).start()

tasks_db = {}


@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    conn = get_db_connection()
    task_row = conn.execute("SELECT id, status, download_url, error FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()

    if task_row:
        # Convert the sqlite3.Row object to a dictionary
        task = {key: task_row[key] for key in task_row.keys()}
        return jsonify(task), 200
    else:
        return jsonify({"status": "unknown"}), 404


@app.route('/generate', methods=['POST'])
def enqueue_generate():
    data = request.json
    logging.info(f"Received data for generation: {data}")
    task_id = str(uuid.uuid4())
    task = {
        'task_id': task_id,
        'type': 'generate',
        'prompt': data.get('prompt'),
        'model_name': data.get('model', 'facebook/musicgen-small'),
        'duration': float(data.get('duration', 5.0)),
        'url_root': request.url_root
    }
    conn = get_db_connection()
    # Updated to insert prompt, duration, and model_name
    conn.execute("INSERT INTO tasks (id, status, prompt, duration, model_name) VALUES (?, 'processing', ?, ?, ?)",
                 (task_id, task['prompt'], task['duration'], task['model_name']))
    conn.commit()
    conn.close()
    task_queue.put(task)
    return jsonify({"message": "Your music generation request is queued.", "task_id": task_id}), 202


@app.route('/generate_continuation', methods=['POST'])
def enqueue_generate_continuation():
    task_id = str(uuid.uuid4())

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    temp_filename = os.path.join('temp_files', f"{uuid.uuid4().hex}_{file.filename}")
    if not os.path.exists('temp_files'):
        os.makedirs('temp_files')
    file.save(temp_filename)
    prompt = request.form.get('description', '')  # Adapt this if your form data structure is different

    task = {
        'task_id': task_id,
        'type': 'generate_continuation',
        'file_path': temp_filename,
        'prompt': prompt,  # Using descriptions as prompt here
        'model_name': 'facebook/musicgen-melody',
        'duration': float(request.form.get('duration', '30')),
        'url_root': request.url_root
    }
    conn = get_db_connection()
    # Updated to insert prompt, duration, and model_name
    conn.execute("INSERT INTO tasks (id, status, prompt, duration, model_name) VALUES (?, 'processing', ?, ?, ?)",
                 (task_id, task['prompt'], task['duration'], task['model_name']))
    conn.commit()
    conn.close()
    task_queue.put(task)
    return jsonify({"message": "Your music generation request is queued.", "task_id": task_id}), 202


@app.route('/generate_followup', methods=['POST'])
def enqueue_generate_followup():
    task_id = str(uuid.uuid4())
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    temp_filename = os.path.join('temp_files', f"{uuid.uuid4().hex}_{file.filename}")
    if not os.path.exists('temp_files'):
        os.makedirs('temp_files')
    file.save(temp_filename)
    prompt = request.form.get('description', '')  # Adapt this if your form data structure is different

    task = {
        'task_id': task_id,
        'type': 'generate_followup',
        'file_path': temp_filename,
        'prompt': prompt,  # Using 'description' as 'prompt' here
        'model_name': request.form.get('model', 'facebook/musicgen-melody'),
        'duration': float(request.form.get('duration', '5')),
        'url_root': request.url_root
    }
    conn = get_db_connection()
    # Updated to insert prompt, duration, and model_name
    conn.execute("INSERT INTO tasks (id, status, prompt, duration, model_name) VALUES (?, 'processing', ?, ?, ?)",
                 (task_id, task['prompt'], task['duration'], task['model_name']))
    conn.commit()
    conn.close()
    task_queue.put(task)
    return jsonify({"message": "Your music generation request is queued.", "task_id": task_id}), 202


def extract_task_id_from_filename(filename):
    """
    Extracts the task_id from the filename.

    Assumes filename format is "{task_id}_generated_audio.wav".

    Args:
        filename (str): The filename from which to extract the task_id.

    Returns:
        str: The extracted task_id, or None if the extraction fails.
    """
    # Split the filename on the underscore and take the first part as the task_id
    parts = filename.split('_')
    if parts:
        task_id = parts[0]  # Assuming the task_id is the first part before the first underscore.
        return task_id
    else:
        logging.error("Filename format does not match expected pattern for extracting task_id.")
        return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
