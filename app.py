import logging
import os
import sqlite3
import uuid
from queue import Queue
from threading import Thread

from flask import Flask, request, jsonify, send_file
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity

from dbmodels import Task, db

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your_secret_key'

bcrypt = Bcrypt(app)
jwt = JWTManager(app)
db.init_app(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
task_queue = Queue()


def verify_password(provided_password, stored_hash):
    return bcrypt.check_password_hash(stored_hash, provided_password)


Thread(target=task_worker, daemon=True).start()


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


def get_db_connection():
    try:
        conn = sqlite3.connect('tasks.db')
        conn.row_factory = sqlite3.Row  # This allows row results to be treated like dictionaries
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise


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


@app.route('/task_status/<task_id>', methods=['GET'])
@jwt_required()
def task_status(task_id):
    current_user_id = get_jwt_identity()

    # Using SQLAlchemy to fetch the task
    task = Task.query.filter_by(id=task_id, user_id=current_user_id).first()

    if task:
        # Serialize the task data to JSON format
        task_data = {
            'id': task.id,
            'status': task.status,
            'download_url': task.download_url,
            'error': task.error,
            'prompt': task.prompt,
            'duration': task.duration,
            'model_name': task.model_name,
            'timestamp': task.timestamp.isoformat() if task.timestamp else None,
            'last_modified': task.last_modified.isoformat() if task.last_modified else None
        }
        return jsonify(task_data), 200
    else:
        return jsonify({"status": "unknown"}), 404


@app.route('/generate', methods=['POST'])
@jwt_required()
def enqueue_generate():
    current_user_id = get_jwt_identity()
    data = request.json

    task_id = str(uuid.uuid4())
    task = Task(
        id=task_id,
        user_id=current_user_id,
        prompt=data.get('prompt'),
        model_name=data.get('model', 'facebook/musicgen-small'),
        duration=float(data.get('duration', 5.0)),
        status='processing'  # Explicitly setting the status
    )

    db.session.add(task)
    db.session.commit()

    # Simulate task queue handling, actual queuing system would be more complex
    task_dict = {
        'task_id': task.id,
        'user_id': task.user_id,
        'type': 'generate',
        'prompt': task.prompt,
        'model_name': task.model_name,
        'duration': task.duration,
        'url_root': request.url_root
    }
    # This would actually push to a real task queue in production
    print(f"Task {task_id} queued for generation. Prepare to handle asynchronously.")

    return jsonify({"message": "Your music generation request is queued.", "task_id": task_id}), 202


@app.route('/generate_continuation', methods=['POST'])
@jwt_required()
def enqueue_generate_continuation():
    current_user_id = get_jwt_identity()  # Get the user ID from the JWT token

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_filename = os.path.join('temp_files', f"{uuid.uuid4().hex}_{file.filename}")
    if not os.path.exists('temp_files'):
        os.makedirs('temp_files')
    file.save(temp_filename)

    task_id = str(uuid.uuid4())
    task = {
        'task_id': task_id,
        'user_id': current_user_id,  # Associate task with the current user
        'type': 'generate_continuation',
        'file_path': temp_filename,
        'prompt': request.form.get('description', ''),
        'model_name': 'facebook/musicgen-melody',
        'duration': float(request.form.get('duration', '30')),
        'url_root': request.url_root
    }

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO tasks (id, user_id, status, prompt, duration, model_name) VALUES (?, ?, 'processing', ?, ?, ?)",
        (task_id, current_user_id, task['prompt'], task['duration'], task['model_name']))  # Include user_id
    conn.commit()
    conn.close()
    task_queue.put(task)
    return jsonify({"message": "Your music generation continuation request is queued.", "task_id": task_id}), 202


@app.route('/generate_followup', methods=['POST'])
@jwt_required()
def enqueue_generate_followup():
    current_user_id = get_jwt_identity()  # Get the user ID from the JWT token

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_filename = os.path.join('temp_files', f"{uuid.uuid4().hex}_{file.filename}")
    if not os.path.exists('temp_files'):
        os.makedirs('temp_files')
    file.save(temp_filename)

    task_id = str(uuid.uuid4())
    task = {
        'task_id': task_id,
        'user_id': current_user_id,  # Associate task with the current user
        'type': 'generate_followup',
        'file_path': temp_filename,
        'prompt': request.form.get('description', ''),
        'model_name': request.form.get('model', 'facebook/musicgen-melody'),
        'duration': float(request.form.get('duration', '5')),
        'url_root': request.url_root
    }

    conn = get_db_connection()
    conn.execute(
        "INSERT INTO tasks (id, user_id, status, prompt, duration, model_name) VALUES (?, ?, 'processing', ?, ?, ?)",
        (task_id, current_user_id, task['prompt'], task['duration'], task['model_name']))  # Include user_id
    conn.commit()
    conn.close()
    task_queue.put(task)
    return jsonify({"message": "Your music generation follow-up request is queued.", "task_id": task_id}), 202


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
