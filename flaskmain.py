from flask import Flask, request, jsonify, send_file
import logging
from audiocraft.models import musicgen
import torch
import torchaudio
import os
import uuid

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_audio(samples: torch.Tensor, sample_rate: int, filename: str = "output_audio.wav"):
    assert samples.dim() == 2 or samples.dim() == 3, "Tensor must be 2 or 3 dimensions"
    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)
    torchaudio.save(filename, samples[0], sample_rate)
    logging.info(f"Audio saved as {filename}.")


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    model_name = data.get('model', 'facebook/musicgen-small')
    duration = float(data.get('duration', 5.0))

    logging.info(f"Received prompt: '{prompt}', Model: {model_name}, Duration: {duration}")

    model = musicgen.MusicGen.get_pretrained(model_name, device='cpu')
    model.set_generation_params(duration=duration)

    res = model.generate([prompt], progress=True)

    unique_filename = f"{uuid.uuid4().hex}_generated_audio.wav"
    save_audio(res, 32000, filename=unique_filename)

    download_url = f"{request.url_root}download/{unique_filename}"
    return jsonify({"message": "Music generated successfully.", "download_url": download_url})


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
