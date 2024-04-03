from flask import Flask, request, send_file
import logging
from audiocraft.models import musicgen
import torch
import torchaudio
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_audio(samples: torch.Tensor, sample_rate: int, filename: str = "output_audio.wav"):
    """Saves audio samples as a WAV file."""
    assert samples.dim() == 2 or samples.dim() == 3, "Tensor must be 2 or 3 dimensions"
    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)  # Add batch dimension if not present
    # Assuming single-channel audio for simplicity, save the first one
    torchaudio.save(filename, samples[0], sample_rate)
    logging.info(f"Audio saved as {filename}.")


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    model_name = data.get('model', 'facebook/musicgen-small')  # Default to musicgen-small if not provided
    duration = float(data.get('duration', 5.0))  # Default to 30 seconds if not provided

    logging.info(f"Received prompt: '{prompt}', Model: {model_name}, Duration: {duration}")

    # Dynamically load the model based on the request
    model = musicgen.MusicGen.get_pretrained(model_name, device='cpu')
    model.set_generation_params(duration=duration)

    # Generate music for the prompt
    res = model.generate([prompt], progress=True)

    # Save the generated audio
    filename = "generated_audio.wav"
    save_audio(res, 32000, filename=filename)

    # Return the audio file
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
