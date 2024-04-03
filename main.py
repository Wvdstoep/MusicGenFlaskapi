import logging
from audiocraft.models import musicgen
import torch
import torchaudio
from datetime import datetime  # Import the datetime module

# Ensure torchaudio is installed: pip install torchaudio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_audio(samples: torch.Tensor, sample_rate: int, base_filename: str = "output_audio"):
    """Saves audio samples as WAV files with a timestamp in the filename.

    Args:
        samples (torch.Tensor): A Tensor of decoded audio samples with shapes [B, C, T] or [C, T].
        sample_rate (int): The sample rate the audio should be saved with.
        base_filename (str): Base filename for the saved audio files, which will be appended with a timestamp and index.
    """
    assert samples.dim() == 2 or samples.dim() == 3, "Tensor must be 2 or 3 dimensions"

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)  # Add batch dimension if not present

    # Generate a timestamp string
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    for i, audio in enumerate(samples, start=1):
        # Append the timestamp and index to the base filename
        filename = f"{base_filename}_{timestamp}_{i}.wav"
        # In case of multi-channel audio, select first channel only for simplicity
        if audio.shape[0] > 1:
            audio = audio[0:1]
        torchaudio.save(filename, audio, sample_rate)
        logging.info(f"Audio saved as {filename}.")


# List of prompts
prompts = [
    'a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions',

]

# Load the model
model = musicgen.MusicGen.get_pretrained('facebook/musicgen-melody', device='cpu')
model.set_generation_params(duration=8)

# Generate music for each prompt with logging
for i, prompt in enumerate(prompts, start=1):
    logging.info(f"Starting generation for prompt {i}/{len(prompts)}: '{prompt}'")
    res = model.generate([prompt], progress=True)
    logging.info(f"Finished generation for prompt {i}/{len(prompts)}")
    # Use the custom save_audio function to save the generated audio
    save_audio(res, 32000, base_filename=f"generated_audio_prompt_{i}")
