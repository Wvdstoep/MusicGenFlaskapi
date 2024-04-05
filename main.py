import logging
from audiocraft.models import musicgen
import torch
import torchaudio
from datetime import datetime

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
    'Provide a build-up that elevates the anticipation before a euphoric chorus in a classic composition.',
    'Create an infectious and energetic synth melody for a club banger.',
    'Generate a bridge that transitions smoothly from a soft verse to a loud chorus with multiple instruments.',
    'Provide a build-up that elevates the anticipation before a euphoric chorus for a trance track.',
    'Provide a dynamic and intense build-up for a fiddle solo composition from soft to hard.',

]

# Load the model
model = musicgen.MusicGen.get_pretrained('facebook/musicgen-melody', device='cpu')
model.set_generation_params(duration=30)

# Generate music for each prompt with logging
for i, prompt in enumerate(prompts, start=1):
    logging.info(f"Starting generation for prompt {i}/{len(prompts)}: '{prompt}'")
    res = model.generate([prompt], progress=True)
    logging.info(f"Finished generation for prompt {i}/{len(prompts)}")
    # Use the custom save_audio function to save the generated audio
    save_audio(res, 32000, base_filename=f"generated_audio_prompt_{i}")
