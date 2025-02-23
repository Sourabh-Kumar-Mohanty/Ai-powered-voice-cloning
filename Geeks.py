import torchaudio
import librosa
import numpy as np
import torch
import soundfile as sf
from speechbrain.pretrained import EncoderClassifier
from elevenlabs import generate, play, set_api_key
from TTS.api import TTS

# Set API key for ElevenLabs (if using)
ELEVENLABS_API_KEY = "your-elevenlabs-api-key"
set_api_key(ELEVENLABS_API_KEY)

# Load Speaker Identification Model (SpeechBrain)
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            savedir="tmp")

def extract_speaker_embedding(audio_path):
    """Extract speaker embedding from an audio file"""
    signal, sr = librosa.load(audio_path, sr=16000)
    signal = torch.tensor(signal).unsqueeze(0)
    embedding = classifier.encode_batch(signal)
    return embedding.squeeze().detach().numpy()

def generate_voice_11labs(text, voice_id):
    """Generate AI voice using ElevenLabs"""
    audio = generate(text=text, voice=voice_id, model="eleven_multilingual_v2")
    play(audio)  # Play the generated voice
    return audio

def generate_voice_coqui(text, speaker_wav):
    """Generate AI voice using Coqui TTS"""
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    output_wav = "generated_voice.wav"
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=output_wav)
    return output_wav

# ---- Example Usage ---- #
# Path to recorded voice sample
recorded_voice = "your_voice_sample.wav"  # Change this to your audio file
lyrics = "Hello, I am Sourabh, how are you?"

# Extract speaker voice embedding
speaker_embedding = extract_speaker_embedding(recorded_voice)
print("Speaker Embedding Extracted!")

# Generate AI voice (Choose ElevenLabs or Coqui)
use_elevenlabs = True  # Change to False to use Coqui

if use_elevenlabs:
    ai_voice = generate_voice_11labs(lyrics, "Rachel")  # Change "Rachel" to a desired ElevenLabs voice ID
else:
    ai_voice = generate_voice_coqui(lyrics, recorded_voice)

print("AI Voice Generation Complete!")
