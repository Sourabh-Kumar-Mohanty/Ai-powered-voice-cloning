from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import librosa
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from elevenlabs import generate, play, set_api_key
from TTS.api import TTS
import soundfile as sf

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
GENERATED_FOLDER = "generated"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# Set your ElevenLabs API key here
ELEVENLABS_API_KEY = "your-elevenlabs-api-key"
set_api_key(ELEVENLABS_API_KEY)

# Load Speaker Identification Model (SpeechBrain)
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp"
)

# Global variable to hold the speaker embedding or voice file path
speaker_audio_path = None

def extract_speaker_embedding(audio_path):
    """Extract speaker embedding from an audio file"""
    signal, sr = librosa.load(audio_path, sr=16000)
    signal = torch.tensor(signal).unsqueeze(0)
    embedding = classifier.encode_batch(signal)
    return embedding.squeeze().detach().numpy()

def noise_reduction(audio_path):
    """
    Dummy noise reduction function.
    Replace this with your actual noise reduction code.
    """
    # For now, we simply return the same file.
    return audio_path

def generate_voice_11labs(text, voice_id="Rachel"):
    """Generate AI voice using ElevenLabs"""
    audio = generate(text=text, voice=voice_id, model="eleven_multilingual_v2")
    # Save the audio bytes to a file
    out_path = os.path.join(GENERATED_FOLDER, "generated_voice_11labs.wav")
    with open(out_path, "wb") as f:
        f.write(audio)
    return out_path

def generate_voice_coqui(text, speaker_wav):
    """Generate AI voice using Coqui TTS"""
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    output_wav = os.path.join(GENERATED_FOLDER, "generated_voice_coqui.wav")
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=output_wav)
    return output_wav

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    global speaker_audio_path
    if "audio" not in request.files:
        return jsonify(success=False, message="No audio file provided."), 400
    file = request.files["audio"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Apply noise reduction (if any)
    processed_path = noise_reduction(file_path)
    speaker_audio_path = processed_path
    
    # Optionally, extract speaker embedding (you can store it for later use)
    embedding = extract_speaker_embedding(processed_path)
    print("Speaker embedding extracted:", embedding.shape)
    
    return jsonify(success=True)

@app.route("/clone_voice", methods=["POST"])
def clone_voice():
    global speaker_audio_path
    if speaker_audio_path is None:
        return jsonify(success=False, message="No speaker audio uploaded."), 400

    data = request.get_json()
    lyrics = data.get("lyrics", "")
    if not lyrics:
        return jsonify(success=False, message="No lyrics provided."), 400

    # Choose the voice generation method: ElevenLabs or Coqui
    use_elevenlabs = True  # Change as needed

    if use_elevenlabs:
        generated_path = generate_voice_11labs(lyrics, voice_id="Rachel")
    else:
        generated_path = generate_voice_coqui(lyrics, speaker_audio_path)
    
    # Return the path (or URL) to the generated audio so the front end can load it
    return jsonify(success=True, audio_url=f"/generated/{os.path.basename(generated_path)}")

@app.route("/generated/<path:filename>")
def serve_generated(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
