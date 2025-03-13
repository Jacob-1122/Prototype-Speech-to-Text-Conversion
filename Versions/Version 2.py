import whisper
from faster_whisper import WhisperModel
import librosa
import noisereduce as nr
from pydub import AudioSegment
import os
import soundfile as sf
from tqdm import tqdm
import torch.hub
import requests




# ------------------- Convert Audio Format -------------------
def convert_audio_format(input_path, output_format="wav"):
    """
    Converts an audio file to WAV format for processing.

    Args:
        input_path (str): Path to the input audio file.
        output_format (str): Desired format (default: WAV).

    Returns:
        str: Path to converted audio file.
    """
    output_path = input_path.rsplit(".", 1)[0] + f".{output_format}"

    print(f"[10%] Converting {input_path} to {output_format}...")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono 16kHz
    audio.export(output_path, format=output_format)

    print(f"[20%] Converted audio saved: {output_path}")
    return output_path


# ------------------- Preprocess Audio -------------------
def preprocess_audio(audio_path, output_path="cleaned_audio.wav"):
    """
    Preprocess an ATC audio file:
    - Convert to mono
    - Resample to 16kHz
    - Reduce background noise
    - Normalize volume
    - Trim silence

    Args:
        audio_path (str): Path to the input audio file.
        output_path (str): Path to save the processed audio.

    Returns:
        str: Path to the processed audio file.
    """
    print(f"[30%] Loading and preprocessing audio: {audio_path}")

    # Load audio with Librosa
    y, sr = librosa.load(audio_path, sr=None, mono=True)  # Preserve original sample rate

    # Resample to 16kHz
    target_sr = 16000
    if sr != target_sr:
        print(f"[40%] Resampling audio to {target_sr}Hz...")
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    print(f"[50%] Reducing noise...")
    y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)

    print(f"[60%] Normalizing and trimming silence...")
    y_normalized = librosa.util.normalize(y_denoised)
    y_trimmed, _ = librosa.effects.trim(y_normalized, top_db=20)

    # Save processed audio
    sf.write(output_path, y_trimmed, sr)
    print(f"[70%] Processed audio saved: {output_path}")

    return output_path


# ------------------- Transcribe Audio -------------------
def transcribe_audio(audio_path):
    """
    Transcribes the given audio file using the faster-whisper model.
    """
    # Automatically detect if GPU (CUDA) is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load optimized Whisper model
    model = WhisperModel("medium", device=device, compute_type="int8")

    print(f"[80%] Transcribing {audio_path}...")

    # Use beam search for better accuracy
    segments, info = model.transcribe(audio_path, beam_size=5)

    # Join transcribed segments
    transcription = " ".join(segment.text for segment in segments)

    print(f"[100%] Transcription completed.")
    return transcription

# ------------------- Batch Process Multiple Files -------------------
def process_audio_directory(directory_path):
    """
    Processes all audio files in a directory sequentially for ATC transcription.

    Args:
        directory_path (str): Path to directory containing audio files.

    Returns:
        dict: Dictionary of file paths and their transcriptions.
    """
    all_files = sorted([os.path.join(directory_path, f) for f in os.listdir(directory_path)
                        if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(('.wav', '.mp3', '.aac'))])

    results = {}

    for file_path in tqdm(all_files, total=len(all_files), desc="Processing Batch"):
        print(f"\n📢 Processing: {file_path}")

        # Convert if necessary
        if not file_path.endswith(".wav"):
            converted_audio = convert_audio_format(file_path)
        else:
            converted_audio = file_path

        # Preprocess the audio (denoising, resampling)
        cleaned_audio = preprocess_audio(converted_audio)

        # Transcribe the cleaned audio
        transcription = transcribe_audio(cleaned_audio)

        results[file_path] = transcription

    return results

def download_whisper_model(model_name="medium"):
    """
    Downloads the Whisper model with a progress bar in Windows.
    """
    model_dir = os.path.expanduser("~/.cache/whisper")
    os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
    model_path = os.path.join(model_dir, f"{model_name}.pt")

    if not os.path.exists(model_path):
        print(f"📥 Downloading Whisper {model_name} model...")

        # Whisper model URL
        model_url = f"https://openaipublic.blob.core.windows.net/whisper/models/{model_name}.pt"

        # Stream download with progress bar
        response = requests.get(model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(model_path, "wb") as file, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                bar.update(len(chunk))

        print(f"✅ Model downloaded: {model_path}")
    else:
        print(f"🔄 Whisper {model_name} model already exists.")

    return model_path


# Use the manually downloaded model
model_path = download_whisper_model("medium")
print(f"✅ Model downloaded at {model_path}")
# ------------------- Run Batch Processing -------------------
if __name__ == "__main__":
    audio_directory = "Audio/"

    # Process files SEQUENTIALLY (one at a time)
    transcriptions = process_audio_directory(audio_directory)

    # Save transcriptions
    with open("transcriptions.txt", "w") as f:
        for file, text in transcriptions.items():
            f.write(f"File: {file}\nTranscription:\n{text}\n\n")

    print("\n✅ All files processed. Transcriptions saved to 'transcriptions.txt'.")
