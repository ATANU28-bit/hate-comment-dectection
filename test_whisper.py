import torch
from transformers import pipeline
import time
from moviepy import AudioFileClip
import scipy.io.wavfile as wavfile
import numpy as np
from src.inference import HateCommentClassifier

print("Loading model...")
transcriber = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-tiny", 
    chunk_length_s=30,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

# Initialize the classifier
classifier = HateCommentClassifier()

def process_video(video_path):
    print(f"Processing video: {video_path}")

    # Extract audio from video using moviepy (which resolves local ffmpeg path issues)
    audio_path = "temp_audio.wav"
    try:
        clip = AudioFileClip(video_path)
        clip.write_audiofile(audio_path, codec='pcm_s16le', fps=16000, logger=None)
        clip.close()
    except Exception as e:
        print(f"Failed to extract audio with moviepy: {e}")
        return

    # Transcribe audio
    print("Transcribing audio...")
    try:
        sample_rate, data = wavfile.read(audio_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32) / 32768.0
        result = transcriber(data, return_timestamps=True)
    except Exception as e:
        print(f"Failed to read or transcribe audio: {e}")
        return

    # Analyze transcriptions for abusive language
    print("Analyzing transcriptions...")
    for segment in result["chunks"]:
        text = segment["text"]
        start_time = segment["timestamp"][0]
        end_time = segment["timestamp"][1]

        classification = classifier.predict(text)
        if classification["label"] == "Abusive":
            print(f"Abusive language detected from {start_time} to {end_time}: {text}")

# Example usage
if __name__ == "__main__":
    video_path = "example_video.mp4"  # Replace with your video file path
    process_video(video_path)
