from moviepy import AudioFileClip
from transformers import pipeline
import torch
import scipy.io.wavfile as wavfile
import numpy as np

def transcribe_with_timestamps(audio_path):
    try:
        clip = AudioFileClip(audio_path)
        wav_path = "temp_audio.wav"
        print("Writing wav...")
        clip.write_audiofile(wav_path, codec='pcm_s16le', fps=16000, logger=None)
        
        print("Reading wav...")
        sample_rate, data = wavfile.read(wav_path)
        # Convert to mono if necessary
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        # Normalize to float32 between -1 and 1
        data = data.astype(np.float32) / 32768.0
        
        print(f"Audio loaded: {len(data)} samples at {sample_rate} Hz")
        print("Loading transcriber...")
        transcriber = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-tiny", 
            chunk_length_s=30,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print("Transcribing...")
        result = transcriber(data, return_timestamps=True)
        return result['chunks']
    except Exception as e:
        print(f"Error in transcription: {e}")
        return []

if __name__ == "__main__":
    from pytubefix import YouTube
    video_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    video = YouTube(video_url)
    audio_stream = video.streams.filter(only_audio=True).first()
    audio_path = "test_audio.mp4"
    audio_stream.download(filename=audio_path)
    chunks = transcribe_with_timestamps(audio_path)
    for c in chunks:
        print(c)
