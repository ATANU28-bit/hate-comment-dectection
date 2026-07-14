import os
import whisper
import torch
try:
    import moviepy.editor as mp
except ImportError:
    import moviepy as mp
import tempfile
from pathlib import Path

class AudioService:
    def __init__(self, model_name="small"):
        self.model = None
        self.model_name = model_name

    def _load_model(self):
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Whisper model: {self.model_name} (Multilingual/GPURready) on {device}...")
            self.model = whisper.load_model(self.model_name, device=device)
            print("Whisper model loaded.")

    def extract_audio(self, video_path):
        """Extract audio from video file and return path to temporary audio file."""
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()

        print(f"Extracting audio from {video_path}...")
        video = mp.VideoFileClip(str(video_path))
        video.audio.write_audiofile(temp_audio_path, logger=None)
        video.close()
        
        return temp_audio_path

    def transcribe(self, file_path):
        """Transcribe audio or video file in its native language."""
        self._load_model()
        
        file_ext = Path(file_path).suffix.lower()
        is_video = file_ext in ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
        
        temp_audio_path = None
        try:
            if is_video:
                temp_audio_path = self.extract_audio(file_path)
                process_path = temp_audio_path
            else:
                process_path = file_path

            print(f"Analyzing speech in {process_path}...")
            # Removed task="translate" to keep native language
            # This allows the multilingual classifier to see the original toxic words
            result = self.model.transcribe(str(process_path))
            
            # segments contains timestamps and text
            return {
                "text": result['text'],
                "segments": [
                    {
                        "start": s['start'],
                        "end": s['end'],
                        "text": s['text']
                    } for s in result['segments']
                ]
            }
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass

if __name__ == "__main__":
    # Quick test
    # service = AudioService()
    # result = service.transcribe("path/to/audio/or/video")
    # print(result['text'])
    pass
