from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import re
from pytubefix import YouTube
try:
    from moviepy import AudioFileClip
except ImportError:
    from moviepy.editor import AudioFileClip
import torch
import scipy.io.wavfile as wavfile
import numpy as np
from transformers import pipeline
import os
import tempfile
import whisper

class YouTubeService:
    def __init__(self):
        self.downloader = YoutubeCommentDownloader()
        self.transcriber = None

    def _load_transcriber(self):
        if self.transcriber is None:
            # Check for GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Whisper 'small' for YouTube speech analysis on {device}...")
            # Optimization: Load on GPU if possible
            self.transcriber = whisper.load_model("small", device=device)

    def get_video_id(self, url):
        """Extract video ID from YouTube URL."""
        # Examples:
        # https://www.youtube.com/watch?v=VIDEO_ID
        # https://youtu.be/VIDEO_ID
        video_id = None
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        return video_id

    def fetch_comments(self, video_url, limit=50):
        """Fetch top comments from a video."""
        video_id = self.get_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        print(f"Fetching comments for video ID: {video_id}")
        
        comments = []
        try:
            # Try fetching without specifics first (defaults to POPULAR usually)
            generator = self.downloader.get_comments(video_id)
            for i, comment in enumerate(generator):
                if i >= limit:
                    break
                comments.append({
                    "author": comment.get('author', 'Unknown'),
                    "text": comment.get('text', ''),
                    "likes": comment.get('votes', 0),
                    "time": comment.get('time', '')
                })
            
            # If still empty, try fallback or just log
            if not comments:
                print(f"Warning: No comments returned by downloader for {video_id}")
                
        except Exception as e:
            print(f"Error fetching comments: {e}")
            # Do not raise if audio transcription might still work
            # Just return empty list so the process can continue
            
        return comments

    def download_audio(self, video_url):
        """Download audio from a YouTube video using a robust multi-stage downloader."""
        temp_dir = tempfile.mkdtemp()
        final_path = os.path.join(temp_dir, "audio.mp3")
        
        # STAGE 1: Try pytubefix (with OAuth support) - Best for Colab bot bypass
        try:
            print(f"Attempting Stage 1 download (pytubefix with OAuth): {video_url}")
            from pytubefix import YouTube
            # use_oauth=True/allow_oauth_cache=True allows using the login from the notebook cell
            yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
            
            # Filter for audio only
            stream = yt.streams.filter(only_audio=True).first()
            if stream:
                downloaded_file = stream.download(output_path=temp_dir, filename="audio_raw")
                # Convert to mp3 using ffmpeg (very fast)
                import subprocess
                subprocess.run(["ffmpeg", "-i", downloaded_file, "-q:a", "0", "-map", "a", final_path, "-y"], capture_output=True)
                
                if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    print(f"✅ Stage 1 SUCCESS: Downloaded {os.path.getsize(final_path)} bytes")
                    return final_path
        except Exception as e:
            print(f"⚠️ Stage 1 Failed (pytubefix): {e}")

        # STAGE 2: Try yt-dlp (Robust fallback)
        try:
            print(f"Attempting Stage 2 download (yt-dlp): {video_url}")
            import subprocess
            output_template = os.path.join(temp_dir, "audio.%(ext)s")
            cmd = [
                "yt-dlp",
                "-x",
                "--audio-format", "mp3",
                "--output", output_template,
                "--no-check-certificate",
                "--prefer-free-formats",
                video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check for file
            files = [f for f in os.listdir(temp_dir) if f.endswith('.mp3')]
            if files:
                final_path = os.path.join(temp_dir, files[0])
                if os.path.getsize(final_path) > 0:
                    print(f"✅ Stage 2 SUCCESS: Downloaded {os.path.getsize(final_path)} bytes")
                    return final_path
            
            if result.returncode != 0:
                print(f"yt-dlp error: {result.stderr}")
        except Exception as e:
            print(f"⚠️ Stage 2 Failed (yt-dlp): {e}")

        raise ValueError("Audio download failed. YouTube is blocking the request. Please run the 'YouTube Authentication' cell in the notebook, enter the code, and try again.")

    def transcribe_audio(self, audio_path):
        """Transcribe audio to text with timestamps using Whisper (Native)."""
        try:
            self._load_transcriber()
            
            print(f"Transcribing YouTube audio: {audio_path}")
            # Use native whisper model transcription
            result = self.transcriber.transcribe(audio_path)
            
            # Convert whisper segments to the format expected by our API
            chunks = []
            for segment in result.get('segments', []):
                chunks.append({
                    'text': segment['text'],
                    'timestamp': (segment['start'], segment['end'])
                })
            
            return chunks
        except Exception as e:
            print(f"Error in transcription: {e}")
            return []
        finally:
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass

    def extract_video_content(self, video_url):
        """Extract audio and transcribe text with timestamps from a YouTube video."""
        audio_path = self.download_audio(video_url)
        try:
            chunks = self.transcribe_audio(audio_path)
            return chunks
        finally:
            # Cleanup audio file and its parent temp dir
            if os.path.exists(audio_path):
                import shutil
                shutil.rmtree(os.path.dirname(audio_path), ignore_errors=True)

if __name__ == "__main__":
    # Test
    yt = YouTubeService()
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw" # Me at the zoo
    comments = yt.fetch_comments(url, limit=5)
    for c in comments:
        print(c)

    # Extract video content
    print("Extracting video content...")
    transcript = yt.extract_video_content(url)
    print("Transcript:", transcript)
