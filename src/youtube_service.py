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
class YouTubeService:
    def __init__(self):
        self.downloader = YoutubeCommentDownloader()
        self.transcriber = None

    def _load_transcriber(self):
        if self.transcriber is None:
            # Using 'base' instead of 'tiny' for better accuracy in languages like Hindi
            self.transcriber = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-base", 
                chunk_length_s=30,
            )

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
        """Download audio from a YouTube video with OAuth bypass."""
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "audio.mp3")
        try:
            # Enabling OAuth is the most reliable way to bypass bot detection.
            print(f"Attempting to download audio for {video_url} using OAuth...")
            # Use 'small' model logic internally if possible or base
            video = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
            audio_stream = video.streams.filter(only_audio=True).first()
            
            if not audio_stream:
                raise ValueError("No audio stream available for this video.")
                
            audio_stream.download(filename=output_path)
            return output_path
        except Exception as e:
            print(f"Error downloading audio: {e}")
            raise e

    def transcribe_audio(self, audio_path):
        """Transcribe audio to text with timestamps using Whisper (Native)."""
        temp_dir = tempfile.mkdtemp()
        wav_path = os.path.join(temp_dir, "temp_audio.wav")
        try:
            self._load_transcriber()
            
            # Use moviepy to safely extract to wav format acceptable by scipy
            clip = AudioFileClip(audio_path)
            clip.write_audiofile(wav_path, codec='pcm_s16le', fps=16000, logger=None)
            clip.close()
            
            # Read via scipy
            sample_rate, data = wavfile.read(wav_path)
            if len(data.shape) > 1:
                data = data.mean(axis=1) # stereo to mono
                
            # Normalize to float32 between -1 and 1
            data = data.astype(np.float32) / 32768.0
            
            # ANALYSIS CHANGE: Removed task="translate" to catch multilingual threats in native speech
            result = self.transcriber(
                data, 
                return_timestamps=True
            )
            
            return result.get('chunks', [])
        except Exception as e:
            print(f"Error in transcription: {e}")
            return []
        finally:
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

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
