from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import re

class YouTubeService:
    def __init__(self):
        self.downloader = YoutubeCommentDownloader()

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
            # generator
            generator = self.downloader.get_comments(video_id, sort_by=SORT_BY_POPULAR)
            for i, comment in enumerate(generator):
                if i >= limit:
                    break
                comments.append({
                    "author": comment.get('author', 'Unknown'),
                    "text": comment.get('text', ''),
                    "likes": comment.get('votes', 0),
                    "time": comment.get('time', '')
                })
        except Exception as e:
            print(f"Error fetching comments: {e}")
            # If fetching fails, return empty or raise
            raise e
            
        return comments

if __name__ == "__main__":
    # Test
    yt = YouTubeService()
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw" # Me at the zoo
    comments = yt.fetch_comments(url, limit=5)
    for c in comments:
        print(c)
