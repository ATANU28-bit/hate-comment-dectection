from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import sys

def test():
    downloader = YoutubeCommentDownloader()
    video_id = 'jNQXAC9IVRw' # Me at the zoo
    try:
        print("Attempting to fetch comments with SORT_BY_POPULAR...")
        generator = downloader.get_comments(video_id, sort_by=SORT_BY_POPULAR)
        for i, comment in enumerate(generator):
            if i >= 1:
                break
            print(f"Success: {comment.get('text')}")
    except Exception as e:
        print(f"Failed with SORT_BY_POPULAR: {e}")

    try:
        print("\nAttempting to fetch comments WITHOUT sort_by...")
        generator = downloader.get_comments(video_id)
        for i, comment in enumerate(generator):
            if i >= 1:
                break
            print(f"Success: {comment.get('text')}")
    except Exception as e:
        print(f"Failed without sort_by: {e}")

if __name__ == "__main__":
    test()
