"""
Step 1: YouTube video downloading functionality for ClipsAI.
"""
import os
import tempfile
from typing import Optional, Tuple

from clipsai.downloader import YTDownloader
from clipsai.media.video_file import VideoFile

class VideoDownloader:
    """Step 1: Download YouTube videos for processing."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the video downloader.
        
        Parameters
        ----------
        output_dir : Optional[str]
            Directory to store downloaded videos. If None, uses a temporary directory.
        """
        self.downloader = YTDownloader()
        self.output_dir = output_dir or tempfile.mkdtemp()
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_video(self, url: str, quality: str = "best") -> Tuple[str, str]:
        """
        Download a video from YouTube.
        
        Parameters
        ----------
        url : str
            YouTube URL to download
        quality : str
            Video quality ("best" or "worst")
            
        Returns
        -------
        Tuple[str, str]
            Tuple of (video path, status message)
        """
        try:
            # Generate output path
            output_path = os.path.join(self.output_dir, f"video_{hash(url)}.mp4")
            
            # Download video
            video_file = self.downloader.download(
                url=url,
                output_path=output_path,
                quality=quality,
                format="mp4"
            )
            
            if video_file is None:
                return None, "Failed to download video. Please check the URL and try again."
                
            return str(video_file.path), f"Successfully downloaded video to: {video_file.path}"
            
        except Exception as e:
            return None, f"Error: {str(e)}" 