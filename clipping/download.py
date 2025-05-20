"""
Step 1: YouTube video downloading functionality for ClipsAI.
"""
import os
import tempfile
import re
from typing import Optional, Tuple, List, Dict

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
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract the video ID from a YouTube URL.
        
        Parameters
        ----------
        url : str
            YouTube URL to extract ID from
            
        Returns
        -------
        Optional[str]
            Video ID if found, None otherwise
        """
        # Handle different YouTube URL formats
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard YouTube URLs
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',  # Short YouTube URLs
            r'(?:embed\/)([0-9A-Za-z_-]{11})'  # Embedded YouTube URLs
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_available_videos(self) -> List[Dict[str, str]]:
        """Get list of available videos in the downloads directory.
        
        Returns:
            List of dictionaries containing video information
        """
        videos = []
        for file in os.listdir(self.output_dir):
            if file.endswith(('.mp4', '.mkv', '.avi')):
                video_path = os.path.join(self.output_dir, file)
                video_id = os.path.splitext(file)[0]  # Remove extension
                videos.append({
                    "path": video_path,
                    "id": video_id,
                    "name": file
                })
        return videos

        
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
            # Extract video ID
            video_id = self._extract_video_id(url)
            if not video_id:
                return None, "Invalid YouTube URL. Could not extract video ID."
            
            # Generate output path using video ID
            output_path = os.path.join(self.output_dir, f"video_{video_id}.mp4")
            
            # Check if video already exists
            if os.path.exists(output_path):
                return str(output_path), f"Video already downloaded at: {output_path}"
            
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