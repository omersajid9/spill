"""
YouTube video downloader implementation.
"""
# standard library imports
import logging
import os
import subprocess
from typing import Optional

# current package imports
from .Downloader import Downloader
from ..media.video_file import VideoFile
from ..media.exceptions import MediaEditorError

# local imports
from ..filesys.file import File
from ..filesys.manager import FileSystemManager

SUCCESS = 0

class YTDownloader(Downloader):
    """
    YouTube video downloader implementation.
    
    This class provides functionality to download videos from YouTube using yt-dlp.
    It inherits from the base Downloader class and implements the abstract download method.
    
    Attributes
    ----------
    _file_system_manager : FileSystemManager
        Manager for file system operations
    """
    
    def __init__(self) -> None:
        """
        Initialize the YouTube downloader.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        super().__init__()
        
        # Check if yt-dlp is installed
        try:
            subprocess.run(
                ["yt-dlp", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise MediaEditorError(
                "yt-dlp is not installed. Please install it using: "
                "pip install yt-dlp"
            )
    
    def download(
        self,
        url: str,
        output_path: str,
        overwrite: bool = True,
        quality: str = "best",
        format: str = "mp4"
    ) -> Optional[VideoFile]:
        """
        Download a video from YouTube.
        
        Parameters
        ----------
        url : str
            YouTube URL of the video to download
        output_path : str
            Path where the video should be saved
        overwrite : bool, optional
            Whether to overwrite existing file, by default True
        quality : str, optional
            Desired video quality (best, worst, or specific format like 720p), 
            by default "best"
        format : str, optional
            Desired output format, by default "mp4"
            
        Returns
        -------
        Optional[VideoFile]
            VideoFile object if download successful, None otherwise
        """
        # Validate inputs
        if not url.startswith(("https://www.youtube.com/", "https://youtu.be/")):
            raise MediaEditorError(f"Invalid YouTube URL: {url}")
            
        if overwrite:
            self._file_system_manager.assert_parent_dir_exists(File(output_path))
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(output_path)
            
        # Build yt-dlp command
        ytdlp_cmd = [
            "yt-dlp",
            "--no-playlist",  # Don't download playlists
            "--no-warnings",  # Suppress warnings
            "--no-progress",  # Don't show progress bar
            "-f", f"bestvideo[ext={format}]+bestaudio[ext={format}]/best[ext={format}]" 
                  if quality == "best" else f"worst[ext={format}]",
            "-o", output_path,
            url
        ]
        
        # Run yt-dlp command
        result = subprocess.run(
            ytdlp_cmd,
            capture_output=True,
            text=True
        )
        
        # Log result
        msg = (
            f"\n{'-' * 40}\n"
            f"url: '{url}'\n"
            f"output_path: '{output_path}'\n"
            f"quality: '{quality}'\n"
            f"format: '{format}'\n"
            f"Terminal return code: '{result.returncode}'\n"
            f"Output: '{result.stdout}'\n"
            f"Err Output: '{result.stderr}'\n"
            f"{'-' * 40}\n"
        )
        
        # Check for success
        if result.returncode != SUCCESS:
            err_msg = (
                f"Downloading video from '{url}' to '{output_path}' was unsuccessful. "
                f"Here is some helpful troubleshooting information:\n{msg}"
            )
            logging.error(err_msg)
            return None
            
        # Return new VideoFile object
        video_file = VideoFile(output_path)
        video_file.assert_exists()
        return video_file
