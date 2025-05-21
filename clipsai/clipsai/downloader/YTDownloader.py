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
from ..media.audiovideo_file import AudioVideoFile
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
    ) -> Optional[AudioVideoFile]:
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
        Optional[AudioVideoFile]
            AudioVideoFile object if download successful, None otherwise
        """
        logging.info(f"Downloading video: {url}")
        logging.info(f"Quality: {quality}, Format: {format}")
        
        # Validate inputs
        if not url.startswith(("https://www.youtube.com/", "https://youtu.be/")):
            raise MediaEditorError(f"Invalid YouTube URL: {url}")
            
        if overwrite:
            self._file_system_manager.assert_parent_dir_exists(File(output_path))
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(output_path)
            
        # Build yt-dlp command based on quality specification
        if quality == "best":
            format_spec = f"bestvideo[ext={format}]+bestaudio[ext={format}]/best[ext={format}]"
        elif quality == "worst":
            format_spec = f"worstvideo[ext={format}]+worstaudio[ext={format}]/worst[ext={format}]"
        else:
            # Try to extract numeric height from quality (e.g., "720p" -> "720")
            try:
                height = ''.join(filter(str.isdigit, quality))
                format_spec = f"bestvideo[ext={format}][height<={height}]+bestaudio[ext={format}]/best[ext={format}]"
            except:
                format_spec = f"best[ext={format}]"
        
        command = [
            "yt-dlp",
            "-f", format_spec,
            "-o", output_path,
            url
        ]
        
        # Run yt-dlp command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        
        # Check for success
        if result.returncode != SUCCESS:
            logging.error(f"Download failed: {result.stderr}")
            return None
            
        # Return new AudioVideoFile object
        video_file = AudioVideoFile(output_path)
        video_file.assert_exists()
        logging.info(f"Download complete: {output_path}")
        return video_file
