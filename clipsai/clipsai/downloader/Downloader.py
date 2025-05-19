"""
Base class for video downloading functionality.
"""
# standard library imports
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# current package imports
from ..media.video_file import VideoFile
from ..media.exceptions import MediaEditorError

# local imports
from ..filesys.file import File
from ..filesys.manager import FileSystemManager
from ..utils.type_checker import TypeChecker

SUCCESS = 0

class Downloader(ABC):
    """
    Abstract base class for video downloading functionality.
    
    This class provides the interface and common functionality for downloading
    videos from various sources. Specific implementations (e.g. YouTube) should
    inherit from this class.
    
    Attributes
    ----------
    _file_system_manager : FileSystemManager
        Manager for file system operations
    _type_checker : TypeChecker
        Utility for type checking
    """
    
    def __init__(self) -> None:
        """
        Initialize the Downloader.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self._file_system_manager = FileSystemManager()
        self._type_checker = TypeChecker()
    
    @abstractmethod
    def download(
        self,
        url: str,
        output_path: str,
        overwrite: bool = True,
        quality: str = "best",
        format: str = "mp4"
    ) -> Optional[VideoFile]:
        """
        Download a video from the source.
        
        Parameters
        ----------
        url : str
            URL of the video to download
        output_path : str
            Path where the video should be saved
        overwrite : bool, optional
            Whether to overwrite existing file, by default True
        quality : str, optional
            Desired video quality, by default "best"
        format : str, optional
            Desired output format, by default "mp4"
            
        Returns
        -------
        Optional[VideoFile]
            VideoFile object if download successful, None otherwise
        """
        pass
    
    def reduce_quality(
        self,
        video_file: VideoFile,
        output_path: str,
        target_bitrate: Optional[int] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        crf: str = "23",
        preset: str = "medium",
        overwrite: bool = True
    ) -> Optional[VideoFile]:
        """
        Reduce the quality of a video file.
        
        Parameters
        ----------
        video_file : VideoFile
            The video file to reduce quality of
        output_path : str
            Path where the reduced quality video should be saved
        target_bitrate : Optional[int], optional
            Target bitrate in bits per second, by default None
        target_resolution : Optional[Tuple[int, int]], optional
            Target resolution as (width, height), by default None
        crf : str, optional
            Constant Rate Factor (0-51, lower is better quality), by default "23"
        preset : str, optional
            Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, 
            slow, slower, veryslow), by default "medium"
        overwrite : bool, optional
            Whether to overwrite existing file, by default True
            
        Returns
        -------
        Optional[VideoFile]
            VideoFile object if quality reduction successful, None otherwise
        """
        # Validate inputs
        self._type_checker.assert_type(video_file, VideoFile)
        if overwrite:
            self._file_system_manager.assert_parent_dir_exists(File(output_path))
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(output_path)
            
        self._file_system_manager.assert_paths_not_equal(
            video_file.path,
            output_path,
            "video_file path",
            "output_path"
        )
        
        # Build ffmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", video_file.path,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", crf
        ]
        
        # Add resolution scaling if requested
        if target_resolution:
            width, height = target_resolution
            ffmpeg_cmd.extend(["-vf", f"scale={width}:{height}"])
            
        # Add bitrate target if requested
        if target_bitrate:
            ffmpeg_cmd.extend(["-b:v", str(target_bitrate)])
            
        # Add output path
        ffmpeg_cmd.append(output_path)
        
        # Run ffmpeg command
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True
        )
        
        # Log result
        msg = (
            f"\n{'-' * 40}\n"
            f"video_file path: '{video_file.path}'\n"
            f"output_path: '{output_path}'\n"
            f"target_bitrate: '{target_bitrate}'\n"
            f"target_resolution: '{target_resolution}'\n"
            f"crf: '{crf}'\n"
            f"preset: '{preset}'\n"
            f"Terminal return code: '{result.returncode}'\n"
            f"Output: '{result.stdout}'\n"
            f"Err Output: '{result.stderr}'\n"
            f"{'-' * 40}\n"
        )
        
        # Check for success
        if result.returncode != SUCCESS:
            err_msg = (
                f"Reducing quality of video file '{video_file.path}' to '{output_path}' "
                f"was unsuccessful. Here is some helpful troubleshooting information:\n{msg}"
            )
            logging.error(err_msg)
            return None
            
        # Return new VideoFile object
        reduced_video = VideoFile(output_path)
        reduced_video.assert_exists()
        return reduced_video
