"""
Video clipping functionality using ClipsAI.
"""
import os
from typing import List, Dict
from clipsai import ClipFinder, Transcriber
from pathlib import Path

class ClipProcessor:
    def __init__(self, data_store_dir: str):
        """Initialize the clip processor.
        
        Args:
            data_store_dir: Base directory for storing data
        """
        self.downloads_dir = os.path.join(data_store_dir, "yt_downloads")
        self.clips_dir = os.path.join(data_store_dir, "yt_clipped")
        os.makedirs(self.clips_dir, exist_ok=True)
    
    def get_available_videos(self) -> List[Dict[str, str]]:
        """Get list of available videos in the downloads directory.
        
        Returns:
            List of dictionaries containing video information
        """
        videos = []
        for file in os.listdir(self.downloads_dir):
            if file.endswith(('.mp4', '.mkv', '.avi')):
                video_path = os.path.join(self.downloads_dir, file)
                video_id = os.path.splitext(file)[0]  # Remove extension
                videos.append({
                    "path": video_path,
                    "id": video_id,
                    "name": file
                })
        return videos
    
    def process_video(self, video_path: str) -> List[Dict[str, str]]:
        """Process a video to find and save clips.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of dictionaries containing clip information
        """
        # Get video ID from filename
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(self.clips_dir, video_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Transcribe the video
        transcriber = Transcriber()
        transcription = transcriber.transcribe(audio_file_path=video_path)
        
        # Find clips
        clipfinder = ClipFinder()
        clips = clipfinder.find_clips(transcription=transcription)
        
        # Process and save clips
        clip_info = []
        for i, clip in enumerate(clips):
            # Create a descriptive filename
            clip_filename = f"clip_{i+1:03d}_{clip.start_time:.1f}s_to_{clip.end_time:.1f}s.mp4"
            clip_path = os.path.join(output_dir, clip_filename)
            
            # TODO: Implement actual video clipping here
            # For now, we'll just return the clip information
            clip_info.append({
                "filename": clip_filename,
                "path": clip_path,
                "start_time": clip.start_time,
                "end_time": clip.end_time
            })
        
        return clip_info 