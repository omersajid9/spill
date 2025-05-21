"""
Video clipping functionality using ClipsAI.
"""
import os
from typing import List, Dict

from .clip import Clip
from .clipfinder import ClipFinder
from ..transcribe.transcriber import Transcriber


from pathlib import Path
import cv2

class ClipProcessor:
    def __init__(self, data_store_dir: str):
        """Initialize the clip processor.
        
        Args:
            data_store_dir: Base directory for storing data
        """
        self.downloads_dir = os.path.join(data_store_dir, "yt_downloads")
        self.clips_dir = os.path.join(data_store_dir, "yt_clipped")
        os.makedirs(self.clips_dir, exist_ok=True)
        
    def _create_manual_clips(self, video_path: str, output_dir: str, video_id: str) -> List[Dict[str, str]]:
        """Create manual clips by splitting video into 149-second segments with no overlap.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save clips
            video_id: ID of the video
            
        Returns:
            List of dictionaries containing clip information
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)  # Convert to int for range()
        
        # Calculate clip parameters
        clip_duration = 149  # seconds
        clip_info = []
        
        # Create clips
        clip_index = 1
        
        for current_time in range(0, duration, clip_duration):
            end_time = min(current_time + clip_duration, duration)
            
            # Create clip filename
            clip_filename = f"video_{video_id}_clip_{clip_index:03d}_{current_time:.1f}s_to_{end_time:.1f}s.mp4"
            clip_path = os.path.join(output_dir, clip_filename)
            
            clip_info.append({
                "filename": clip_filename,
                "path": clip_path,
                "start_time": current_time,
                "end_time": end_time
            })
            
            clip_index += 1
            
        cap.release()
        return clip_info
        
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
        print(output_dir)
        # Check if clips already exist
        if os.path.exists(output_dir):
            clips = []
            for file in os.listdir(output_dir):
                print("file", file)
                if file.endswith('.mp4'):
                    parts = file.split('.mp4')[0].split('_')
                    print(parts)
                    try:
                        start_time = float(parts[4][:-1])
                        end_time = float(parts[6][:-1])
                        print(start_time, end_time)
                        clips.append({
                            "filename": file,
                            "path": os.path.join(output_dir, file),
                            "start_time": start_time,
                            "end_time": end_time
                        })
                    except (IndexError, ValueError) as e:
                        continue
            return clips
            
        # If no clips exist, process the video
        os.makedirs(output_dir, exist_ok=True)
        
        # Transcribe and find clips
        transcriber = Transcriber()
        transcription = transcriber.transcribe(audio_file_path=video_path)
        clipfinder = ClipFinder()
        clips = clipfinder.find_clips(transcription=transcription)
        
        # If no clips found, create manual clips
        if not clips:
            return self._create_manual_clips(video_path, output_dir, video_id)
        
        # Save clip information
        clip_info = []
        for i, clip in enumerate(clips):
            clip_filename = f"video_{video_id}_clip_{i+1:03d}_{clip.start_time:.1f}s_to_{clip.end_time:.1f}s.mp4"
            clip_path = os.path.join(output_dir, clip_filename)
            clip_info.append({
                "filename": clip_filename,
                "path": clip_path,
                "start_time": clip.start_time,
                "end_time": clip.end_time
            })
        
        return clip_info 