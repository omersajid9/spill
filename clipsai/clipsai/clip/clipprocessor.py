"""
Video clipping functionality using ClipsAI.
"""
import os
from typing import List, Dict
import logging

from .clip import Clip
from .clipfinder import ClipFinder
from ..transcribe.transcriber import Transcriber
from ..media.editor import MediaEditor
from ..media.audiovideo_file import AudioVideoFile


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
        self.media_editor = MediaEditor()
        
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
            clip_filename = f"{video_id}_clip_{clip_index:03d}_{current_time:.1f}s_to_{end_time:.1f}s.mp4"
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
        
        # Use the video from yt_downloads instead of the temporary path
        downloaded_video_path = os.path.join(self.downloads_dir, f"{video_id}.mp4")
        if not os.path.exists(downloaded_video_path):
            logging.error(f"Video not found in downloads directory: {downloaded_video_path}")
            return []
            
        output_dir = os.path.join(self.clips_dir, video_id)
        logging.info(f"Processing video: {downloaded_video_path}")
        logging.info(f"Output directory: {output_dir}")
        
        # Check if clips already exist
        if os.path.exists(output_dir):
            clips = []
            for file in os.listdir(output_dir):
                if file.endswith('.mp4'):
                    parts = file.split('.mp4')[0].split('_')
                    try:
                        start_time = float(parts[4][:-1])
                        end_time = float(parts[6][:-1])
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
        transcription = transcriber.transcribe(audio_file_path=downloaded_video_path)
        clipfinder = ClipFinder()
        clips = clipfinder.find_clips(transcription=transcription)
        
        # If no clips found, create manual clips
        if not clips:
            clip_info = self._create_manual_clips(downloaded_video_path, output_dir, video_id)
        else:
            # Save clip information
            clip_info = []
            for i, clip in enumerate(clips):
                clip_filename = f"{video_id}_clip_{i+1:03d}_{clip.start_time:.1f}s_to_{clip.end_time:.1f}s.mp4"
                clip_path = os.path.join(output_dir, clip_filename)
                clip_info.append({
                    "filename": clip_filename,
                    "path": clip_path,
                    "start_time": clip.start_time,
                    "end_time": clip.end_time
                })
        
        # Create the actual video clips using MediaEditor
        video_file = AudioVideoFile(downloaded_video_path)
        for clip in clip_info:
            logging.info(f"Creating clip: {clip['filename']}")
            logging.info(f"Input video: {downloaded_video_path}")
            logging.info(f"Output path: {clip['path']}")
            success = self.media_editor.trim(
                media_file=video_file,
                start_time=clip["start_time"],
                end_time=clip["end_time"],
                trimmed_media_file_path=clip["path"],
                overwrite=True,
                video_codec="libx264",
                audio_codec="aac",
                crf="23",
                preset="medium"
            )
            if not success:
                logging.error(f"Failed to create clip: {clip['filename']}")
            else:
                logging.info(f"Successfully created clip: {clip['filename']}")
        
        return clip_info 