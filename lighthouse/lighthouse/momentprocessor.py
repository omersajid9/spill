"""
Moment retrieval functionality using Lighthouse API.
"""
import os
import logging
import subprocess
from typing import List, Dict, Optional, Set, Tuple
import torch

from .models import CGDETRPredictor

class MomentProcessor:
    def __init__(self, data_store_dir: str):
        """Initialize the moment processor.
        
        Args:
            data_store_dir: Base directory for storing data
        """
        self.moments_dir = os.path.join(data_store_dir, "yt_moments")
        os.makedirs(self.moments_dir, exist_ok=True)
        
        # Initialize Lighthouse model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weight_dir = os.path.join(data_store_dir, "weights")  # This will be /root/spill/data/weights
        os.makedirs(self.weight_dir, exist_ok=True)
        
        # Load required model weights
        self._load_weights()
        
        # Initialize model
        self.weight_path = os.path.join(self.weight_dir, 'clip_slowfast_cg_detr_qvhighlight.ckpt')
        logging.info(f"Initializing CGDETRPredictor on {self.device}")
        
        self.model = CGDETRPredictor(
            self.weight_path, 
            device=self.device, 
            feature_name='clip_slowfast',
            slowfast_path=os.path.join(self.weight_dir, 'SLOWFAST_8x8_R50.pkl'),
            pann_path=os.path.join(self.weight_dir, 'Cnn14_mAP=0.431.pth')
        )
        logging.info("Model initialized successfully")
    
    def _load_weights(self) -> None:
        """Download and load required model weights."""
        # Download CLIP-SlowFast-CG-DETR weights
        clip_slowfast_path = os.path.join(self.weight_dir, 'clip_slowfast_cg_detr_qvhighlight.ckpt')
        if not os.path.exists(clip_slowfast_path):
            logging.info("Downloading CLIP-SlowFast-CG-DETR weights...")
            command = f'wget -P {self.weight_dir} https://zenodo.org/records/13960580/files/clip_slowfast_cg_detr_qvhighlight.ckpt'
            subprocess.run(command, shell=True, check=True)
        
        # Download SlowFast weights
        slowfast_path = os.path.join(self.weight_dir, 'SLOWFAST_8x8_R50.pkl')
        if not os.path.exists(slowfast_path):
            logging.info("Downloading SlowFast weights...")
            command = f'wget -P {self.weight_dir} https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl'
            subprocess.run(command, shell=True, check=True)
        
        # Download PANN weights
        pann_path = os.path.join(self.weight_dir, 'Cnn14_mAP=0.431.pth')
        if not os.path.exists(pann_path):
            logging.info("Downloading PANN weights...")
            command = f'wget -P {self.weight_dir} https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth'
            subprocess.run(command, shell=True, check=True)
        
        logging.info("All model weights downloaded successfully.")
    
    def _is_duplicate_moment(self, moment: Dict[str, float], existing_moments: List[Dict[str, float]], 
                            clip_start_time: float, clip_end_time: float,
                            time_threshold: float = 1.0) -> bool:
        """Check if a moment is a duplicate of any existing moment within the same clip.
        
        Args:
            moment: The moment to check
            existing_moments: List of existing moments
            clip_start_time: Start time of the current clip
            clip_end_time: End time of the current clip
            time_threshold: Time threshold in seconds to consider moments as duplicates
            
        Returns:
            bool: True if the moment is a duplicate
        """
        # Only check moments that fall within this clip's time range
        clip_moments = [
            m for m in existing_moments 
            if m['start_time'] >= clip_start_time and m['end_time'] <= clip_end_time
        ]
        
        for existing in clip_moments:
            # Check if the moments overlap significantly
            overlap_start = max(moment['start_time'], existing['start_time'])
            overlap_end = min(moment['end_time'], existing['end_time'])
            overlap_duration = max(0, overlap_end - overlap_start)
            moment_duration = moment['end_time'] - moment['start_time']
            
            # If more than 50% of the moment overlaps with an existing moment, consider it a duplicate
            if overlap_duration > 0.5 * moment_duration:
                return True
        return False
    
    def _extract_moment_video(self, clip_path: str, start_time: float, end_time: float, 
                            output_path: str) -> bool:
        """Extract a moment video using ffmpeg.
        
        Args:
            clip_path: Path to the source clip
            start_time: Start time of the moment
            end_time: End time of the moment
            output_path: Path to save the moment video
            
        Returns:
            bool: True if extraction was successful
        """
        try:
            duration = end_time - start_time
            command = [
                'ffmpeg', '-y',  # Overwrite output file if exists
                '-ss', str(start_time),  # Start time
                '-i', clip_path,  # Input file
                '-t', str(duration),  # Duration
                '-c', 'copy',  # Copy codec (fast)
                '-avoid_negative_ts', '1',  # Avoid negative timestamps
                output_path
            ]
            logging.info(f"Running ffmpeg command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"FFmpeg error: {result.stderr}")
                return False
                
            logging.info(f"Successfully created moment video: {output_path}")
            return True
        except Exception as e:
            logging.error(f"Error creating moment video: {str(e)}")
            return False
    
    def _get_clip_timestamps(self, clip_filename: str) -> float:
        """Extract start time from clip filename.
        
        Args:
            clip_filename: Name of the clip file (e.g., video_dQw4w9WgXcQ_clip_001_0.0s_to_149.0s.mp4)
            
        Returns:
            Start time in seconds
        """
        try:
            # Extract the time part (e.g., "0.0s_to_149.0s")
            time_part = clip_filename.split('_clip_')[1].split('.mp4')[0]
            # Get the start time part and remove 's'
            start_time_str = time_part.split('_to_')[0].replace('s', '')
            # Convert to float, handling decimal points correctly
            start_time = float(start_time_str.split("_")[1])
            logging.info(f"Parsed start time from filename: {start_time_str} -> {start_time}")
            return start_time
        except Exception as e:
            logging.error(f"Error parsing clip timestamps from {clip_filename}: {str(e)}")
            return 0.0

    def _process_predictions(self, predictions: Dict, video_id: str, clip_path: str, 
                           output_dir: str, existing_moments: List[Dict[str, float]], 
                           clip_start_time: float) -> List[Dict[str, float]]:
        """Process model predictions into moments.
        
        Args:
            predictions: Model predictions
            video_id: Video ID
            clip_path: Path to the clip file
            output_dir: Directory to save moments
            existing_moments: List of existing moments to check for duplicates
            clip_start_time: Start time of the clip in the original video
            
        Returns:
            List of moment dictionaries (only the highest confidence moment)
        """
        if not predictions or 'pred_relevant_windows' not in predictions:
            logging.error(f"Invalid prediction format. Expected key: pred_relevant_windows. Got: {list(predictions.keys())}")
            return []
        
        # Only process the highest confidence moment from this clip
        if not predictions['pred_relevant_windows']:
            return []
            
        start_time, end_time, confidence = predictions['pred_relevant_windows'][0]
        
        # Convert to absolute timestamps
        abs_start_time = start_time + clip_start_time
        abs_end_time = end_time + clip_start_time
        
        logging.info(f"Converting timestamps: relative ({start_time:.1f}s - {end_time:.1f}s) -> absolute ({abs_start_time:.1f}s - {abs_end_time:.1f}s)")
        
        # Break if confidence is below threshold
        if confidence < 0.75:
            logging.info(f"Confidence {confidence:.2f} below threshold 0.75, skipping clip")
            return []
        
        # Create moment dictionary
        moment = {
            "start_time": abs_start_time,
            "end_time": abs_end_time,
            "confidence": confidence
        }
        
        # Get clip end time from filename
        clip_filename = os.path.basename(clip_path)
        clip_end_time = float(clip_filename.split('_to_')[1].replace('s.mp4', ''))
        
        # Check for duplicates within this clip only
        if self._is_duplicate_moment(moment, existing_moments, clip_start_time, clip_end_time):
            logging.info(f"Skipping duplicate moment within clip: {abs_start_time:.1f}s - {abs_end_time:.1f}s")
            return []
                
        # Create moment filename and path
        moment_filename = f"video_{video_id}_moment_{len(existing_moments)+1:03d}_{abs_start_time:.1f}s_to_{abs_end_time:.1f}s_conf_{confidence:.2f}.mp4"
        moment_path = os.path.join(output_dir, moment_filename)
        
        # Extract moment video
        if self._extract_moment_video(clip_path, start_time, end_time, moment_path):
            moment.update({
                "filename": moment_filename,
                "path": moment_path
            })
            existing_moments.append(moment)
            return [moment]
        
        return []
        
    def process_clip(self, clip_path: str, query: str) -> List[Dict[str, float]]:
        """Process a clip to find moments matching the query.
        
        Args:
            clip_path: Path to the clip file
            query: Text query to search for moments
            
        Returns:
            List of dictionaries containing moment information
        """
        logging.info(f"Processing clip on {self.device}: {clip_path}")
        logging.info(f"Query: {query}")
        
        try:
            # Get video_id and clip_id from the input path
            clip_dir = os.path.dirname(clip_path)
            video_id = os.path.basename(clip_dir).replace('video_', '')
            clip_filename = os.path.basename(clip_path)
            clip_id = os.path.splitext(clip_filename)[0]  # e.g. video_<id>_clip_001_...
            clip_start_time = self._get_clip_timestamps(clip_filename)
            sanitized_query = query.replace(' ', '_').lower()
            logging.info(f"Sanitized query for folder name: {sanitized_query}")

            # Create output directory for this clip's moments (per-clip directory)
            output_dir = os.path.join(self.moments_dir, f"video_{video_id}", sanitized_query, clip_id)

            # Check if moments already exist for this clip
            if os.path.exists(output_dir):
                moments = []
                for file in os.listdir(output_dir):
                    if file.endswith('.mp4'):
                        parts = file.split('.mp4')[0].split('_')
                        try:
                            start_time = float(parts[4][:-1])
                            end_time = float(parts[6][:-1])
                            confidence = float(parts[8])
                            moments.append({
                                "filename": file,
                                "path": os.path.join(output_dir, file),
                                "start_time": start_time,
                                "end_time": end_time,
                                "confidence": confidence
                            })
                        except (IndexError, ValueError) as e:
                            logging.error(f"Error parsing moment file {file}: {str(e)}")
                            continue
                if moments:
                    logging.info(f"Found {len(moments)} existing moments for clip {clip_filename}")
                    return moments

            # If no moments exist, process the clip
            os.makedirs(output_dir, exist_ok=True)

            # Encode video features
            logging.info(f"Encoding video features on {self.device}...")
            video = self.model.encode_video(clip_path)

            # Get moment predictions
            logging.info(f"Getting moment predictions on {self.device}...")
            predictions = self.model.predict(query, video)

            # Process predictions
            moments = self._process_predictions(predictions, video_id, clip_path, output_dir, [], clip_start_time)

            if moments:
                logging.info(f"Found moment with confidence: {moments[0]['confidence']:.2f}")
            else:
                logging.info("No valid moments found")
            return moments

        except Exception as e:
            logging.error(f"Error processing clip: {str(e)}", exc_info=True)
            return []
