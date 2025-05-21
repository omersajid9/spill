"""
Gradio demo for YouTube video downloading and clipping using ClipsAI.
"""
import gradio as gr
import os
import logging
import nltk
import torch
import gc

from clipsai.downloader.videodownloader import VideoDownloader
from clipsai.clip.clipprocessor import ClipProcessor
from lighthouse.momentprocessor import MomentProcessor

from lighthouse.models import *

# Constants
TOPK_MOMENT = 100  # Maximum number of clip buttons to show
TOPK_MOMENT_RETRIEVAL = 40  # Maximum number of moment buttons to show

def clear_gpu_memory():
    """Clear GPU memory by emptying cache and running garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def create_demo(data_store_dir: str) -> gr.Blocks:
    """Create and return the Gradio interface."""
    # Initialize processors
    output_dir = os.path.join(data_store_dir, "yt_downloads")
    downloader = VideoDownloader(output_dir)
    clip_processor = ClipProcessor(data_store_dir)
    moment_processor = MomentProcessor(data_store_dir)
    
    def process_video_for_clipping(video_path: str) -> tuple:
        """Process video and return clip information and buttons."""
        if not video_path:
            return "Please select a video first", *([gr.update(visible=False)] * TOPK_MOMENT)
        
        try:
            # Clear GPU memory before processing
            clear_gpu_memory()
            
            clips = clip_processor.process_video(video_path)
            
            # Create button updates
            button_updates = []
            for i in range(TOPK_MOMENT):
                if i < len(clips):
                    clip = clips[i]
                    button_text = f"clip {i+1}: [{clip['start_time']:.1f}s - {clip['end_time']:.1f}s]"
                    button_updates.append(gr.update(value=button_text, visible=True))
                else:
                    button_updates.append(gr.update(visible=False))
            
            return f"Found {len(clips)} clips", *button_updates
        except Exception as e:
            return f"Error processing video: {str(e)}", *([gr.update(visible=False)] * TOPK_MOMENT)
    
    def process_clip_for_moments(video_path: str, query: str) -> tuple:
        """Process clip and return moment information and buttons."""
        if not video_path or not query:
            return "Please select a video and enter a query first", *([gr.update(visible=False)] * TOPK_MOMENT_RETRIEVAL)
        
        try:
            # Clear GPU memory before processing
            clear_gpu_memory()
            
            # Get video ID from the video path
            video_id = os.path.splitext(os.path.basename(video_path))[0].replace('video_', '')
            logging.info(f"CLIP PATH: {video_path}")
            logging.info(f"Processing video ID: {video_id}")
            
            # Look for clips in yt_clipped directory
            clips_dir = os.path.join(data_store_dir, "yt_clipped", f"video_{video_id}")
            logging.info(f"Looking for clips in directory: {clips_dir}")
            
            # Get all clips for this video
            clip_files = [f for f in os.listdir(clips_dir) if f.endswith('.mp4')]
            logging.info(f"Found {len(clip_files)} clip files: {clip_files}")
            
            if not clip_files:
                logging.error(f"No clips found in {clips_dir}")
                return "No clips found for this video", *([gr.update(visible=False)] * TOPK_MOMENT_RETRIEVAL)
            
            # Process each clip
            all_moments = []
            for clip_file in clip_files:
                full_clip_path = os.path.join(clips_dir, clip_file)
                logging.info(f"Processing clip: {clip_file}")
                moments = moment_processor.process_clip(full_clip_path, query)
                logging.info(f"Found {len(moments)} moments in clip {clip_file}")
                all_moments.extend(moments)
                
                # Clear GPU memory after processing each clip
                clear_gpu_memory()
            
            logging.info(f"Total moments found across all clips: {len(all_moments)}")
            
            # Create button updates
            button_updates = []
            for i in range(TOPK_MOMENT_RETRIEVAL):
                if i < len(all_moments):
                    moment = all_moments[i]
                    button_text = f"moment {i+1}: [{moment['start_time']:.1f}s - {moment['end_time']:.1f}s] (conf: {moment['confidence']:.2f})"
                    button_updates.append(gr.update(value=button_text, visible=True))
                else:
                    button_updates.append(gr.update(visible=False))
            
            return f"Found {len(all_moments)} moments", *button_updates
        except Exception as e:
            return f"Error processing clip for moments: {str(e)}", *([gr.update(visible=False)] * TOPK_MOMENT_RETRIEVAL)
    
    # Get available videos for dropdown
    available_videos = downloader.get_available_videos()
    video_choices = [(v["name"], v["path"]) for v in available_videos]
    
    # Create JavaScript code for clip buttons
    js_codes = ["""() => {{
            let moment_text = document.getElementById('result_{}').textContent;

            // Extract the time range part
            let timeRange = moment_text.split(':')[1].trim();
            // Remove [ and ] and split by -
            let times = timeRange.slice(1, -1).split('-');
            // Get start time (remove 's' and convert to float)
            let startTime = parseFloat(times[0].trim().replace('s', ''));
            
            let video = document.getElementsByTagName("video")[0];
            if (video) {{
                video.currentTime = startTime;
                video.play();
            }} else {{
                console.log('Video element not found');
            }}
        }}""".format(i, i) for i in range(TOPK_MOMENT)]
    
    # Create JavaScript code for moment buttons
    moment_js_codes = ["""() => {{
            let moment_text = document.getElementById('moment_result_{}').textContent;

            // Extract the time range part
            let timeRange = moment_text.split(':')[1].trim();
            // Remove [ and ] and split by -
            let times = timeRange.slice(1, -1).split('-');
            // Get start time (remove 's' and convert to float)
            let startTime = parseFloat(times[0].trim().replace('s', ''));
            
            let video = document.getElementsByTagName("video")[0];
            if (video) {{
                video.currentTime = startTime;
                video.play();
            }} else {{
                console.log('Video element not found');
            }}
        }}""".format(i, i) for i in range(TOPK_MOMENT_RETRIEVAL)]
    
    # Create the interface using Blocks
    with gr.Blocks(title="ClipsAI Video Processing") as interface:
        gr.Markdown("# ClipsAI Video Processing")
        
        with gr.Row():
            # Left Column - Video Download and Playback
            with gr.Column():
                with gr.Group():
                    url_input = gr.Textbox(
                        label="YouTube URL",
                        placeholder="Enter YouTube video URL here..."
                    )
                    download_btn = gr.Button("Download Video")
                    video_output = gr.Video(label="Video Player")
                    download_status = gr.Textbox(label="Status")
                
                def download_and_clear(url):
                    # Clear GPU memory before downloading
                    clear_gpu_memory()
                    return downloader.download_video(url, "worst")
                
                download_btn.click(
                    fn=download_and_clear,
                    inputs=[url_input],
                    outputs=[video_output, download_status]
                )
                
                gr.Examples(
                    examples=[
                        ["https://www.youtube.com/watch?v=ZCcm1oTf3Cg"]
                    ],
                    inputs=[url_input]
                )
            
            # Middle Column - Clip Selection
            with gr.Column():
                gr.Markdown("## Generated Clips")
                clip_btn = gr.Button("Generate Clips")
                clip_status = gr.Textbox(label="Status")
                
                # Create clip buttons
                with gr.Group():
                    gr.Markdown("## Retrieved Clips")
                    clip_buttons = []
                    for i in range(TOPK_MOMENT):
                        btn = gr.Button(
                            value=f'clip {i+1}',
                            visible=False,
                            elem_id=f'result_{i}'
                        )
                        clip_buttons.append(btn)
                
                # Set up clip button click handlers
                for i, btn in enumerate(clip_buttons):
                    btn.click(None, None, None, js=js_codes[i])
                
                clip_btn.click(
                    fn=process_video_for_clipping,
                    inputs=[video_output],
                    outputs=[clip_status] + clip_buttons
                )
            
            # Right Column - Moment Retrieval
            with gr.Column():
                gr.Markdown("## Moment Retrieval")
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="Enter your query here..."
                )
                moment_btn = gr.Button("Find Moments")
                moment_status = gr.Textbox(label="Status")
                
                # Create moment buttons
                with gr.Group():
                    gr.Markdown("## Retrieved Moments")
                    moment_buttons = []
                    for i in range(TOPK_MOMENT_RETRIEVAL):
                        btn = gr.Button(
                            value=f'moment {i+1}',
                            visible=False,
                            elem_id=f'moment_result_{i}'
                        )
                        moment_buttons.append(btn)
                
                # Set up moment button click handlers
                for i, btn in enumerate(moment_buttons):
                    btn.click(None, None, None, js=moment_js_codes[i])
                
                moment_btn.click(
                    fn=process_clip_for_moments,
                    inputs=[video_output, query_input],  # Pass video path instead of clip button
                    outputs=[moment_status] + moment_buttons
                )
    
    return interface

def initialize():
    """Initialize required resources."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Download NLTK data if needed (silently)
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)

if __name__ == "__main__":
    data_store_dir = "/root/spill/data"
    initialize()
    demo = create_demo(data_store_dir)
    
    # Get server name from environment or use default
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    
    # Launch with proper configuration for sharing
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,  # Explicitly set port
        share=True,  # Enable sharing
        show_error=True,  # Show detailed errors
        show_api=False,  # Hide API docs
        quiet=False,  # Show startup messages
        allowed_paths=[data_store_dir]  # Allow access to our download directory
    )
