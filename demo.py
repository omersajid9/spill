"""
Gradio demo for YouTube video downloading and clipping using ClipsAI.
"""
import gradio as gr
import os
import logging
import nltk

from clipsai.downloader.videodownloader import VideoDownloader
from clipsai.clip.clipprocessor import ClipProcessor

from lighthouse.models import *

# Constants
TOPK_MOMENT = 40  # Maximum number of clip buttons to show

def create_demo(data_store_dir: str) -> gr.Blocks:
    """Create and return the Gradio interface."""
    # Initialize processors
    output_dir = os.path.join(data_store_dir, "yt_downloads")
    downloader = VideoDownloader(output_dir)
    clip_processor = ClipProcessor(data_store_dir)
    
    def process_video_for_clipping(video_path: str) -> tuple:
        """Process video and return clip information and buttons."""
        if not video_path:
            return "Please select a video first", *([gr.update(visible=False)] * TOPK_MOMENT)
        
        try:
            clips = clip_processor.process_video(video_path)
            
            # Create button updates
            button_updates = []
            for i in range(TOPK_MOMENT):
                if i < len(clips):
                    clip = clips[i]
                    button_text = f"moment {i+1}: [{clip['start_time']:.1f}s - {clip['end_time']:.1f}s]"
                    button_updates.append(gr.update(value=button_text, visible=True))
                else:
                    button_updates.append(gr.update(visible=False))
            
            return f"Found {len(clips)} clips", *button_updates
        except Exception as e:
            return f"Error processing video: {str(e)}", *([gr.update(visible=False)] * TOPK_MOMENT)
    
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
                
                download_btn.click(
                    fn=lambda url: downloader.download_video(url, "worst"),  # Always use worst quality
                    inputs=[url_input],
                    outputs=[video_output, download_status]
                )
                
                gr.Examples(
                    examples=[
                        ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
                    ],
                    inputs=[url_input]
                )
            
            # Right Column - Clip Selection
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
