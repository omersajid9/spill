"""
Gradio demo for YouTube video downloading and clipping using ClipsAI.
"""
import gradio as gr
from download import VideoDownloader
from clip import ClipProcessor
import os

def create_demo(data_store_dir: str) -> gr.Blocks:
    """Create and return the Gradio interface."""
    # Initialize processors
    output_dir = os.path.join(data_store_dir, "yt_downloads")
    downloader = VideoDownloader(output_dir)
    clip_processor = ClipProcessor(data_store_dir)
    
    def process_video_for_clipping(video_path: str) -> tuple:
        """Process video and return clip information."""
        if not video_path:
            return None, "Please select a video first"
        
        try:
            clips = clip_processor.process_video(video_path)
            # Format clip information for display
            clip_info = "\n".join([
                f"Clip {i+1}: {clip['filename']} ({clip['start_time']:.1f}s - {clip['end_time']:.1f}s)"
                for i, clip in enumerate(clips)
            ])
            return clip_info, f"Successfully processed {len(clips)} clips"
        except Exception as e:
            return None, f"Error processing video: {str(e)}"
    
    # Get available videos for dropdown
    available_videos = downloader.get_available_videos()
    video_choices = [(v["name"], v["path"]) for v in available_videos]
    
    # Create the interface using Blocks
    with gr.Blocks(title="ClipsAI Video Processing") as interface:
        gr.Markdown("# ClipsAI Video Processing")
        
        with gr.Tabs() as tabs:
            # Step 1: Video Download Interface
            with gr.Tab("Step 1: Download Video"):
                with gr.Row():
                    with gr.Column():
                        url_input = gr.Textbox(
                            label="YouTube URL",
                            placeholder="Enter YouTube video URL here..."
                        )
                        quality_radio = gr.Radio(
                            choices=["best", "worst"],
                            value="best",
                            label="Video Quality"
                        )
                        download_btn = gr.Button("Download Video")
                    
                    with gr.Column():
                        video_output = gr.Video(label="Downloaded Video")
                        download_status = gr.Textbox(label="Status")
                
                download_btn.click(
                    fn=downloader.download_video,
                    inputs=[url_input, quality_radio],
                    outputs=[video_output, download_status]
                )
                
                gr.Examples(
                    examples=[
                        ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "best"],
                        ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "worst"]
                    ],
                    inputs=[url_input, quality_radio]
                )
            
            # Step 2: Video Clipping Interface
            with gr.Tab("Step 2: Clip Video"):
                with gr.Row():
                    with gr.Column():
                        video_dropdown = gr.Dropdown(
                            choices=video_choices,
                            label="Select Video to Clip",
                            type="value"
                        )
                        clip_btn = gr.Button("Generate Clips")
                    
                    with gr.Column():
                        clip_output = gr.Textbox(
                            label="Generated Clips",
                            lines=10
                        )
                        clip_status = gr.Textbox(label="Status")
                
                clip_btn.click(
                    fn=process_video_for_clipping,
                    inputs=[video_dropdown],
                    outputs=[clip_output, clip_status]
                )
    
    return interface
