#!/usr/bin/env python3
"""
Real-Time Interactive Video Generation with Gradio

A much cleaner implementation using Gradio's built-in streaming capabilities.
Perfect for real-time video generation with dynamic prompt switching.
"""

import gradio as gr
import torch
import threading
import queue
import time
import numpy as np
import cv2
from typing import List, Optional, Generator
import os
import sys
from dataclasses import dataclass
import tempfile
from collections import deque

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from pipeline.realtime_streaming_inference import RealTimeStreamingPipeline


class GradioVideoStreamer:
    """
    Handles video streaming for Gradio interface
    Much simpler than Streamlit approach!
    """
    
    def __init__(self, config_path: str):
        self.pipeline = RealTimeStreamingPipeline(config_path)
        self.video_frames = deque(maxlen=160)  # ~10 seconds at 16fps
        # Increase buffered queue to absorb bursts and higher decode throughput
        self.display_queue = deque(maxlen=8192)  # buffered frames for paced UI consumption
        self.frame_lock = threading.Lock()
        self.last_display_time = 0.0
        self.last_display_frame = None
        # Refresh faster (~20 fps) to drain the queue more aggressively
        self.target_interval_s = (1.0 / 20.0)
        
        # Setup pipeline callback
        self.pipeline.add_frame_callback(self._frame_callback)
        
        print("[Gradio] Video streamer initialized with RealTimeStreamingPipeline!")
        
    def _frame_callback(self, frame: np.ndarray, frame_idx: int):
        """Callback to receive frames from pipeline"""
        with self.frame_lock:
            # Store frame with timestamp
            self.video_frames.append({
                'frame': frame.copy(),
                'frame_idx': frame_idx,
                'timestamp': time.time()
            })
            # Enqueue for paced display
            self.display_queue.append(frame.copy())
    
    def get_display_queue_size(self) -> int:
        """Return current number of frames waiting to be displayed"""
        with self.frame_lock:
            return len(self.display_queue)
    
    def start_generation(self, prompt: str):
        """Start video generation with initial prompt"""
        if not prompt.strip():
            return "❌ Please enter a prompt first!"
            
        try:
            self.pipeline.start_generation(prompt.strip())
            return f"✅ Started generating: '{prompt}'"
        except Exception as e:
            return f"❌ Error starting generation: {e}"
    
    def send_prompt(self, prompt: str):
        """Send new prompt during generation"""
        if not prompt.strip():
            return "❌ Please enter a prompt!"
            
        if not self.pipeline.is_running:
            return "❌ No video currently generating. Start a new video first!"
            
        try:
            changed = self.pipeline.send_prompt(prompt.strip())
            if changed:
                return f"✅ Switched to: '{prompt}'"
            else:
                return f"💭 Same as current prompt"
        except Exception as e:
            return f"❌ Error sending prompt: {e}"
    
    def new_video(self, prompt: str):
        """Start a new video after previous one finished"""
        if not prompt.strip():
            return "❌ Please enter a prompt!"
            
        try:
            self.pipeline.reset_for_new_video()
            self.pipeline.start_generation(prompt.strip())
            # Clear frame buffer
            with self.frame_lock:
                self.video_frames.clear()
                self.display_queue.clear()
                self.last_display_time = 0.0
                self.last_display_frame = None
            return f"✅ New video started: '{prompt}'"
        except Exception as e:
            return f"❌ Error starting new video: {e}"
    
    def get_status(self):
        """Get current generation status"""
        status = self.pipeline.get_status()
        
        if not status.is_running and not status.start_time:
            return "⏳ Ready to start"
        elif status.is_running:
            # Derive remaining seconds from measured fps to avoid config assumptions
            remaining_frames = max(0, self.pipeline.total_frames - status.current_frame)
            measured_fps = max(1e-6, status.fps)  # frames per second
            secs_left = int(remaining_frames / measured_fps)
            return f"🎬 Generating • Frame {status.current_frame}/{self.pipeline.total_frames} • {secs_left}s left • {measured_fps:.1f} fps"
        elif self.pipeline.is_finished():
            return "✅ Video completed! Ready for new video"
        else:
            return "⏹️ Stopped"
    
    def get_latest_frame_for_display(self):
        """Get latest frame for Gradio display"""
        with self.frame_lock:
            if not self.video_frames:
                return None
            
            # Return the most recent frame
            latest = self.video_frames[-1]
            frame = latest['frame']
            
            # Convert to RGB for proper display (frames come as RGB already)
            return frame
    
    def get_next_frame_for_interval(self, interval_s: float = None):
        """Return next frame at fixed interval; otherwise keep last displayed frame."""
        if interval_s is None:
            interval_s = self.target_interval_s
        now = time.time()
        with self.frame_lock:
            should_emit = (self.last_display_time == 0.0) or ((now - self.last_display_time) >= interval_s)
            if should_emit and self.display_queue:
                self.last_display_frame = self.display_queue.popleft()
                self.last_display_time = now
            return self.last_display_frame
    
    def stop_generation(self):
        """Stop current generation"""
        try:
            self.pipeline.stop_generation()
            return "⏹️ Generation stopped"
        except Exception as e:
            return f"❌ Error stopping: {e}"


# Global streamer instance
streamer = None


def initialize_app():
    """Initialize the application"""
    global streamer
    try:
        config_path = "configs/longlive_interactive_inference.yaml"
        print(f"[Gradio] Initializing with config: {config_path}")
        streamer = GradioVideoStreamer(config_path)
        print("[Gradio] Streamer created successfully!")
        return "✅ App initialized! Ready to generate videos."
    except Exception as e:
        print(f"[Gradio] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Initialization failed: {e}"




def update_display():
    """Update video display (called periodically)"""
    if streamer is None:
        return None, "❌ Not initialized"
    
    try:
        frame = streamer.get_next_frame_for_interval(1.0 / 20.0)
        status = streamer.get_status()
        queue_size = streamer.get_display_queue_size()
        
        # Debug info (only print occasionally to avoid spam)
        if hasattr(update_display, '_call_count'):
            update_display._call_count += 1
        else:
            update_display._call_count = 1
            
        if update_display._call_count % 10 == 0:  # Print every 10th call (every 5 seconds)
            frame_info = f"Frame: {frame is not None}" if frame is not None else "No frame"
            print(f"[Gradio] Update #{update_display._call_count}: {frame_info}, Queue={queue_size}, Status: {status}")
        
        status_with_queue = f"{status} • Queue: {queue_size}"
        return frame, status_with_queue
        
    except Exception as e:
        print(f"[Gradio] Error in update_display: {e}")
        return None, f"❌ Update error: {e}"


def smart_submit(prompt):
    """Smart submit - starts new video or switches prompt based on current state"""
    if streamer is None:
        return "❌ App not initialized", None, "❌ Not initialized"
    
    if not prompt.strip():
        return "❌ Please enter a prompt!", None, streamer.get_status()
    
    try:
        # If not running or finished, start new video
        if not streamer.pipeline.is_running or streamer.pipeline.is_finished():
            result = streamer.new_video(prompt)
        else:
            # If running, switch prompt
            result = streamer.send_prompt(prompt)
        
        status = streamer.get_status()
        frame = streamer.get_latest_frame_for_display()
        return result, frame, status
    except Exception as e:
        return f"❌ Error: {e}", None, "❌ Error"


def create_interface():
    """Create simplified Gradio interface"""
    
    with gr.Blocks(title="Real-Time Video Generation") as app:
        
        # Auto-initialize on page load
        app.load(initialize_app)
        
        # Video display
        video_frame = gr.Image(
            label="Generated Video",
            type="numpy"
        )
        
        # Text input
        prompt_input = gr.Textbox(
            label="Enter Your Prompt",
            placeholder="A majestic eagle soaring over snow-capped mountains",
            lines=2
        )
        
        # Submit button
        submit_btn = gr.Button("Submit", variant="primary")
        
        # Status (hidden from user but needed for updates)
        status_display = gr.Textbox(
            label="Status",
            visible=False
        )
        
        # Event handlers
        submit_btn.click(
            smart_submit,
            inputs=[prompt_input],
            outputs=[status_display, video_frame, status_display]
        )
        
        # Auto-refresh timer for video frames
        refresh_timer = gr.Timer(1.0 / 20.0)
        refresh_timer.tick(
            update_display,
            outputs=[video_frame, status_display]
        )
    
    return app


if __name__ == "__main__":
    # Create and launch the interface
    app = create_interface()
    
    print("🎬 Starting Real-Time Interactive Video Generation App with Gradio...")
    print("📍 The app will be available at the URL shown below")
    print("🔗 Use SSH tunnel if running remotely: ssh -L 7860:localhost:7860 user@host")
    
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Standard Gradio port
        share=False             # Set to True for public sharing
    )