#!/usr/bin/env python3

import cv2
import sys
import os

def count_frames(video_path):
    """Count the number of frames in a video file."""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    print(f"Video: {video_path}")
    print(f"Total frames: {frame_count}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    
    return frame_count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_video_frames.py <video_path>")
        print("Example: python count_video_frames.py /tmp/rank0-0-0_lora.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    count_frames(video_path)