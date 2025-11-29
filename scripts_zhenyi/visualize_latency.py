#!/usr/bin/env python3

import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def load_latency_data(input_dir):
    """Load all latency JSON files from the input directory"""
    json_files = glob.glob(os.path.join(input_dir, "latency_analysis_*.json"))
    all_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['source_file'] = os.path.basename(json_file)
                all_data.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_data

def create_latency_plots(data_list, output_dir):
    """Create various latency analysis plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig_size = (15, 10)
    
    # Plot 1: Inter-frame latency over time
    plt.figure(figsize=fig_size)
    
    for i, data in enumerate(data_list):
        if 'inter_frame_latencies' in data and data['inter_frame_latencies']:
            frame_indices = [x['frame_idx'] for x in data['inter_frame_latencies']]
            latencies = [x['latency_ms'] for x in data['inter_frame_latencies']]
            
            plt.plot(frame_indices, latencies, 
                    label=f"Run {i+1} ({data.get('source_file', 'unknown')})",
                    alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Frame Index')
    plt.ylabel('Inter-frame Latency (ms)')
    plt.title('Inter-frame Latency Over Time (Steady-State Analysis)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inter_frame_latency_timeline.png'), dpi=300)
    plt.close()
    
    # Plot 2: Latency distribution histogram
    plt.figure(figsize=fig_size)
    
    all_latencies = []
    for data in data_list:
        if 'inter_frame_latencies' in data and data['inter_frame_latencies']:
            latencies = [x['latency_ms'] for x in data['inter_frame_latencies']]
            all_latencies.extend(latencies)
    
    if all_latencies:
        plt.hist(all_latencies, bins=50, alpha=0.7, edgecolor='black', density=True)
        plt.axvline(np.mean(all_latencies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_latencies):.2f} ms')
        plt.axvline(np.median(all_latencies), color='green', linestyle='--', 
                   label=f'Median: {np.median(all_latencies):.2f} ms')
        plt.axvline(np.percentile(all_latencies, 95), color='orange', linestyle='--', 
                   label=f'95th %ile: {np.percentile(all_latencies, 95):.2f} ms')
        
    plt.xlabel('Inter-frame Latency (ms)')
    plt.ylabel('Density')
    plt.title('Inter-frame Latency Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 3: Box plot comparison
    plt.figure(figsize=fig_size)
    
    latency_data_per_run = []
    labels = []
    
    for i, data in enumerate(data_list):
        if 'inter_frame_latencies' in data and data['inter_frame_latencies']:
            latencies = [x['latency_ms'] for x in data['inter_frame_latencies']]
            latency_data_per_run.append(latencies)
            labels.append(f"Run {i+1}")
    
    if latency_data_per_run:
        plt.boxplot(latency_data_per_run, labels=labels)
        plt.ylabel('Inter-frame Latency (ms)')
        plt.title('Inter-frame Latency Distribution by Run')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latency_boxplot.png'), dpi=300)
        plt.close()
    
    # Plot 4: Running statistics
    plt.figure(figsize=fig_size)
    
    for i, data in enumerate(data_list):
        if 'inter_frame_latencies' in data and data['inter_frame_latencies']:
            latencies = [x['latency_ms'] for x in data['inter_frame_latencies']]
            
            # Calculate running mean and std
            running_mean = []
            running_std = []
            
            for j in range(1, len(latencies) + 1):
                running_mean.append(np.mean(latencies[:j]))
                running_std.append(np.std(latencies[:j]))
            
            frame_indices = list(range(1, len(latencies) + 1))
            
            plt.subplot(2, 1, 1)
            plt.plot(frame_indices, running_mean, label=f"Run {i+1}", alpha=0.7)
            
            plt.subplot(2, 1, 2)
            plt.plot(frame_indices, running_std, label=f"Run {i+1}", alpha=0.7)
    
    plt.subplot(2, 1, 1)
    plt.ylabel('Running Mean (ms)')
    plt.title('Running Statistics of Inter-frame Latency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.xlabel('Frame Index')
    plt.ylabel('Running Std Dev (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'running_statistics.png'), dpi=300)
    plt.close()

def create_summary_report(data_list, output_dir):
    """Create a summary report of latency statistics"""
    
    report_lines = []
    report_lines.append("INTER-FRAME LATENCY ANALYSIS SUMMARY")
    report_lines.append("="*50)
    report_lines.append("")
    
    # Overall statistics across all runs
    all_latencies = []
    for data in data_list:
        if 'inter_frame_latencies' in data and data['inter_frame_latencies']:
            latencies = [x['latency_ms'] for x in data['inter_frame_latencies']]
            all_latencies.extend(latencies)
    
    if all_latencies:
        report_lines.append("OVERALL STATISTICS (All Runs Combined)")
        report_lines.append("-" * 30)
        report_lines.append(f"Total frames analyzed: {len(all_latencies)}")
        report_lines.append(f"Mean inter-frame latency: {np.mean(all_latencies):.2f} ms")
        report_lines.append(f"Median inter-frame latency: {np.median(all_latencies):.2f} ms")
        report_lines.append(f"Standard deviation: {np.std(all_latencies):.2f} ms")
        report_lines.append(f"Min latency: {np.min(all_latencies):.2f} ms")
        report_lines.append(f"Max latency: {np.max(all_latencies):.2f} ms")
        report_lines.append(f"95th percentile: {np.percentile(all_latencies, 95):.2f} ms")
        report_lines.append(f"99th percentile: {np.percentile(all_latencies, 99):.2f} ms")
        
        # Calculate target FPS metrics
        target_fps_16 = 1000 / 62.5  # 16 FPS = 62.5ms per frame
        target_fps_24 = 1000 / 41.67  # 24 FPS = 41.67ms per frame
        target_fps_30 = 1000 / 33.33  # 30 FPS = 33.33ms per frame
        
        frames_under_16fps = sum(1 for x in all_latencies if x <= 62.5)
        frames_under_24fps = sum(1 for x in all_latencies if x <= 41.67)
        frames_under_30fps = sum(1 for x in all_latencies if x <= 33.33)
        
        report_lines.append("")
        report_lines.append("TARGET FPS ANALYSIS")
        report_lines.append("-" * 20)
        report_lines.append(f"Frames meeting 16 FPS target: {frames_under_16fps}/{len(all_latencies)} ({100*frames_under_16fps/len(all_latencies):.1f}%)")
        report_lines.append(f"Frames meeting 24 FPS target: {frames_under_24fps}/{len(all_latencies)} ({100*frames_under_24fps/len(all_latencies):.1f}%)")
        report_lines.append(f"Frames meeting 30 FPS target: {frames_under_30fps}/{len(all_latencies)} ({100*frames_under_30fps/len(all_latencies):.1f}%)")
    
    report_lines.append("")
    report_lines.append("PER-RUN STATISTICS")
    report_lines.append("-" * 20)
    
    for i, data in enumerate(data_list):
        report_lines.append(f"\nRun {i+1}: {data.get('source_file', 'unknown')}")
        if 'statistics' in data:
            stats = data['statistics']
            report_lines.append(f"  Frames: {stats.get('num_frames', 0)}")
            report_lines.append(f"  Mean latency: {stats.get('mean_latency_ms', 0):.2f} ms")
            report_lines.append(f"  Median latency: {stats.get('median_latency_ms', 0):.2f} ms")
            report_lines.append(f"  95th percentile: {stats.get('p95_latency_ms', 0):.2f} ms")
            report_lines.append(f"  Average FPS: {stats.get('average_fps', 0):.2f}")
            report_lines.append(f"  Total time: {stats.get('total_generation_time_s', 0):.2f} s")
    
    # Save report
    report_path = os.path.join(output_dir, 'latency_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Also print to console
    print('\n'.join(report_lines))
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Visualize inter-frame latency analysis results')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Directory containing latency analysis JSON files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualization plots (default: same as input_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading latency data from: {args.input_dir}")
    data_list = load_latency_data(args.input_dir)
    
    if not data_list:
        print("No latency data files found!")
        return
    
    print(f"Found {len(data_list)} latency analysis files")
    
    print("Creating visualizations...")
    create_latency_plots(data_list, args.output_dir)
    
    print("Generating summary report...")
    report_path = create_summary_report(data_list, args.output_dir)
    
    print(f"\nAnalysis complete!")
    print(f"Visualizations saved to: {args.output_dir}")
    print(f"Summary report: {report_path}")
    
    print("\nGenerated files:")
    for filename in ['inter_frame_latency_timeline.png', 'latency_distribution.png', 
                     'latency_boxplot.png', 'running_statistics.png', 'latency_analysis_report.txt']:
        filepath = os.path.join(args.output_dir, filename)
        if os.path.exists(filepath):
            print(f"  - {filename}")

if __name__ == "__main__":
    main()