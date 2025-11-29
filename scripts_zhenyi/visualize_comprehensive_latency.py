#!/usr/bin/env python3

import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from collections import defaultdict

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_comprehensive_data(input_dir):
    """Load all comprehensive analysis JSON files"""
    json_files = glob.glob(os.path.join(input_dir, "comprehensive_analysis_*.json"))
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

def create_component_timing_plots(data_list, output_dir):
    """Create component-level timing plots"""
    
    # Collect all component data
    component_data = defaultdict(lambda: defaultdict(list))
    
    for data in data_list:
        comp_stats = data.get('comprehensive_statistics', {}).get('component_breakdown', {})
        for comp_name, comp_timing in comp_stats.items():
            if 'gpu_timing' in comp_timing:
                gpu_timing = comp_timing['gpu_timing']
                component_data[comp_name]['mean'].append(gpu_timing.get('mean_ms', 0))
                component_data[comp_name]['p95'].append(gpu_timing.get('p95_ms', 0))
                component_data[comp_name]['max'].append(gpu_timing.get('max_ms', 0))
                component_data[comp_name]['total'].append(gpu_timing.get('total_ms', 0))
                component_data[comp_name]['count'].append(comp_timing.get('count', 0))

    if not component_data:
        print("No component data found")
        return

    # 1. Component timing comparison (mean latencies)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    components = list(component_data.keys())
    means = [np.mean(component_data[comp]['mean']) for comp in components]
    p95s = [np.mean(component_data[comp]['p95']) for comp in components]
    
    # Sort by mean latency
    sorted_indices = np.argsort(means)[::-1]
    components_sorted = [components[i] for i in sorted_indices]
    means_sorted = [means[i] for i in sorted_indices]
    p95s_sorted = [p95s[i] for i in sorted_indices]
    
    x_pos = np.arange(len(components_sorted))
    
    # Mean latencies
    bars1 = ax1.bar(x_pos, means_sorted, alpha=0.7, label='Mean Latency')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Component-Level Mean Latency Breakdown')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(components_sorted, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms',
                ha='center', va='bottom', fontsize=8)
    
    # P95 latencies
    bars2 = ax2.bar(x_pos, p95s_sorted, alpha=0.7, color='orange', label='P95 Latency')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Component-Level P95 Latency Breakdown')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(components_sorted, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_latency_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_denoising_step_analysis(data_list, output_dir):
    """Analyze denoising step timing patterns"""
    
    denoising_data = []
    
    for data in data_list:
        component_timings = data.get('component_timings', {})
        
        # Collect denoising step data
        for step_type in ['denoising_step', 'denoising_step_final']:
            if step_type in component_timings:
                for timing in component_timings[step_type]:
                    denoising_data.append({
                        'step_type': step_type,
                        'timestep': timing.get('timestep', 0),
                        'step_index': timing.get('step_index', 0),
                        'block_idx': timing.get('block_idx', 0),
                        'gpu_time_ms': timing.get('gpu_time_ms', 0),
                        'is_final': timing.get('is_final_step', False)
                    })
    
    if not denoising_data:
        print("No denoising step data found")
        return
    
    df = pd.DataFrame(denoising_data)
    
    # Plot 1: Denoising timing by timestep
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Timing by timestep
    timestep_groups = df.groupby('timestep')['gpu_time_ms']
    timesteps = list(timestep_groups.groups.keys())
    timestep_means = [timestep_groups.get_group(ts).mean() for ts in timesteps]
    timestep_stds = [timestep_groups.get_group(ts).std() for ts in timesteps]
    
    ax1.bar(range(len(timesteps)), timestep_means, yerr=timestep_stds, alpha=0.7, capsize=5)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('GPU Time (ms)')
    ax1.set_title('Denoising Time by Timestep')
    ax1.set_xticks(range(len(timesteps)))
    ax1.set_xticklabels([f'{int(ts)}' for ts in timesteps])
    ax1.grid(True, alpha=0.3)
    
    # Timing by step index
    step_groups = df.groupby('step_index')['gpu_time_ms']
    step_indices = list(step_groups.groups.keys())
    step_means = [step_groups.get_group(si).mean() for si in step_indices]
    step_stds = [step_groups.get_group(si).std() for si in step_indices]
    
    ax2.bar(step_indices, step_means, yerr=step_stds, alpha=0.7, capsize=5)
    ax2.set_xlabel('Denoising Step Index')
    ax2.set_ylabel('GPU Time (ms)')
    ax2.set_title('Denoising Time by Step Index')
    ax2.grid(True, alpha=0.3)
    
    # Timing by block (cross-block analysis)
    if 'block_idx' in df.columns:
        block_groups = df.groupby('block_idx')['gpu_time_ms']
        block_indices = list(block_groups.groups.keys())
        block_means = [block_groups.get_group(bi).mean() for bi in block_indices[:10]]  # First 10 blocks
        
        ax3.plot(block_indices[:10], block_means, 'o-', alpha=0.7)
        ax3.set_xlabel('Block Index')
        ax3.set_ylabel('Mean GPU Time (ms)')
        ax3.set_title('Denoising Time by Block (First 10 blocks)')
        ax3.grid(True, alpha=0.3)
    
    # Final vs intermediate steps
    final_times = df[df['is_final'] == True]['gpu_time_ms']
    intermediate_times = df[df['is_final'] == False]['gpu_time_ms']
    
    if len(final_times) > 0 and len(intermediate_times) > 0:
        ax4.hist(intermediate_times, bins=20, alpha=0.7, label='Intermediate Steps', density=True)
        ax4.hist(final_times, bins=20, alpha=0.7, label='Final Steps', density=True)
        ax4.set_xlabel('GPU Time (ms)')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution: Final vs Intermediate Steps')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'denoising_step_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_kv_operations_analysis(data_list, output_dir):
    """Analyze KV cache operations timing"""
    
    kv_data = []
    
    for data in data_list:
        component_timings = data.get('component_timings', {})
        
        if 'kv_operations' in component_timings:
            for timing in component_timings['kv_operations']:
                kv_data.append({
                    'operation': timing.get('operation', 'unknown'),
                    'block_idx': timing.get('block_idx', -1),
                    'gpu_time_ms': timing.get('gpu_time_ms', 0),
                    'cache_size': timing.get('cache_size', 0),
                    'policy': timing.get('policy', 'unknown')
                })
    
    if not kv_data:
        print("No KV operations data found")
        return
    
    df = pd.DataFrame(kv_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # KV operations by type
    op_groups = df.groupby('operation')['gpu_time_ms']
    operations = list(op_groups.groups.keys())
    op_means = [op_groups.get_group(op).mean() for op in operations]
    op_stds = [op_groups.get_group(op).std() for op in operations]
    
    bars = ax1.bar(operations, op_means, yerr=op_stds, alpha=0.7, capsize=5)
    ax1.set_ylabel('GPU Time (ms)')
    ax1.set_title('KV Operations Timing by Type')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars, op_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.2f}ms',
                ha='center', va='bottom', fontsize=9)
    
    # KV cache update timing across blocks
    kv_update_data = df[df['operation'] == 'kv_cache_update']
    if len(kv_update_data) > 0:
        block_groups = kv_update_data.groupby('block_idx')['gpu_time_ms']
        block_indices = sorted(list(block_groups.groups.keys()))[:15]  # First 15 blocks
        block_means = [block_groups.get_group(bi).mean() for bi in block_indices]
        
        ax2.plot(block_indices, block_means, 'o-', alpha=0.7, linewidth=2, markersize=6)
        ax2.set_xlabel('Block Index')
        ax2.set_ylabel('KV Cache Update Time (ms)')
        ax2.set_title('KV Cache Update Timing Across Blocks')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kv_operations_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_device_sync_analysis(data_list, output_dir):
    """Analyze device synchronization overhead"""
    
    sync_data = []
    
    for data in data_list:
        component_timings = data.get('component_timings', {})
        
        if 'device_sync' in component_timings:
            for timing in component_timings['device_sync']:
                sync_data.append({
                    'sync_type': timing.get('sync_type', 'unknown'),
                    'gpu_time_ms': timing.get('gpu_time_ms', 0),
                    'frame_idx': timing.get('frame_idx', 0),
                    'block_idx': timing.get('block_idx', 0)
                })
    
    if not sync_data:
        print("No device sync data found")
        return
    
    df = pd.DataFrame(sync_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sync timing by type
    sync_groups = df.groupby('sync_type')['gpu_time_ms']
    sync_types = list(sync_groups.groups.keys())
    sync_means = [sync_groups.get_group(st).mean() for st in sync_types]
    sync_counts = [len(sync_groups.get_group(st)) for st in sync_types]
    
    bars = ax1.bar(sync_types, sync_means, alpha=0.7)
    ax1.set_ylabel('GPU Time (ms)')
    ax1.set_title('Device Sync Timing by Type')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars, sync_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}',
                ha='center', va='bottom', fontsize=8)
    
    # Total sync overhead across frames
    frame_groups = df.groupby('frame_idx')['gpu_time_ms']
    frame_indices = sorted(list(frame_groups.groups.keys()))[:50]  # First 50 frames
    frame_totals = [frame_groups.get_group(fi).sum() for fi in frame_indices]
    
    ax2.plot(frame_indices, frame_totals, alpha=0.7, linewidth=2)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Total Sync Overhead (ms)')
    ax2.set_title('Device Sync Overhead per Frame')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'device_sync_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_cross_block_detailed_analysis(data_list, output_dir):
    """Create detailed cross-block vs intra-block analysis"""
    
    cross_block_data = []
    
    for data in data_list:
        stats = data.get('comprehensive_statistics', {})
        cross_vs_intra = stats.get('cross_vs_intra_block', {})
        
        if cross_vs_intra:
            cross_block_data.append(cross_vs_intra)
    
    if not cross_block_data:
        print("No cross-block analysis data found")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect metrics
    cross_means = []
    intra_means = []
    latency_multipliers = []
    additional_latencies = []
    
    for data in cross_block_data:
        if 'cross_block_worst_case' in data and 'intra_block_best_case' in data:
            cross_means.append(data['cross_block_worst_case']['mean_ms'])
            intra_means.append(data['intra_block_best_case']['mean_ms'])
            
        if 'comparison' in data:
            latency_multipliers.append(data['comparison']['latency_multiplier'])
            additional_latencies.append(data['comparison']['additional_latency_ms'])
    
    # Cross-block vs intra-block comparison
    if cross_means and intra_means:
        x_pos = np.arange(len(cross_means))
        width = 0.35
        
        ax1.bar(x_pos - width/2, cross_means, width, label='Cross-block (worst case)', alpha=0.7)
        ax1.bar(x_pos + width/2, intra_means, width, label='Intra-block (best case)', alpha=0.7)
        ax1.set_ylabel('Mean Latency (ms)')
        ax1.set_title('Cross-block vs Intra-block Latency')
        ax1.set_xlabel('Run Index')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Latency multiplier distribution
    if latency_multipliers:
        ax2.hist(latency_multipliers, bins=10, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(latency_multipliers), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(latency_multipliers):.1f}x')
        ax2.set_xlabel('Latency Multiplier (Cross-block / Intra-block)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Cross-block Latency Multiplier')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Additional latency
    if additional_latencies:
        ax3.hist(additional_latencies, bins=10, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(additional_latencies), color='red', linestyle='--',
                   label=f'Mean: {np.mean(additional_latencies):.1f}ms')
        ax3.set_xlabel('Additional Latency (ms)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Additional Cross-block Latency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Summary statistics
    if cross_means and intra_means and latency_multipliers:
        summary_stats = [
            ['Cross-block Mean', f'{np.mean(cross_means):.2f} ms'],
            ['Intra-block Mean', f'{np.mean(intra_means):.2f} ms'],
            ['Latency Multiplier', f'{np.mean(latency_multipliers):.1f}x'],
            ['Additional Latency', f'{np.mean(additional_latencies):.2f} ms']
        ]
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=summary_stats,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax4.set_title('Cross-block vs Intra-block Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_block_detailed_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_report(data_list, output_dir):
    """Create a comprehensive text report"""
    
    report_lines = []
    report_lines.append("COMPREHENSIVE LATENCY ANALYSIS REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    
    # Overall summary
    if data_list:
        sample_stats = data_list[0].get('comprehensive_statistics', {})
        config = sample_stats.get('configuration', {})
        
        report_lines.append("CONFIGURATION")
        report_lines.append("-" * 20)
        report_lines.append(f"Frames per block: {config.get('num_frame_per_block', 'N/A')}")
        report_lines.append(f"Total frames: {config.get('total_frames', 'N/A')}")
        report_lines.append(f"Total blocks: {config.get('total_blocks', 'N/A')}")
        report_lines.append("")
    
    # Component breakdown
    component_summary = defaultdict(lambda: {'means': [], 'totals': []})
    
    for data in data_list:
        comp_breakdown = data.get('comprehensive_statistics', {}).get('component_breakdown', {})
        for comp_name, comp_data in comp_breakdown.items():
            if 'gpu_timing' in comp_data:
                gpu_timing = comp_data['gpu_timing']
                component_summary[comp_name]['means'].append(gpu_timing.get('mean_ms', 0))
                component_summary[comp_name]['totals'].append(gpu_timing.get('total_ms', 0))
    
    report_lines.append("COMPONENT TIMING SUMMARY")
    report_lines.append("-" * 30)
    
    for comp_name, comp_data in component_summary.items():
        if comp_data['means']:
            mean_avg = np.mean(comp_data['means'])
            total_avg = np.mean(comp_data['totals'])
            report_lines.append(f"{comp_name.replace('_', ' ').title()}:")
            report_lines.append(f"  Average mean latency: {mean_avg:.2f} ms")
            report_lines.append(f"  Average total time: {total_avg:.2f} ms")
    
    report_lines.append("")
    
    # Cross-block analysis
    cross_block_summary = []
    for data in data_list:
        cross_vs_intra = data.get('comprehensive_statistics', {}).get('cross_vs_intra_block', {})
        if 'comparison' in cross_vs_intra:
            comp = cross_vs_intra['comparison']
            cross_block_summary.append(comp)
    
    if cross_block_summary:
        report_lines.append("WORST CASE (CROSS-BLOCK) ANALYSIS")
        report_lines.append("-" * 40)
        avg_multiplier = np.mean([c['latency_multiplier'] for c in cross_block_summary])
        avg_additional = np.mean([c['additional_latency_ms'] for c in cross_block_summary])
        avg_percentage = np.mean([c['cross_block_percentage'] for c in cross_block_summary])
        
        report_lines.append(f"Average latency multiplier: {avg_multiplier:.1f}x")
        report_lines.append(f"Average additional latency: {avg_additional:.2f} ms")
        report_lines.append(f"Cross-block percentage: {avg_percentage:.1f}%")
        report_lines.append("")
        
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("- Cross-block transitions are the primary latency bottleneck")
        report_lines.append("- Consider optimizing KV cache updates")
        report_lines.append("- Look into reducing context processing overhead")
        report_lines.append("- Frame batching optimization may help")
    
    # Save report
    report_path = os.path.join(output_dir, 'comprehensive_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Visualize comprehensive latency analysis results')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing comprehensive analysis JSON files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualization plots (default: same as input_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading comprehensive analysis data from: {args.input_dir}")
    data_list = load_comprehensive_data(args.input_dir)
    
    if not data_list:
        print("No comprehensive analysis data files found!")
        return
    
    print(f"Found {len(data_list)} comprehensive analysis files")
    
    print("Creating visualizations...")
    create_component_timing_plots(data_list, args.output_dir)
    create_denoising_step_analysis(data_list, args.output_dir)
    create_kv_operations_analysis(data_list, args.output_dir)
    create_device_sync_analysis(data_list, args.output_dir)
    create_cross_block_detailed_analysis(data_list, args.output_dir)
    
    print("Generating comprehensive report...")
    report_path = create_comprehensive_report(data_list, args.output_dir)
    
    print(f"\nComprehensive analysis visualization complete!")
    print(f"Visualizations saved to: {args.output_dir}")
    print(f"Report saved to: {report_path}")
    
    print("\nGenerated files:")
    generated_files = [
        'component_latency_breakdown.png',
        'denoising_step_analysis.png', 
        'kv_operations_analysis.png',
        'device_sync_analysis.png',
        'cross_block_detailed_analysis.png',
        'comprehensive_analysis_report.txt'
    ]
    
    for filename in generated_files:
        filepath = os.path.join(args.output_dir, filename)
        if os.path.exists(filepath):
            print(f"  - {filename}")

if __name__ == "__main__":
    main()