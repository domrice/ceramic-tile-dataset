"""
Script to analyze YOLO training results from results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_results(csv_path):
    """
    Load results CSV file
    
    Args:
        csv_path (str): Path to results.csv file
        
    Returns:
        pd.DataFrame: Loaded results dataframe
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Results file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} epochs from {csv_path}")
    return df


def analyze_training_time(df):
    """
    Analyze training time metrics
    
    Args:
        df (pd.DataFrame): Results dataframe
        
    Returns:
        dict: Training time statistics
    """
    print("\n" + "="*50)
    print("TRAINING TIME ANALYSIS")
    print("="*50)
    
    total_time = df['time'].iloc[-1]  # Last epoch time is cumulative
    num_epochs = len(df)
    
    # Calculate time per epoch
    df['time_per_epoch'] = df['time'].diff()
    df.loc[0, 'time_per_epoch'] = df['time'].iloc[0]  # First epoch
    
    avg_time_per_epoch = df['time_per_epoch'].mean()
    min_time_per_epoch = df['time_per_epoch'].min()
    max_time_per_epoch = df['time_per_epoch'].max()
    
    # Convert to readable format
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    print(f"Total Training Time: {format_time(total_time)} ({total_time:.1f} seconds)")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Average Time per Epoch: {format_time(avg_time_per_epoch)} ({avg_time_per_epoch:.1f} seconds)")
    print(f"Fastest Epoch: {format_time(min_time_per_epoch)} ({min_time_per_epoch:.1f} seconds)")
    print(f"Slowest Epoch: {format_time(max_time_per_epoch)} ({max_time_per_epoch:.1f} seconds)")
    
    return {
        'total_time': total_time,
        'num_epochs': num_epochs,
        'avg_time_per_epoch': avg_time_per_epoch,
        'min_time_per_epoch': min_time_per_epoch,
        'max_time_per_epoch': max_time_per_epoch
    }


def analyze_performance_metrics(df):
    """
    Analyze model performance metrics
    
    Args:
        df (pd.DataFrame): Results dataframe
        
    Returns:
        dict: Performance statistics
    """
    print("\n" + "="*50)
    print("PERFORMANCE METRICS ANALYSIS")
    print("="*50)
    
    # Get final epoch metrics (best performance)
    final_epoch = df.iloc[-1]
    
    # Get best metrics across all epochs
    best_map50 = df['metrics/mAP50(B)'].max()
    best_map50_95 = df['metrics/mAP50-95(B)'].max()
    best_precision = df['metrics/precision(B)'].max()
    best_recall = df['metrics/recall(B)'].max()
    
    # Find epochs with best metrics
    best_map50_epoch = df['metrics/mAP50(B)'].idxmax() + 1
    best_map50_95_epoch = df['metrics/mAP50-95(B)'].idxmax() + 1
    
    print("FINAL EPOCH METRICS:")
    print(f"  Precision: {final_epoch['metrics/precision(B)']:.4f}")
    print(f"  Recall: {final_epoch['metrics/recall(B)']:.4f}")
    print(f"  mAP@0.5: {final_epoch['metrics/mAP50(B)']:.4f}")
    print(f"  mAP@0.5:0.95: {final_epoch['metrics/mAP50-95(B)']:.4f}")
    
    print("\nBEST METRICS ACHIEVED:")
    print(f"  Best Precision: {best_precision:.4f}")
    print(f"  Best Recall: {best_recall:.4f}")
    print(f"  Best mAP@0.5: {best_map50:.4f} (Epoch {best_map50_epoch})")
    print(f"  Best mAP@0.5:0.95: {best_map50_95:.4f} (Epoch {best_map50_95_epoch})")
    
    return {
        'final_precision': final_epoch['metrics/precision(B)'],
        'final_recall': final_epoch['metrics/recall(B)'],
        'final_map50': final_epoch['metrics/mAP50(B)'],
        'final_map50_95': final_epoch['metrics/mAP50-95(B)'],
        'best_precision': best_precision,
        'best_recall': best_recall,
        'best_map50': best_map50,
        'best_map50_95': best_map50_95,
        'best_map50_epoch': best_map50_epoch,
        'best_map50_95_epoch': best_map50_95_epoch
    }


def analyze_loss_metrics(df):
    """
    Analyze training and validation loss metrics
    
    Args:
        df (pd.DataFrame): Results dataframe
        
    Returns:
        dict: Loss statistics
    """
    print("\n" + "="*50)
    print("LOSS ANALYSIS")
    print("="*50)
    
    # Filter out NaN values for validation losses
    df_clean = df.dropna(subset=['val/box_loss', 'val/cls_loss', 'val/dfl_loss'])
    
    if len(df_clean) == 0:
        print("No validation loss data available")
        return {}
    
    final_epoch = df.iloc[-1]
    
    print("FINAL EPOCH LOSSES:")
    print(f"  Training Box Loss: {final_epoch['train/box_loss']:.4f}")
    print(f"  Training Class Loss: {final_epoch['train/cls_loss']:.4f}")
    print(f"  Training DFL Loss: {final_epoch['train/dfl_loss']:.4f}")
    
    if not pd.isna(final_epoch['val/box_loss']):
        print(f"  Validation Box Loss: {final_epoch['val/box_loss']:.4f}")
        print(f"  Validation Class Loss: {final_epoch['val/cls_loss']:.4f}")
        print(f"  Validation DFL Loss: {final_epoch['val/dfl_loss']:.4f}")
    
    # Best (lowest) losses
    best_train_box = df['train/box_loss'].min()
    best_val_box = df_clean['val/box_loss'].min() if len(df_clean) > 0 else None
    
    print(f"\nBEST LOSSES:")
    print(f"  Best Training Box Loss: {best_train_box:.4f}")
    if best_val_box:
        print(f"  Best Validation Box Loss: {best_val_box:.4f}")
    
    return {
        'final_train_box_loss': final_epoch['train/box_loss'],
        'final_train_cls_loss': final_epoch['train/cls_loss'],
        'final_train_dfl_loss': final_epoch['train/dfl_loss'],
        'best_train_box_loss': best_train_box,
        'best_val_box_loss': best_val_box
    }


def plot_training_curves(df, output_dir="plots"):
    """
    Create training curve plots
    
    Args:
        df (pd.DataFrame): Results dataframe
        output_dir (str): Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLO Training Analysis', fontsize=16)
    
    # 1. Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
    if not df['val/box_loss'].isna().all():
        df_val = df.dropna(subset=['val/box_loss'])
        ax1.plot(df_val['epoch'], df_val['val/box_loss'], label='Val Box Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Box Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 2. mAP Metrics
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
    ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.set_title('Mean Average Precision')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Precision and Recall
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='purple')
    ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='brown')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision and Recall')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Learning Rate
    ax4 = axes[1, 1]
    ax4.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='red')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / "training_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {plot_path}")
    
    # Show plot
    plt.show()


def generate_summary_report(time_stats, perf_stats, loss_stats, filename):
    """
    Generate a summary report
    
    Args:
        time_stats (dict): Training time statistics
        perf_stats (dict): Performance statistics
        loss_stats (dict): Loss statistics
        output_dir (str): Directory to save report
    """
    output_path = Path("reports")
    output_path.mkdir(exist_ok=True)
    
    report_path = output_path / filename
    
    with open(report_path, 'w') as f:
        f.write("YOLO TRAINING SUMMARY REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("TRAINING TIME:\n")

        total_seconds = time_stats['total_time']
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        f.write(f"  Total Time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_seconds:.1f} seconds)\n")
        f.write(f"  Epochs: {time_stats['num_epochs']}\n")
        avg_seconds = time_stats['avg_time_per_epoch']
        avg_hours = int(avg_seconds // 3600)
        avg_minutes = int((avg_seconds % 3600) // 60)
        avg_secs = int(avg_seconds % 60)
        f.write(f"  Avg Time/Epoch: {avg_hours:02d}:{avg_minutes:02d}:{avg_secs:02d} ({avg_seconds:.1f} seconds)\n\n")
        
        f.write("FINAL PERFORMANCE:\n")
        f.write(f"  Precision: {perf_stats['final_precision']:.4f}\n")
        f.write(f"  Recall: {perf_stats['final_recall']:.4f}\n")
        f.write(f"  mAP@0.5: {perf_stats['final_map50']:.4f}\n")
        f.write(f"  mAP@0.5:0.95: {perf_stats['final_map50_95']:.4f}\n\n")
        
        f.write("BEST PERFORMANCE:\n")
        f.write(f"  Best mAP@0.5: {perf_stats['best_map50']:.4f} (Epoch {perf_stats['best_map50_epoch']})\n")
        f.write(f"  Best mAP@0.5:0.95: {perf_stats['best_map50_95']:.4f} (Epoch {perf_stats['best_map50_95_epoch']})\n\n")
        
        f.write("FINAL LOSSES:\n")
        f.write(f"  Train Box Loss: {loss_stats['final_train_box_loss']:.4f}\n")
        f.write(f"  Train Class Loss: {loss_stats['final_train_cls_loss']:.4f}\n")
        f.write(f"  Best Train Box Loss: {loss_stats['best_train_box_loss']:.4f}\n")
    
    print(f"\nSummary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze YOLO training results")
    parser.add_argument("--results", default="../models/training_results_x/results.csv", 
                       help="Path to results.csv file")
    parser.add_argument("--filename", default="training_summary.txt", 
                       help="Output filename for the summary report")
    parser.add_argument("--no-plots", action="store_true", 
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    print("YOLO Training Results Analysis")
    print("="*50)
    
    # Load results
    df = load_results(args.results)
    if df is None:
        return
    
    # Analyze different aspects
    time_stats = analyze_training_time(df)
    perf_stats = analyze_performance_metrics(df)
    loss_stats = analyze_loss_metrics(df)
    
    # Generate summary report
    generate_summary_report(time_stats, perf_stats, loss_stats, args.filename)

    print(f"Files saved to : {Path('reports')}")

if __name__ == "__main__":
    main()
    