#!/usr/bin/env python3
"""
Display Model Logs Tool

This script analyzes and visualizes the results from model training logs to
help track progress and understand which parameters are most important in
reducing the MAE for the distance estimation task.

Usage:
    python display_model_logs.py [--log-file model_training_log.json] [--top N] [--detailed]
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import pandas as pd

# Default log file path
LOG_FILE_PATH = os.path.join("model_logs", "model_training_log.json")
SPECIALIZED_LOG_FILES = {
    "xgb": "xgb_trainer_log.json",
    "lgbm": "lgbm_trainer_log.json",
    "auto": "auto_trainer_log.json"
}
TOP_N = 10  # Default number of top models to display

# Important parameters to display in detailed mode for different models
IMPORTANT_PARAMS = {
    "XGBoost": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"],
    "LightGBM": ["n_estimators", "num_leaves", "learning_rate", "max_depth", "feature_fraction"],
    "RandomForest": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"],
    "SVR": ["C", "gamma", "epsilon", "kernel"],
    "HistGradient": ["max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization"],
    "Generic": ["n_estimators", "max_depth", "learning_rate"]  # Fallback parameters
}


def load_logs(log_file_path):
    """Load the logs from the specified file."""
    try:
        with open(log_file_path, 'r') as f:
            logs = json.load(f)
        return logs
    except Exception as e:
        print(f"Error loading log file: {str(e)}")
        return []


def display_top_models(logs, top_n=TOP_N, detailed=False):
    """Display the top N models by MAE.
    
    Args:
        logs: List of log entries
        top_n: Number of top models to display
        detailed: Whether to display detailed parameter information
    """
    if not logs:
        print("No logs found.")
        return
    
    # Sort logs by MAE
    sorted_logs = sorted(logs, key=lambda x: x["metrics"]["mae"])
    
    # Take top N
    top_logs = sorted_logs[:top_n]
    
    # Simple table view (non-detailed mode)
    if not detailed:
        print(f"\n===== Top {len(top_logs)} Models by MAE =====")
        print(f"{'Rank':<5}{'MAE':<10}{'R²':<10}{'Model Type':<15}{'Time (s)':<10}{'Date':<20}")
        print("-" * 70)
        
        for i, log in enumerate(top_logs):
            mae = log["metrics"]["mae"]
            r2 = log["metrics"]["r2"]
            model_type = log.get("additional_info", {}).get("model_type", "Unknown")
            training_time = log.get("training_time", 0)
            timestamp = log.get("timestamp", "Unknown")
            
            print(f"{i+1:<5}{mae:<10.4f}{r2:<10.4f}{model_type:<15}{training_time:<10.1f}{timestamp:<20}")
    else:
        # Detailed view with parameters
        print(f"\n===== Top {len(top_logs)} Models by MAE (Detailed View) =====")
        
        for i, log in enumerate(top_logs):
            mae = log["metrics"]["mae"]
            r2 = log["metrics"]["r2"]
            model_type = log.get("additional_info", {}).get("model_type", "Unknown")
            training_time = log.get("training_time", 0)
            timestamp = log.get("timestamp", "Unknown")
            
            print(f"\n{'-'*100}")
            print(f"Rank {i+1}: MAE = {mae:.4f}, R² = {r2:.4f}")
            print(f"Model Type: {model_type}")
            print(f"Training Time: {training_time:.1f}s")
            print(f"Date: {timestamp}")
            
            # Get model parameters - safely handle missing keys
            model_params = {}
            if "model_params" in log:
                model_params = log["model_params"]
            # For log entries where params are in a different format
            elif "config" in log and "model_params" in log["config"]:
                model_params = log["config"]["model_params"]
            
            if not model_params:
                print("\nNo model parameters found for this entry")
                continue
                
            # Determine which parameters to display based on model type
            param_list = IMPORTANT_PARAMS.get("Generic", [])  # Default
            
            if "XGBoost" in model_type:
                param_list = IMPORTANT_PARAMS["XGBoost"]
            elif "LightGBM" in model_type:
                param_list = IMPORTANT_PARAMS["LightGBM"]
            elif "RandomForest" in model_type:
                param_list = IMPORTANT_PARAMS["RandomForest"]
            elif "SVR" in model_type:
                param_list = IMPORTANT_PARAMS["SVR"]
            elif "HistGradient" in model_type:
                param_list = IMPORTANT_PARAMS["HistGradient"]
            
            # Check for important parameters in the model
            important_params_present = []
            for param in param_list:
                # Handle regressor__ prefix (from pipeline naming)
                if param in model_params:
                    important_params_present.append((param, model_params[param]))
                elif f"regressor__{param}" in model_params:
                    important_params_present.append((param, model_params[f"regressor__{param}"]))
            
            if important_params_present:
                print("\nKey Parameters:")
                for param, value in important_params_present:
                    print(f"  {param:<20}: {value}")
            
            # Display additional configuration details
            if "additional_info" in log:
                add_info = log["additional_info"]
                print("\nConfiguration:")
                
                # Common configuration details
                if "config_rgb" in add_info:
                    print(f"  {'load_rgb':<20}: {add_info['config_rgb']}")
                if "config_downsample" in add_info:
                    print(f"  {'downsample_factor':<20}: {add_info['config_downsample']}")
                if "feature_count" in add_info:
                    print(f"  {'feature_count':<20}: {add_info['feature_count']}")
                if "dataset_size" in add_info:
                    print(f"  {'dataset_size':<20}: {add_info['dataset_size']}")
                
                # Any other potentially interesting configuration information
                other_interesting_keys = ["randomized_search_best_score", "feature_extraction_method"]
                for key in other_interesting_keys:
                    if key in add_info:
                        print(f"  {key:<20}: {add_info[key]}")
            
            # Display all parameters if there are relatively few
            if len(model_params) <= 5 or len(important_params_present) == 0:
                print("\nAll Parameters:")
                for param, value in model_params.items():
                    # Skip parameters already shown
                    if any(param == p[0] or param == f"regressor__{p[0]}" for p in important_params_present):
                        continue
                    print(f"  {param:<20}: {value}")
    
    # Always display the best model's parameters at the end
    best_model = top_logs[0]
    print("\n===== Best Model Parameters =====")
    
    # Safely access model parameters
    model_params = {}
    if "model_params" in best_model:
        model_params = best_model["model_params"]
    elif "config" in best_model and "model_params" in best_model["config"]:
        model_params = best_model["config"]["model_params"]
    
    if model_params:
        for param, value in model_params.items():
            print(f"{param}: {value}")
    else:
        print("No model parameters found for the best model")
    
    # Additional info for the best model
    if "additional_info" in best_model:
        print("\n===== Additional Info for Best Model =====")
        for key, value in best_model["additional_info"].items():
            print(f"{key}: {value}")


def plot_mae_progress(logs):
    """Plot the MAE progress over time."""
    if not logs:
        print("No logs found.")
        return
    
    # Extract timestamps and MAEs
    timestamps = []
    maes = []
    model_types = []
    
    for log in logs:
        try:
            timestamp = log.get("timestamp", "")
            if timestamp:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                timestamps.append(dt)
                maes.append(log["metrics"]["mae"])
                model_types.append(log.get("additional_info", {}).get("model_type", "Unknown"))
        except Exception as e:
            print(f"Error processing log entry: {str(e)}")
    
    if not timestamps:
        print("No valid timestamps found in logs.")
        return
    
    # Sort by timestamp
    sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    sorted_timestamps = [timestamps[i] for i in sorted_indices]
    sorted_maes = [maes[i] for i in sorted_indices]
    sorted_model_types = [model_types[i] for i in sorted_indices]
    
    # Get unique model types for coloring
    unique_model_types = list(set(sorted_model_types))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_model_types)))
    type_to_color = {t: colors[i] for i, t in enumerate(unique_model_types)}
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot points with color by model type
    for model_type in unique_model_types:
        indices = [i for i, t in enumerate(sorted_model_types) if t == model_type]
        plt.scatter(
            [sorted_timestamps[i] for i in indices],
            [sorted_maes[i] for i in indices],
            label=model_type,
            alpha=0.7
        )
    
    # Plot overall trend line
    z = np.polyfit(range(len(sorted_timestamps)), sorted_maes, 1)
    p = np.poly1d(z)
    plt.plot(sorted_timestamps, p(range(len(sorted_timestamps))), "r--", alpha=0.8, label="Trend")
    
    plt.title("MAE Progress Over Time")
    plt.xlabel("Time")
    plt.ylabel("MAE (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis to show dates nicely
    plt.gcf().autofmt_xdate()
    
    # Save and show
    plt.savefig("mae_progress.png")
    plt.show()


def plot_parameter_importance(logs):
    """Plot the relationship between key parameters and MAE."""
    if not logs:
        print("No logs found.")
        return
    
    # Extract parameters and MAEs
    data = defaultdict(list)
    maes = []
    
    for log in logs:
        try:
            mae = log["metrics"]["mae"]
            if mae > 5:  # Skip outliers
                continue
                
            maes.append(mae)
            
            # Extract parameters
            params = log["model_params"]
            for param, value in params.items():
                if isinstance(value, (int, float)):
                    data[param].append(value)
                else:
                    # Skip non-numeric parameters
                    continue
            
            # Extract important config values
            if "additional_info" in log:
                if "config_rgb" in log["additional_info"]:
                    data["load_rgb"].append(1 if log["additional_info"]["config_rgb"] else 0)
                if "config_downsample" in log["additional_info"]:
                    data["downsample_factor"].append(log["additional_info"]["config_downsample"])
        except Exception as e:
            print(f"Error processing log entry for parameter importance: {str(e)}")
    
    # Filter parameters with enough data points
    filtered_params = {k: v for k, v in data.items() if len(v) == len(maes) and len(v) > 5}
    
    if not filtered_params:
        print("Not enough parameter data for analysis.")
        return
    
    # Create a DataFrame for correlation analysis
    df = pd.DataFrame(filtered_params)
    df['mae'] = maes
    
    # Calculate correlations with MAE
    correlations = df.corr()['mae'].drop('mae').sort_values(ascending=False)
    
    print("\n===== Parameter Correlations with MAE =====")
    for param, corr in correlations.items():
        print(f"{param:<30}: {corr:>8.4f}")
    
    # Plot top parameters vs MAE
    num_params = min(6, len(filtered_params))
    top_params = correlations.index[:num_params].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(top_params):
        if i >= num_params:
            break
            
        ax = axes[i]
        ax.scatter(df[param], df['mae'], alpha=0.7)
        ax.set_title(f"{param} vs MAE (corr: {correlations[param]:.4f})")
        ax.set_xlabel(param)
        ax.set_ylabel("MAE (m)")
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df[param], df['mae'], 1)
        p = np.poly1d(z)
        ax.plot(df[param], p(df[param]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig("parameter_importance.png")
    plt.show()


def analyze_specialized_trainers():
    """Analyze and compare the logs from specialized trainers."""
    logs = {}
    
    # Load logs from each specialized trainer
    for name, file_path in SPECIALIZED_LOG_FILES.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    logs[name] = json.load(f)
                print(f"Loaded {len(logs[name])} logs from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        else:
            print(f"Log file {file_path} not found.")
    
    if not logs:
        print("No specialized trainer logs found.")
        return
    
    # Plot MAE progress comparison
    plt.figure(figsize=(12, 6))
    
    for name, log_data in logs.items():
        if not log_data:
            continue
            
        # Extract iterations and MAEs
        iterations = []
        maes = []
        
        for entry in log_data:
            try:
                # Skip failed runs
                if entry.get("mae", 999) > 10:
                    continue
                    
                iterations.append(entry["iteration"])
                maes.append(entry["mae"])
            except Exception as e:
                print(f"Error processing log entry: {str(e)}")
        
        if not iterations:
            continue
            
        # Sort by iteration
        sorted_indices = sorted(range(len(iterations)), key=lambda i: iterations[i])
        sorted_iterations = [iterations[i] for i in sorted_indices]
        sorted_maes = [maes[i] for i in sorted_indices]
        
        # Plot progress
        plt.plot(sorted_iterations, sorted_maes, marker='o', linestyle='-', alpha=0.7, label=name.upper())
        
        # Find and annotate best MAE
        if sorted_maes:
            best_mae = min(sorted_maes)
            best_idx = sorted_maes.index(best_mae)
            plt.annotate(
                f"{best_mae:.4f}",
                (sorted_iterations[best_idx], sorted_maes[best_idx]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
    
    plt.title("MAE Progress Comparison of Specialized Trainers")
    plt.xlabel("Iteration")
    plt.ylabel("MAE (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Target MAE line
    plt.axhline(y=0.08, color='g', linestyle='--', alpha=0.5, label="Target (0.08)")
    
    plt.savefig("specialized_trainers_comparison.png")
    plt.show()
    
    # Print best results for each trainer
    print("\n===== Best Results for Each Trainer =====")
    for name, log_data in logs.items():
        if not log_data:
            continue
            
        valid_logs = [entry for entry in log_data if entry.get("mae", 999) < 10]
        if not valid_logs:
            continue
            
        best_entry = min(valid_logs, key=lambda x: x["mae"])
        print(f"\n{name.upper()} Best Result:")
        print(f"MAE: {best_entry['mae']:.4f}")
        print(f"Iteration: {best_entry['iteration']}")
        print(f"Phase: {best_entry.get('phase', 'unknown')}")
        
        if "config" in best_entry:
            print("\nBest Configuration:")
            print(f"  load_rgb: {best_entry['config'].get('load_rgb', 'unknown')}")
            print(f"  downsample_factor: {best_entry['config'].get('downsample_factor', 'unknown')}")
            
            if "model_params" in best_entry["config"]:
                print("\nModel Parameters:")
                for param, value in best_entry["config"]["model_params"].items():
                    print(f"  {param}: {value}")


def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Analyze and visualize model training logs")
    parser.add_argument("--log-file", type=str, default=LOG_FILE_PATH,
                        help=f"Path to log file (default: {LOG_FILE_PATH})")
    parser.add_argument("--top", type=int, default=TOP_N,
                        help=f"Number of top models to display (default: {TOP_N})")
    parser.add_argument("--specialized", action="store_true",
                        help="Analyze specialized trainer logs")
    parser.add_argument("--detailed", action="store_true",
                        help="Display detailed model parameters")
    args = parser.parse_args()
    
    # Check if specialized trainer analysis was requested
    if args.specialized:
        analyze_specialized_trainers()
        return
    
    # Load logs
    logs = load_logs(args.log_file)
    
    if logs:
        print(f"Loaded {len(logs)} log entries.")
        
        # Display top models
        display_top_models(logs, args.top, args.detailed)
        
        # Visualizations
        try:
            import pandas as pd
            plot_mae_progress(logs)
            plot_parameter_importance(logs)
        except ImportError:
            print("\nWarning: pandas is required for visualizations. Please install with: pip install pandas")
        except Exception as e:
            print(f"\nError creating visualizations: {str(e)}")
    else:
        print("No logs found or unable to load log file.")


if __name__ == "__main__":
    main()