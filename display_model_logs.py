#!/usr/bin/env python3
"""
Display Model Logs Utility

This script reads the model training log file and displays the information
in a readable format, with options for filtering and sorting.
"""

import os
import json
import argparse
from tabulate import tabulate
import datetime
from colorama import Fore, Style, init
from pathlib import Path

# Initialize colorama for cross-platform colored output
init()

def load_log_data(logs_folder="model_logs", log_filename="model_training_log.json"):
    """Load the model log data from the JSON file."""
    # Get the absolute path of the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, logs_folder, log_filename)
    
    if not os.path.exists(log_file):
        print(f"{Fore.RED}Log file not found at {log_file}{Style.RESET_ALL}")
        return []
    
    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        return log_data
    except json.JSONDecodeError:
        print(f"{Fore.RED}Error decoding JSON from log file{Style.RESET_ALL}")
        return []

def format_parameters(params):
    """Format model parameters for readable display."""
    # Format parameters to a more concise representation
    formatted = {}
    for key, value in params.items():
        # Skip complex parameters or convert to simpler representation
        if isinstance(value, (dict, list)) and len(str(value)) > 50:
            formatted[key] = f"[complex value]"
        else:
            formatted[key] = value
    return formatted

def display_models(log_data, sort_by="timestamp", reverse=False, limit=None, detailed=False):
    """Display models in a tabular format with optional sorting and filtering."""
    if not log_data:
        print(f"{Fore.YELLOW}No model logs found.{Style.RESET_ALL}")
        return
    
    # Prepare table data
    table_data = []
    for i, entry in enumerate(log_data):
        # Basic information for all views
        model_type = entry.get("additional_info", {}).get("model_type", "Unknown")
        training_time = entry.get("training_time_formatted", "N/A")
        mae = entry.get("metrics", {}).get("mae", "N/A")
        r2 = entry.get("metrics", {}).get("r2", "N/A")
        
        # Get RGB and downsampling factor from additional_info
        add_info = entry.get("additional_info", {})
        rgb = add_info.get("config_rgb", add_info.get("load_rgb", "N/A"))
        downsample = add_info.get("config_downsample", add_info.get("downsample_factor", "N/A"))
        
        row = [
            i+1,  # Index for reference
            entry.get("timestamp", "Unknown"),
            model_type,
            rgb,
            downsample,
            mae,
            r2,
            training_time,
        ]
        table_data.append(row)
    
    # Sort data if needed
    if sort_by == "timestamp":
        table_data.sort(key=lambda x: x[1], reverse=reverse)
    elif sort_by == "mae":
        table_data.sort(key=lambda x: float(x[5]) if x[5] != "N/A" else float('inf'), reverse=reverse)
    elif sort_by == "r2":
        table_data.sort(key=lambda x: float(x[6]) if x[6] != "N/A" else float('-inf'), reverse=reverse)
    elif sort_by == "training_time":
        table_data.sort(key=lambda x: x[7] if x[7] != "N/A" else "999:99:99", reverse=reverse)
    elif sort_by == "rgb":
        table_data.sort(key=lambda x: str(x[3]), reverse=reverse)
    elif sort_by == "downsample":
        # Sort numerically if possible, otherwise as strings
        def get_downsample_key(x):
            try:
                return float(x[4])
            except (ValueError, TypeError):
                return str(x[4])
        table_data.sort(key=get_downsample_key, reverse=reverse)
    
    # Limit the number of entries if needed
    if limit and limit > 0:
        table_data = table_data[:limit]
    
    # Display the table
    headers = ["#", "Timestamp", "Model Type", "RGB", "Downsample", "MAE", "R² (%)", "Training Time"]
    print(f"\n{Fore.CYAN}== Model Training Log =={Style.RESET_ALL}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # If detailed view is requested, show more information about each model
    if detailed:
        for i, entry in enumerate(log_data):
            if limit and i >= limit:
                break
                
            print(f"\n{Fore.GREEN}Model #{i+1} Details:{Style.RESET_ALL}")
            print(f"  {Fore.BLUE}Name:{Style.RESET_ALL} {entry.get('model_name', 'Unnamed')}")
            
            # Display dataset configuration prominently
            add_info = entry.get("additional_info", {})
            rgb = add_info.get("config_rgb", add_info.get("load_rgb", "N/A"))
            downsample = add_info.get("config_downsample", add_info.get("downsample_factor", "N/A"))
            
            print(f"  {Fore.BLUE}Dataset Config:{Style.RESET_ALL}")
            print(f"    - RGB: {rgb}")
            print(f"    - Downsample Factor: {downsample}")
            
            # Display parameters
            params = entry.get("parameters", {})
            formatted_params = format_parameters(params)
            print(f"  {Fore.BLUE}Parameters:{Style.RESET_ALL}")
            for key, value in formatted_params.items():
                print(f"    - {key}: {value}")
            
            # Display additional info if available
            additional_info = entry.get("additional_info", {})
            if additional_info:
                print(f"  {Fore.BLUE}Additional Info:{Style.RESET_ALL}")
                for key, value in additional_info.items():
                    if key != "model_type" and key not in ("config_rgb", "load_rgb", "config_downsample", "downsample_factor"):
                        print(f"    - {key}: {value}")
            
            print(f"  {Fore.BLUE}Training Time:{Style.RESET_ALL} {entry.get('training_time_formatted', 'N/A')} ({entry.get('training_time_seconds', 'N/A')} seconds)")
            print("  " + "-" * 50)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Display model training logs in a readable format")
    parser.add_argument("--sort", choices=["timestamp", "mae", "r2", "training_time", "rgb", "downsample"], 
                        default="timestamp", help="Sort results by field")
    parser.add_argument("--reverse", action="store_true", help="Reverse the sorting order")
    parser.add_argument("--limit", type=int, help="Limit the number of models displayed")
    parser.add_argument("--detailed", action="store_true", help="Show detailed information for each model")
    return parser.parse_args()

if __name__ == "__main__":
    # Check if tabulate is installed
    try:
        import tabulate
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.call(["pip", "install", "tabulate", "colorama"])
        # Re-import after installation
        import tabulate
        from colorama import Fore, Style, init
        init()
    
    args = parse_arguments()
    log_data = load_log_data()
    display_models(
        log_data, 
        sort_by=args.sort, 
        reverse=args.reverse, 
        limit=args.limit, 
        detailed=args.detailed
    )