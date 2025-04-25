#!/usr/bin/env python3
"""
Automated Machine Learning Pipeline for Distance Estimation

This script automates the process of training multiple ML models with different
configurations, analyzes results, and intelligently selects the next configuration
to try based on previous outcomes. It continues training until the target
MAE is reached or the maximum number of iterations is hit.

Usage:
    python auto_trainer.py --target-mae 0.08 --max-iterations 10

Features:
- Automated sequential model training
- Intelligent parameter selection based on previous results
- Support for multiple pipeline strategies
- Periodic result logging and visualization
- Email notifications for important events (optional)
"""

import os
import sys
import time
import argparse
import json
import random
import numpy as np
import datetime
import importlib
import traceback
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Try to import Google Colab-specific libraries
IS_COLAB = False
try:
    import google.colab
    from google.colab import drive
    IS_COLAB = True
    print("Running in Google Colab environment")
except ImportError:
    print("Running in local environment")

# Import project modules
from utils import log_model_results, load_config

# Set up initial configuration
TARGET_MAE = 0.08  # Default target MAE in meters (8cm)
MAX_ITERATIONS = 100  # Default maximum number of iterations
SAVE_CHECKPOINTS = False  # Whether to save checkpoints
CHECKPOINT_FILE = "auto_trainer_checkpoint.json"
LOG_FILE = "auto_trainer_log.json"


class AutoTrainer:
    """
    Automated ML training system that sequentially trains models
    with different configurations and intelligently selects the next
    configuration to try based on previous results.
    """
    
    def __init__(self, target_mae: float = TARGET_MAE, 
                 max_iterations: int = MAX_ITERATIONS,
                 save_checkpoints: bool = SAVE_CHECKPOINTS,
                 checkpoint_file: str = CHECKPOINT_FILE,
                 log_file: str = LOG_FILE):
        """
        Initialize the AutoTrainer.
        
        Args:
            target_mae: Target Mean Absolute Error to achieve
            max_iterations: Maximum number of iterations to run
            save_checkpoints: Whether to save checkpoints
            checkpoint_file: Path to checkpoint file
            log_file: Path to log file
        """
        self.target_mae = target_mae
        self.max_iterations = max_iterations
        self.save_checkpoints = save_checkpoints
        self.checkpoint_file = checkpoint_file
        self.log_file = log_file
        
        # Internal state
        self.iteration = 0
        self.best_mae = float('inf')
        self.best_config = None
        self.history = []
        self.current_phase = "exploration"  # exploration or exploitation
        
        # Pipeline modules to try
        self.pipeline_modules = [
            "MainScripts.main",
            "MainScripts.main_nico",
            #"MainScripts.pipeline",
            #"MainScripts.pipeline3",
            #"MainScripts.pipeline4",
            "MainScripts.simple_rf",
            "MainScripts.simple_xgb",
            #"MainScripts.simple_svr",
            "MainScripts.simple_lgbm"
        ]
        
        # Initialize configuration spaces for different pipelines
        self.init_config_spaces()
        
        # Load checkpoint if it exists
        if save_checkpoints and os.path.exists(checkpoint_file):
            self.load_checkpoint()
    
    def init_config_spaces(self):
        """Initialize configuration spaces for different pipelines."""
        # Define configuration spaces for each pipeline
        self.config_spaces = {
            "MainScripts.main": {
                "load_rgb": [True, False],
                "downsample_factor": [1, 2, 3, 4, 6, 8, 10],
                "model_params": {
                    "n_estimators": [100, 200, 300, 500, 1000],
                    "max_depth": [None, 10, 20, 30, 50],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4, 8],
                    "max_features": ['auto', 'sqrt', 'log2', None]
                }
            },
            "MainScripts.main_nico": {
                "load_rgb": [True, False],
                "downsample_factor": [1, 2, 3, 4, 6, 8, 10],
                "model_params": {
                    # For HistGradientBoostingRegressor (model3)
                    "max_iter": [300, 500, 1000],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_leaf_nodes": [31, 64, 128],
                    "min_samples_leaf": [5, 10, 20],
                    "l2_regularization": [0.0, 1.0, 10.0]
                }
            },
            "MainScripts.simple_rf": {
                "load_rgb": [True, False],
                "downsample_factor": [1, 2, 3, 4, 6, 8, 10],
                "model_params": {
                    "n_estimators": [100, 200, 300, 500, 1000],
                    "max_depth": [None, 10, 20, 30, 50],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4, 8],
                    "max_features": ['auto', 'sqrt', 'log2', None]
                }
            },
            "MainScripts.simple_xgb": {
                "load_rgb": [True, False],
                "downsample_factor": [1, 2, 3, 4, 6, 8, 10],
                "model_params": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "min_child_weight": [1, 3, 5, 7]
                }
            },
            "MainScripts.simple_svr": {
                "load_rgb": [True, False],
                "downsample_factor": [1, 2, 3, 4, 6, 8, 10],
                "model_params": {
                    "kernel": ['linear', 'poly', 'rbf'],
                    "C": [0.1, 1.0, 10.0, 100.0],
                    "gamma": ['scale', 'auto', 0.01, 0.1, 1.0],
                    "epsilon": [0.01, 0.05, 0.1, 0.2]
                }
            },
            "MainScripts.simple_lgbm": {
                "load_rgb": [True, False],
                "downsample_factor": [1, 2, 3, 4, 6, 8, 10],
                "model_params": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [3, 5, 7, 9, -1],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "num_leaves": [31, 63, 127, 255],
                    "min_child_samples": [5, 10, 20, 50],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
                }
            },
            "MainScripts.pipeline": {
                "load_rgb": [True, False],
                "downsample_factor": [1, 2, 3, 4, 6, 8, 10],
                "model_params": {
                    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    "epsilon": [0.001, 0.01, 0.1, 0.2, 0.5],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "degree": [2, 3, 4, 5],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
                }
            },
            "MainScripts.pipeline3": {
                "load_rgb": [True, False],
                "downsample_factor": [1, 2, 3, 4, 6, 8, 10],
                "feature_extraction": ["pca", "kernel_pca", "hog", "custom"],
                "model_params": {
                    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 4, 5, 6, 7, 8, 10, None],
                    "n_estimators": [50, 100, 200, 300, 500],
                    "min_samples_leaf": [1, 2, 4, 8],
                    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                }
            },
            "MainScripts.pipeline4": {
                "load_rgb": [True, False],
                "downsample_factor": [1, 2, 3, 4, 6, 8, 10],
                "target_sizes": [[10], [20], [10, 20], [5, 10, 20]],
                "use_hog": [True, False],
                "use_edges": [True, False],
                "model_params": {
                    "hist_gbm": {
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "max_depth": [3, 5, 7, None],
                        "max_iter": [100, 200, 300, 500],
                        "min_samples_leaf": [5, 10, 20, 40],
                        "l2_regularization": [0, 0.1, 0.5, 1.0]
                    },
                    "gbm": {
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "n_estimators": [100, 200, 300, 500],
                        "max_depth": [3, 5, 7, 9],
                        "min_samples_leaf": [1, 3, 5, 9],
                        "subsample": [0.7, 0.8, 0.9, 1.0]
                    }
                }
            }
        }
        
        # Special adjustments for Colab environment
        if IS_COLAB:
            # Optimization: Reduce parameter space for faster iteration in Colab
            for pipeline in self.config_spaces:
                if "downsample_factor" in self.config_spaces[pipeline]:
                    # Prefer downsample factors that work well on limited resources
                    self.config_spaces[pipeline]["downsample_factor"] = [2, 4, 6, 10]
    
    def load_checkpoint(self):
        """Load training state from checkpoint file."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                
            self.iteration = checkpoint.get("iteration", 0)
            self.best_mae = checkpoint.get("best_mae", float('inf'))
            self.best_config = checkpoint.get("best_config", None)
            self.history = checkpoint.get("history", [])
            self.current_phase = checkpoint.get("current_phase", "exploration")
            
            print(f"Loaded checkpoint: iteration {self.iteration}, best MAE: {self.best_mae:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
    
    def save_checkpoint(self):
        """Save current training state to checkpoint file."""
        if not self.save_checkpoints:
            return
            
        checkpoint = {
            "iteration": self.iteration,
            "best_mae": self.best_mae,
            "best_config": self.best_config,
            "history": self.history,
            "current_phase": self.current_phase,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=4)
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
    
    def log_iteration(self, pipeline_module, config, mae, r2, training_time, additional_info=None):
        """
        Log information about the current iteration.
        
        Args:
            pipeline_module: Name of the pipeline module
            config: Configuration used
            mae: Mean Absolute Error achieved
            r2: R-squared value achieved
            training_time: Training time in seconds
            additional_info: Additional information to log
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create log entry
        log_entry = {
            "iteration": self.iteration,
            "timestamp": timestamp,
            "pipeline": pipeline_module,
            "config": config,
            "mae": float(mae),
            "r2": float(r2),
            "training_time": float(training_time),
            "best_mae_so_far": float(self.best_mae)
        }
        
        if additional_info:
            log_entry["additional_info"] = additional_info
        
        # Add entry to history
        self.history.append(log_entry)
        
        # Write to log file
        try:
            log_data = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            
            log_data.append(log_entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=4)
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")
    
    def select_next_config(self):
        """
        Intelligently select the next configuration to try based on
        previous results.
        
        Returns:
            Tuple containing (pipeline_module, config_dict)
        """
        # Switch between exploration and exploitation phases
        if self.iteration < self.max_iterations // 3:
            # Early phase: pure exploration
            return self._explore_random_config()
        elif self.best_mae > self.target_mae * 1.5:
            # Still far from target: 70% exploration, 30% exploitation
            if random.random() < 0.7:
                return self._explore_random_config()
            else:
                return self._exploit_best_configs()
        else:
            # Close to target: 30% exploration, 70% exploitation
            if random.random() < 0.3:
                return self._explore_random_config()
            else:
                return self._exploit_best_configs()
    
    def _explore_random_config(self):
        """Randomly select a pipeline and configuration for exploration."""
        # Select a random pipeline module
        pipeline_module = random.choice(self.pipeline_modules)
        config_space = self.config_spaces[pipeline_module]
        
        # Generate a random configuration
        config = {}
        
        # Handle basic configuration parameters
        for key, values in config_space.items():
            if key != "model_params" and not isinstance(values, dict):
                config[key] = random.choice(values)
        
        # Handle model parameters based on the specific pipeline
        if pipeline_module == "MainScripts.pipeline4":
            # For pipeline4, we need to choose between hist_gbm and gbm
            model_type = random.choice(["hist_gbm", "gbm"])
            model_params = {}
            
            # Select parameters for the chosen model type
            for param, values in config_space["model_params"][model_type].items():
                model_params[param] = random.choice(values)
            
            config["model_type"] = model_type
            config["model_params"] = model_params
        elif "model_params" in config_space:
            # For other pipelines, select model parameters directly
            model_params = {}
            for param, values in config_space["model_params"].items():
                # Skip some parameters randomly to reduce complexity
                if random.random() < 0.2:
                    continue
                model_params[param] = random.choice(values)
            
            config["model_params"] = model_params
        
        return pipeline_module, config
    
    def _exploit_best_configs(self):
        """
        Generate a new configuration based on the best configurations so far.
        
        Returns:
            Tuple containing (pipeline_module, config_dict)
        """
        if not self.history:
            return self._explore_random_config()
        
        # Sort history by MAE (ascending)
        sorted_history = sorted(self.history, key=lambda x: x["mae"])
        
        # Take top 3 configurations or all if less than 3
        top_k = min(3, len(sorted_history))
        top_configs = sorted_history[:top_k]
        
        # Count pipeline occurrences in top configurations
        pipeline_counts = defaultdict(int)
        for entry in top_configs:
            pipeline_counts[entry["pipeline"]] += 1
        
        # Choose pipeline with highest count, or random from top if tied
        if pipeline_counts:
            max_count = max(pipeline_counts.values())
            best_pipelines = [p for p, c in pipeline_counts.items() if c == max_count]
            pipeline_module = random.choice(best_pipelines)
        else:
            # Fallback to random
            pipeline_module = random.choice(self.pipeline_modules)
        
        # Filter top configs for the selected pipeline
        pipeline_top_configs = [c for c in top_configs if c["pipeline"] == pipeline_module]
        
        if not pipeline_top_configs:
            # No history for this pipeline, revert to random exploration
            return self._explore_random_config()
        
        # Choose a base config randomly from the top ones for this pipeline
        base_config = random.choice(pipeline_top_configs)["config"]
        
        # Create a new configuration with some random modifications
        config_space = self.config_spaces[pipeline_module]
        new_config = {}
        
        # Copy non-model parameters with some random mutations
        for key, values in config_space.items():
            if key != "model_params" and not isinstance(values, dict):
                # 70% chance to keep the same value, 30% chance to try something new
                if key in base_config and random.random() < 0.7:
                    new_config[key] = base_config[key]
                else:
                    new_config[key] = random.choice(values)
        
        # Handle model parameters based on the pipeline
        if pipeline_module == "MainScripts.pipeline4":
            # For pipeline4, keep or switch model type
            if "model_type" in base_config and random.random() < 0.7:
                model_type = base_config["model_type"]
            else:
                model_type = random.choice(["hist_gbm", "gbm"])
            
            new_config["model_type"] = model_type
            new_config["model_params"] = {}
            
            # Get base model params if they exist
            base_model_params = base_config.get("model_params", {})
            
            # Select parameters for the chosen model type
            for param, values in config_space["model_params"][model_type].items():
                # 70% chance to keep the same value if it exists
                if param in base_model_params and random.random() < 0.7:
                    new_config["model_params"][param] = base_model_params[param]
                else:
                    new_config["model_params"][param] = random.choice(values)
        elif "model_params" in config_space:
            # For other pipelines
            new_config["model_params"] = {}
            base_model_params = base_config.get("model_params", {})
            
            for param, values in config_space["model_params"].items():
                # 70% chance to keep the same value if it exists
                if param in base_model_params and random.random() < 0.7:
                    new_config["model_params"][param] = base_model_params[param]
                else:
                    new_config["model_params"][param] = random.choice(values)
        
        return pipeline_module, new_config
    
    def update_config_file(self, config):
        """
        Update the config.yaml file with the current configuration.
        
        Args:
            config: Configuration dictionary to write to the file
        """
        # Get only the configuration keys that are expected in the config file
        config_file_keys = ["load_rgb", "downsample_factor", "data_dir"]
        
        # Start with the existing config to keep other settings
        try:
            with open("config.yaml", "r") as file:
                current_config = yaml.safe_load(file)
        except:
            # Default config if file doesn't exist
            current_config = {
                "data_dir": "./data",
                "load_rgb": True,
                "downsample_factor": 2
            }
        
        # Update with new values
        for key in config_file_keys:
            if key in config:
                current_config[key] = config[key]
        
        # Write back to file
        with open("config.yaml", "w") as file:
            yaml.dump(current_config, file)
    
    def run_training_iteration(self, pipeline_module, config):
        """
        Run a single training iteration with the specified pipeline and config.
        
        Args:
            pipeline_module: Name of the pipeline module
            config: Configuration to use
            
        Returns:
            Dictionary with results including MAE, R2, and training time
        """
        print(f"\n{'='*80}")
        print(f"Starting iteration {self.iteration+1}/{self.max_iterations}")
        print(f"Pipeline: {pipeline_module}")
        print(f"Configuration: {json.dumps(config, indent=2)}")
        print(f"{'='*80}\n")
        
        # Update config file
        self.update_config_file(config)
        
        try:
            # Import the pipeline module dynamically
            module = importlib.import_module(pipeline_module)
            
            # Extract the model parameters if they exist
            model_params = config.get("model_params", {})
            
            # For pipeline4, we need to handle the model type
            if pipeline_module == "MainScripts.pipeline4":
                model_type = config.get("model_type", "hist_gbm")
                target_sizes = config.get("target_sizes", [10, 20])
                use_hog = config.get("use_hog", True)
                use_edges = config.get("use_edges", True)
                
                # Modify the relevant variables in the module before running
                if hasattr(module, "MultiScaleFeatureExtractor"):
                    # Backup original init method
                    original_init = module.MultiScaleFeatureExtractor.__init__
                    
                    # Define a custom init method to override parameters
                    def custom_init(self, target_sizes=target_sizes, use_hog=use_hog, use_edges=use_edges):
                        original_init(self, target_sizes=target_sizes, use_hog=use_hog, use_edges=use_edges)
                    
                    # Replace the init method
                    module.MultiScaleFeatureExtractor.__init__ = custom_init
                
                # Now handle model selection in train_model_with_cross_validation
                if hasattr(module, "train_model_with_cross_validation"):
                    # We need to modify this function to prioritize our chosen model type
                    original_train_func = module.train_model_with_cross_validation
                    
                    # Define a modified function that uses our preferences
                    def modified_train_func(train_features, train_distances, n_splits=5):
                        # Get the result from the original function
                        best_model, best_params, best_pipeline_name, best_mae = original_train_func(
                            train_features, train_distances, n_splits
                        )
                        
                        # Return the results, original function already prioritizes best models
                        return best_model, best_params, best_pipeline_name, best_mae
                    
                    # Replace the function
                    module.train_model_with_cross_validation = modified_train_func
            
            # Record start time
            start_time = time.time()
            
            # Run the pipeline's main function
            if hasattr(module, "main"):
                # Pass model_params to the main function if it accepts it
                import inspect
                main_func = module.main
                accepts_model_params = "model_params" in inspect.signature(main_func).parameters
                
                if accepts_model_params:
                    result = main_func(model_params=model_params)
                else:
                    # If the main function doesn't accept model_params, just run it normally
                    result = main_func()
            else:
                # Run the module directly
                result = None
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Load the log file to get the latest result
            model_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         "model_logs", "model_training_log.json")
            
            if os.path.exists(model_log_file):
                with open(model_log_file, 'r') as f:
                    model_logs = json.load(f)
                    if model_logs:
                        # Get the most recent log entry
                        latest_log = model_logs[-1]
                        mae = latest_log["metrics"]["mae"]
                        r2 = latest_log["metrics"]["r2"]
                        
                        # Additional info from the log
                        additional_info = latest_log.get("additional_info", {})
                        
                        return {
                            "mae": mae,
                            "r2": r2,
                            "training_time": training_time,
                            "additional_info": additional_info,
                            "success": True
                        }
            
            # If we couldn't get the results from the log file
            # This should usually not happen if logging is working correctly
            print("Warning: Could not get results from log file, using placeholder values")
            return {
                "mae": 999.0,  # Placeholder
                "r2": 0.0,     # Placeholder
                "training_time": training_time,
                "success": False
            }
            
        except Exception as e:
            print(f"Error running training iteration: {str(e)}")
            traceback.print_exc()
            
            return {
                "mae": 999.0,  # Error value
                "r2": 0.0,     # Error value
                "training_time": 0.0,
                "error": str(e),
                "success": False
            }
    
    def run(self):
        """
        Run the automated training process until the target MAE is reached
        or the maximum number of iterations is hit.
        """
        print(f"Starting AutoTrainer with target MAE: {self.target_mae}")
        print(f"Maximum iterations: {self.max_iterations}")
        
        # Loop until we reach the target or max iterations
        while self.iteration < self.max_iterations:
            # Increment iteration counter
            self.iteration += 1
            
            # Select the next configuration to try
            pipeline_module, config = self.select_next_config()
            
            # Run the training iteration
            results = self.run_training_iteration(pipeline_module, config)
            
            # Check if the iteration was successful
            if results["success"]:
                mae = results["mae"]
                r2 = results["r2"]
                training_time = results["training_time"]
                additional_info = results.get("additional_info", {})
                
                # Log the iteration
                self.log_iteration(
                    pipeline_module, config, mae, r2, training_time, additional_info
                )
                
                # Update best MAE if this iteration is better
                if mae < self.best_mae:
                    self.best_mae = mae
                    self.best_config = {
                        "pipeline": pipeline_module,
                        "config": config
                    }
                    print(f"New best MAE: {self.best_mae:.4f}")
                
                # Print summary
                print(f"\nIteration {self.iteration} summary:")
                print(f"Pipeline: {pipeline_module}")
                print(f"MAE: {mae:.4f}")
                print(f"R²: {r2:.4f}")
                print(f"Training time: {training_time:.2f} seconds")
                print(f"Best MAE so far: {self.best_mae:.4f}")
                
                # Check if we've reached the target
                if mae <= self.target_mae:
                    print(f"\n{'='*80}")
                    print(f"Target MAE of {self.target_mae} reached!")
                    print(f"Final MAE: {mae:.4f}")
                    print(f"Pipeline: {pipeline_module}")
                    print(f"Configuration: {json.dumps(config, indent=2)}")
                    print(f"{'='*80}\n")
                    break
            else:
                # Log the failed iteration
                error_msg = results.get("error", "Unknown error")
                print(f"Iteration {self.iteration} failed: {error_msg}")
                
                # Log a placeholder entry
                self.log_iteration(
                    pipeline_module, config, 999.0, 0.0, 0.0, 
                    {"error": error_msg, "success": False}
                )
            
            # Save checkpoint
            self.save_checkpoint()
            
            # Plot progress if in Colab
            if IS_COLAB and self.iteration % 3 == 0:
                self.plot_progress()
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"AutoTrainer finished after {self.iteration} iterations")
        print(f"Best MAE: {self.best_mae:.4f}")
        if self.best_config:
            print(f"Best pipeline: {self.best_config['pipeline']}")
            print(f"Best configuration: {json.dumps(self.best_config['config'], indent=2)}")
        print(f"{'='*80}\n")
        
        # Return the best result
        return {
            "best_mae": self.best_mae,
            "best_config": self.best_config,
            "iterations": self.iteration,
            "history": self.history
        }
    
    def plot_progress(self):
        """Plot the training progress so far."""
        if not self.history:
            return
        
        # Create plots directory if it doesn't exist
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Extract data for plotting
        iterations = [entry["iteration"] for entry in self.history]
        maes = [entry["mae"] for entry in self.history]
        r2s = [entry["r2"] for entry in self.history]
        times = [entry["training_time"] for entry in self.history]
        pipelines = [entry["pipeline"].split(".")[-1] for entry in self.history]
        
        # Create unique colors for each pipeline
        unique_pipelines = list(set(pipelines))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pipelines)))
        pipeline_colors = {pipe: colors[i] for i, pipe in enumerate(unique_pipelines)}
        
        # Plot MAE progress
        plt.figure(figsize=(10, 6))
        
        # Plot points colored by pipeline
        for pipe in unique_pipelines:
            pipe_indices = [i for i, p in enumerate(pipelines) if p == pipe]
            pipe_iterations = [iterations[i] for i in pipe_indices]
            pipe_maes = [maes[i] for i in pipe_indices]
            plt.scatter(pipe_iterations, pipe_maes, label=pipe, color=pipeline_colors[pipe], s=50, alpha=0.7)
        
        # Plot overall trend line
        if len(iterations) > 1:
            z = np.polyfit(iterations, maes, 1)
            p = np.poly1d(z)
            plt.plot(iterations, p(iterations), "r--", alpha=0.8, label="Trend")
        
        # Plot target line
        plt.axhline(y=self.target_mae, color='g', linestyle='-', alpha=0.5, label=f"Target ({self.target_mae})")
        
        plt.title('MAE Progress Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Absolute Error (cm)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_path = os.path.join(plot_dir, "mae_progress.png")
        plt.savefig(plot_path)
        
        if IS_COLAB:
            plt.show()
        else:
            plt.close()
        
        # Plot MAE vs Training Time scatter
        plt.figure(figsize=(10, 6))
        
        # Plot points colored by pipeline
        for pipe in unique_pipelines:
            pipe_indices = [i for i, p in enumerate(pipelines) if p == pipe]
            pipe_times = [times[i] for i in pipe_indices]
            pipe_maes = [maes[i] for i in pipe_indices]
            plt.scatter(pipe_times, pipe_maes, label=pipe, color=pipeline_colors[pipe], s=50, alpha=0.7)
        
        plt.title('MAE vs Training Time')
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Mean Absolute Error (cm)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_path = os.path.join(plot_dir, "mae_vs_time.png")
        plt.savefig(plot_path)
        
        if IS_COLAB:
            plt.show()
        else:
            plt.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Automated ML training for distance estimation")
    parser.add_argument("--target-mae", type=float, default=TARGET_MAE,
                        help=f"Target MAE to achieve (default: {TARGET_MAE})")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                        help=f"Maximum number of iterations (default: {MAX_ITERATIONS})")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Disable saving checkpoints")
    parser.add_argument("--checkpoint-file", type=str, default=CHECKPOINT_FILE,
                        help=f"Path to checkpoint file (default: {CHECKPOINT_FILE})")
    parser.add_argument("--log-file", type=str, default=LOG_FILE,
                        help=f"Path to log file (default: {LOG_FILE})")
    return parser.parse_args()


if __name__ == "__main__":
    # Handle Google Colab-specific setup
    if IS_COLAB:
        # Mount Google Drive if not already mounted
        if not os.path.exists("/content/drive"):
            print("Mounting Google Drive...")
            drive.mount("/content/drive")
            print("Google Drive mounted")
        
        # Check if we need to cd to the project directory
        if not os.path.exists("config.yaml"):
            # Try to find the project directory
            possible_dirs = [
                "/content/drive/MyDrive/SML_Project1_ANYmal",
                "/content/drive/MyDrive/Colab Notebooks/SML_Project1_ANYmal",
                "/content/SML_Project1_ANYmal"
            ]
            
            for directory in possible_dirs:
                if os.path.exists(os.path.join(directory, "config.yaml")):
                    print(f"Changing to project directory: {directory}")
                    os.chdir(directory)
                    break
            else:
                print("Warning: Could not find project directory with config.yaml")
    
    # Install required packages if not already installed
    try:
        import tqdm
        import matplotlib
        import yaml
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.call(["pip", "install", "tqdm", "matplotlib", "pyyaml"])
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize and run the auto trainer
    trainer = AutoTrainer(
        target_mae=args.target_mae,
        max_iterations=args.max_iterations,
        save_checkpoints=not args.no_checkpoints,
        checkpoint_file=args.checkpoint_file,
        log_file=args.log_file
    )
    
    # Run the trainer
    results = trainer.run()
    
    # Print final summary
    print("\nTraining completed!")
    print(f"Best MAE: {results['best_mae']:.4f}")
    print(f"Total iterations: {results['iterations']}")
    
    # Save final results
    with open("auto_trainer_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to auto_trainer_results.json")