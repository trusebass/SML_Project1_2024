#!/usr/bin/env python3
"""
Specialized LightGBM Auto-Trainer for Distance Estimation

This script automates the process of optimizing LightGBM models with different
configurations. It focuses exclusively on refining LightGBM parameters to achieve
the target MAE, using insights from previous training runs.

Usage:
    python auto_trainer_lgbm.py --target-mae 0.08 --max-iterations 50

Features:
- Advanced parameter space exploration focused on LightGBM
- Specialized adaptive parameter adjustment strategy
- Automatic refinement of promising configurations
- Detailed logging and visualization of the optimization process
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
MAX_ITERATIONS = 50  # Default maximum number of iterations
SAVE_CHECKPOINTS = True  # Whether to save checkpoints
CHECKPOINT_FILE = "lgbm_trainer_checkpoint.json"
LOG_FILE = "lgbm_trainer_log.json"

class LGBMAutoTrainer:
    """
    Specialized automated training system for LightGBM models with
    adaptive parameter refinement to achieve the target MAE.
    """
    
    def __init__(self, target_mae: float = TARGET_MAE, 
                 max_iterations: int = MAX_ITERATIONS,
                 save_checkpoints: bool = SAVE_CHECKPOINTS,
                 checkpoint_file: str = CHECKPOINT_FILE,
                 log_file: str = LOG_FILE):
        """
        Initialize the LGBMAutoTrainer.
        
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
        self.current_phase = "exploration"  # exploration, refinement, or exploitation
        self.phase_transitions = {
            # Transition from exploration to refinement
            "exploration_to_refinement": max_iterations // 3,
            # Transition from refinement to exploitation
            "refinement_to_exploitation": max_iterations // 3 * 2
        }
        
        # Only use the simple_lgbm pipeline
        self.pipeline_module = "MainScripts.simple_lgbm"
        
        # LightGBM-specific parameter space
        self.init_param_space()
        
        # Parameter combination blacklist to avoid repeating exactly same configs
        self.config_blacklist = set()
        
        # Load checkpoint if it exists
        if save_checkpoints and os.path.exists(checkpoint_file):
            self.load_checkpoint()
    
    def init_param_space(self):
        """Initialize LightGBM-specific parameter spaces with refined ranges."""
        # Define initial parameter space for LightGBM
        self.param_space = {
            "load_rgb": [True, False],
            "downsample_factor": [1, 2, 3, 4, 6, 8, 10],
            "model_params": {
                # Core parameters
                "n_estimators": [100, 200, 300, 500, 800, 1000, 1500],
                "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9, 11, 15, -1],  # -1 means no limit
                
                # Leaf-wise parameters (LightGBM specialty)
                "num_leaves": [31, 63, 127, 255, 511],
                "min_data_in_leaf": [5, 10, 20, 50, 100],
                "min_child_samples": [5, 10, 20, 50, 100],
                
                # Randomness parameters
                "feature_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
                "bagging_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
                "bagging_freq": [0, 1, 5, 10],
                
                # Regularization parameters
                "lambda_l1": [0, 0.001, 0.01, 0.1, 1.0, 10.0],  # L1 regularization
                "lambda_l2": [0, 0.001, 0.01, 0.1, 1.0, 10.0],  # L2 regularization
                "min_gain_to_split": [0, 0.1, 0.5],
                
                # Categorical feature parameters
                "max_cat_threshold": [16, 32, 64],
                "cat_smooth": [1, 10, 50],
                "cat_l2": [1, 10],
                
                # Other parameters
                "objective": ['regression', 'regression_l1', 'huber', 'fair', 'poisson'],
                "boosting": ['gbdt', 'dart', 'goss'],
                "metric": ['mae', 'mse', 'rmse', 'l1', 'l2'],
                "verbosity": [-1],  # Disable LightGBM verbosity
                
                # LightGBM specific speed-ups
                "force_row_wise": [True, False],
                "use_missing": [True, False],
                "zero_as_missing": [True, False]
            }
        }
        
        # Phase-specific parameter spaces (will be refined during training)
        self.refinement_space = None  # Will be created based on best performers
        self.exploitation_space = None  # Will be created based on best performers
        
        # Special adjustments for Colab environment
        if IS_COLAB:
            # Optimization: Reduce parameter space for faster iteration in Colab
            if "downsample_factor" in self.param_space:
                # Prefer downsample factors that work well on limited resources
                self.param_space["downsample_factor"] = [2, 4, 6, 10]
            
            # Limit max_depth to avoid memory issues
            self.param_space["model_params"]["max_depth"] = [3, 5, 7, -1]
            
            # Limit num_leaves for better memory efficiency
            self.param_space["model_params"]["num_leaves"] = [31, 63, 127]
    
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
            self.refinement_space = checkpoint.get("refinement_space", None)
            self.exploitation_space = checkpoint.get("exploitation_space", None)
            
            # Reconstruct config blacklist
            self.config_blacklist = set()
            for entry in self.history:
                config_hash = self._get_config_hash(entry["config"])
                self.config_blacklist.add(config_hash)
            
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
            "refinement_space": self.refinement_space,
            "exploitation_space": self.exploitation_space,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=4)
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
    
    def log_iteration(self, config, mae, r2, training_time, additional_info=None):
        """
        Log information about the current iteration.
        
        Args:
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
            "pipeline": self.pipeline_module,
            "config": config,
            "mae": float(mae),
            "r2": float(r2),
            "training_time": float(training_time),
            "best_mae_so_far": float(self.best_mae),
            "phase": self.current_phase
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
    
    def _get_config_hash(self, config):
        """Generate a simple hash for a config to check for duplicates."""
        # Convert config to a hashable representation
        config_str = json.dumps(config, sort_keys=True)
        return hash(config_str)
    
    def update_current_phase(self):
        """Update the current phase based on iteration count and performance."""
        # Check for phase transitions based on iteration count
        if self.iteration >= self.phase_transitions["refinement_to_exploitation"]:
            if self.current_phase != "exploitation":
                print(f"Transitioning to exploitation phase at iteration {self.iteration}")
                self.current_phase = "exploitation"
                self._create_exploitation_space()
        elif self.iteration >= self.phase_transitions["exploration_to_refinement"]:
            if self.current_phase != "refinement" and self.current_phase != "exploitation":
                print(f"Transitioning to refinement phase at iteration {self.iteration}")
                self.current_phase = "refinement"
                self._create_refinement_space()
    
    def _create_refinement_space(self):
        """Create a refined parameter space based on best performers."""
        if not self.history or len(self.history) < 5:
            # Not enough history to refine
            return
        
        # Sort history by MAE (ascending)
        sorted_history = sorted(self.history, key=lambda x: x["mae"])
        
        # Take top N configurations
        top_n = min(5, len(sorted_history))
        top_configs = sorted_history[:top_n]
        
        # Initialize refined parameter space
        refined_space = {
            "load_rgb": set(),
            "downsample_factor": set(),
            "model_params": defaultdict(set)
        }
        
        # Collect parameter values from top performers
        for entry in top_configs:
            config = entry["config"]
            
            # Basic params
            refined_space["load_rgb"].add(config.get("load_rgb", True))
            refined_space["downsample_factor"].add(config.get("downsample_factor", 2))
            
            # Model params
            model_params = config.get("model_params", {})
            for param, value in model_params.items():
                refined_space["model_params"][param].add(value)
        
        # Convert sets to lists
        self.refinement_space = {
            "load_rgb": list(refined_space["load_rgb"]),
            "downsample_factor": list(refined_space["downsample_factor"]),
            "model_params": {k: list(v) for k, v in refined_space["model_params"].items()}
        }
        
        # Add additional values around the best performers for exploration
        # This helps avoid getting stuck in local minima
        self._expand_refinement_space()
        
        print(f"Created refinement parameter space with {len(self.refinement_space['model_params'])} parameters")
    
    def _expand_refinement_space(self):
        """Expand the refinement space with values around the best performers."""
        if not self.refinement_space:
            return
        
        # Add nearby integer values for relevant parameters
        for param in ["n_estimators", "max_depth", "num_leaves", "min_data_in_leaf", "min_child_samples"]:
            if param in self.refinement_space["model_params"]:
                values = self.refinement_space["model_params"][param]
                expanded = set(values)
                
                for val in values:
                    if isinstance(val, int) and val not in [-1, 0]:  # Skip special values
                        # Add nearby values (slightly bigger and smaller)
                        if val > 1:
                            expanded.add(int(val * 0.8))
                        expanded.add(int(val * 1.2) + 1)
                
                self.refinement_space["model_params"][param] = list(expanded)
        
        # Add nearby float values for relevant parameters
        for param in ["learning_rate", "feature_fraction", "bagging_fraction", 
                     "lambda_l1", "lambda_l2"]:
            if param in self.refinement_space["model_params"]:
                values = self.refinement_space["model_params"][param]
                expanded = set(values)
                
                for val in values:
                    if isinstance(val, (float, int)) and val is not None:
                        # Add nearby values
                        if val > 0:
                            expanded.add(max(0, val * 0.5))
                        expanded.add(min(1.0 if param in ["feature_fraction", "bagging_fraction"] else 100.0, val * 1.5))
                
                self.refinement_space["model_params"][param] = list(expanded)
        
        # Ensure we have at least the original options for categorical parameters
        for param in ["objective", "boosting", "metric"]:
            if param in self.refinement_space["model_params"]:
                continue  # Already has values from top performers
            
            # Add the original options
            if param in self.param_space["model_params"]:
                self.refinement_space["model_params"][param] = self.param_space["model_params"][param]
    
    def _create_exploitation_space(self):
        """Create an exploitation parameter space focused on the very best performers."""
        if not self.history or len(self.history) < 10:
            # Not enough history, use refinement space if available
            if self.refinement_space:
                self.exploitation_space = self.refinement_space
            return
        
        # Sort history by MAE (ascending)
        sorted_history = sorted(self.history, key=lambda x: x["mae"])
        
        # Take top N configurations
        top_n = min(3, len(sorted_history))
        top_configs = sorted_history[:top_n]
        
        # Initialize exploitation parameter space
        exploitation_space = {
            "load_rgb": set(),
            "downsample_factor": set(),
            "model_params": defaultdict(set)
        }
        
        # Collect parameter values from top performers
        for entry in top_configs:
            config = entry["config"]
            
            # Basic params
            exploitation_space["load_rgb"].add(config.get("load_rgb", True))
            exploitation_space["downsample_factor"].add(config.get("downsample_factor", 2))
            
            # Model params
            model_params = config.get("model_params", {})
            for param, value in model_params.items():
                exploitation_space["model_params"][param].add(value)
        
        # Convert sets to lists
        self.exploitation_space = {
            "load_rgb": list(exploitation_space["load_rgb"]),
            "downsample_factor": list(exploitation_space["downsample_factor"]),
            "model_params": {k: list(v) for k, v in exploitation_space["model_params"].items()}
        }
        
        # For exploitation, add very fine-grained variations
        self._add_exploitation_variations()
        
        print(f"Created exploitation parameter space with {len(self.exploitation_space['model_params'])} parameters")
    
    def _add_exploitation_variations(self):
        """Add very fine-grained variations to the exploitation space."""
        if not self.exploitation_space:
            return
        
        # For n_estimators, add more options around the best
        if "n_estimators" in self.exploitation_space["model_params"]:
            values = self.exploitation_space["model_params"]["n_estimators"]
            best_value = max(values)  # Usually more trees is better
            variations = set(values)
            
            # Add fine-grained variations
            steps = [0.9, 0.95, 1.05, 1.1, 1.2]
            for step in steps:
                new_value = int(best_value * step)
                if new_value > 0:
                    variations.add(new_value)
            
            self.exploitation_space["model_params"]["n_estimators"] = list(variations)
        
        # For learning_rate, add more fine-grained options
        if "learning_rate" in self.exploitation_space["model_params"]:
            values = self.exploitation_space["model_params"]["learning_rate"]
            variations = set(values)
            
            # Get the min and max values
            min_val = min(values)
            max_val = max(values)
            
            # Add values in between
            for i in range(1, 4):
                new_val = min_val + (max_val - min_val) * (i / 4)
                variations.add(round(new_val, 4))
            
            # Add values slightly outside the range
            if min_val > 0.001:
                variations.add(round(min_val * 0.7, 4))
            if max_val < 0.5:
                variations.add(round(max_val * 1.3, 4))
            
            self.exploitation_space["model_params"]["learning_rate"] = list(variations)
        
        # For num_leaves, add more options to fine-tune the tree structure
        if "num_leaves" in self.exploitation_space["model_params"]:
            values = self.exploitation_space["model_params"]["num_leaves"]
            variations = set(values)
            
            # LightGBM often performs well with powers of 2 minus 1
            for i in range(4, 10):  # 2^4-1 to 2^10-1
                leaves = (2 ** i) - 1
                variations.add(leaves)
            
            # Also add some values around the existing ones
            for val in values:
                if val > 10:
                    variations.add(val - 5)
                    variations.add(val + 5)
            
            self.exploitation_space["model_params"]["num_leaves"] = list(variations)
        
        # For feature_fraction and bagging_fraction, add more fine-grained options
        for param in ["feature_fraction", "bagging_fraction"]:
            if param in self.exploitation_space["model_params"]:
                values = self.exploitation_space["model_params"][param]
                variations = set(values)
                
                # Add more fine-grained options
                for val in values:
                    variations.add(round(max(0.5, val - 0.05), 2))
                    variations.add(round(min(1.0, val + 0.05), 2))
                
                self.exploitation_space["model_params"][param] = list(variations)
    
    def select_next_config(self):
        """
        Intelligently select the next configuration to try based on
        current phase and previous results.
        
        Returns:
            Config dictionary
        """
        # Update the current phase
        self.update_current_phase()
        
        # Select configuration based on current phase
        if self.current_phase == "exploration":
            return self._explore_random_config()
        elif self.current_phase == "refinement":
            if self.refinement_space and random.random() < 0.8:
                return self._select_from_refined_space()
            else:
                return self._explore_random_config()
        else:  # exploitation
            if self.exploitation_space and random.random() < 0.9:
                return self._select_from_exploitation_space()
            elif self.refinement_space:
                return self._select_from_refined_space()
            else:
                return self._explore_random_config()
    
    def _explore_random_config(self):
        """Randomly select a configuration for exploration."""
        # Generate a random configuration
        config = {}
        
        # Handle basic configuration parameters
        for key, values in self.param_space.items():
            if key != "model_params":
                config[key] = random.choice(values)
        
        # Handle model parameters
        model_params = {}
        
        # Randomly select a subset of parameters to avoid too complex models
        all_params = list(self.param_space["model_params"].keys())
        # Always include core parameters
        core_params = ["n_estimators", "max_depth", "learning_rate", "num_leaves"]
        # Randomly select some additional parameters
        additional_params = [p for p in all_params if p not in core_params]
        selected_additional = random.sample(
            additional_params,
            k=min(len(additional_params), random.randint(3, 7))
        )
        selected_params = core_params + selected_additional
        
        # Set values for selected parameters
        for param in selected_params:
            values = self.param_space["model_params"][param]
            model_params[param] = random.choice(values)
        
        config["model_params"] = model_params
        
        # Ensure parameter compatibility
        self._ensure_param_compatibility(config["model_params"])
        
        # Check if this exact config has been tried before
        config_hash = self._get_config_hash(config)
        if config_hash in self.config_blacklist:
            # Try again with a different random configuration
            return self._explore_random_config()
        
        # Add to blacklist
        self.config_blacklist.add(config_hash)
        
        return config
    
    def _ensure_param_compatibility(self, model_params):
        """Ensure parameter compatibility for LightGBM."""
        # Make sure num_leaves is compatible with max_depth if both are specified
        if "num_leaves" in model_params and "max_depth" in model_params:
            max_depth = model_params["max_depth"]
            num_leaves = model_params["num_leaves"]
            
            # If max_depth is not -1 (no limit), ensure num_leaves is consistent
            if max_depth != -1:
                # In a balanced binary tree, max leaves for depth d is 2^d
                max_possible_leaves = 2 ** max_depth
                if num_leaves > max_possible_leaves:
                    # Reduce num_leaves to be compatible with max_depth
                    model_params["num_leaves"] = max_possible_leaves
        
        # Make sure the boosting type is compatible with other parameters
        if "boosting" in model_params:
            boosting = model_params["boosting"]
            
            # GOSS (Gradient-based One-Side Sampling) doesn't use bagging
            if boosting == "goss" and "bagging_fraction" in model_params:
                del model_params["bagging_fraction"]
                if "bagging_freq" in model_params:
                    del model_params["bagging_freq"]
    
    def _select_from_refined_space(self):
        """Select a configuration from the refined parameter space."""
        if not self.refinement_space:
            return self._explore_random_config()
        
        # Generate a configuration from the refined space
        config = {}
        
        # Handle basic configuration parameters
        for key, values in self.refinement_space.items():
            if key != "model_params":
                config[key] = random.choice(values)
        
        # Handle model parameters
        model_params = {}
        
        # Include core parameters and randomly select additional ones
        all_params = list(self.refinement_space["model_params"].keys())
        core_params = ["n_estimators", "max_depth", "learning_rate", "num_leaves"]
        core_params = [p for p in core_params if p in all_params]
        
        # Randomly select some additional parameters
        additional_params = [p for p in all_params if p not in core_params]
        selected_additional = random.sample(
            additional_params,
            k=min(len(additional_params), random.randint(len(additional_params)//2, len(additional_params)))
        )
        selected_params = core_params + selected_additional
        
        # Set values for selected parameters
        for param in selected_params:
            values = self.refinement_space["model_params"][param]
            model_params[param] = random.choice(values)
        
        config["model_params"] = model_params
        
        # Ensure parameter compatibility
        self._ensure_param_compatibility(config["model_params"])
        
        # Check if this exact config has been tried before
        config_hash = self._get_config_hash(config)
        if config_hash in self.config_blacklist:
            # With 50% chance, try again with a different configuration
            if random.random() < 0.5:
                return self._select_from_refined_space()
            # Otherwise, make a small variation to the config
            for param in model_params:
                if param in ["n_estimators", "max_depth", "num_leaves"] and model_params[param] not in [-1, 0]:
                    # Small variation to numeric parameters
                    model_params[param] += random.choice([-1, 1])
                    if model_params[param] <= 0 and param == "n_estimators":
                        model_params[param] = 1
        
        # Add to blacklist
        self.config_blacklist.add(config_hash)
        
        return config
    
    def _select_from_exploitation_space(self):
        """Select a configuration from the exploitation parameter space."""
        if not self.exploitation_space:
            if self.refinement_space:
                return self._select_from_refined_space()
            return self._explore_random_config()
        
        # Generate a configuration from the exploitation space
        config = {}
        
        # Handle basic configuration parameters
        for key, values in self.exploitation_space.items():
            if key != "model_params":
                config[key] = random.choice(values)
        
        # Handle model parameters
        model_params = {}
        
        # In exploitation phase, try to include most parameters
        all_params = list(self.exploitation_space["model_params"].keys())
        
        # Set values for parameters
        for param in all_params:
            values = self.exploitation_space["model_params"][param]
            model_params[param] = random.choice(values)
        
        config["model_params"] = model_params
        
        # Ensure parameter compatibility
        self._ensure_param_compatibility(config["model_params"])
        
        # Check if this exact config has been tried before
        config_hash = self._get_config_hash(config)
        if config_hash in self.config_blacklist:
            # With 70% chance, make a small variation to avoid repetition
            if random.random() < 0.7:
                # Make a small random variation to one parameter
                param_to_vary = random.choice(list(model_params.keys()))
                
                if param_to_vary in ["n_estimators", "max_depth", "num_leaves"] and model_params[param_to_vary] not in [-1, 0]:
                    # Small variation to numeric parameters
                    variation = random.randint(1, 5)
                    sign = random.choice([-1, 1])
                    model_params[param_to_vary] += sign * variation
                    if model_params[param_to_vary] <= 0 and param_to_vary == "n_estimators":
                        model_params[param_to_vary] = 1
                elif param_to_vary in ["learning_rate", "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2"]:
                    # Small variation to float parameters
                    variation = random.uniform(0.01, 0.05)
                    sign = random.choice([-1, 1])
                    model_params[param_to_vary] = max(0.001, model_params[param_to_vary] + sign * variation)
                    if param_to_vary in ["feature_fraction", "bagging_fraction"]:
                        model_params[param_to_vary] = min(1.0, model_params[param_to_vary])
            else:
                # Try a completely different config from the exploitation space
                return self._select_from_exploitation_space()
        
        # Add to blacklist
        self.config_blacklist.add(config_hash)
        
        return config
    
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
    
    def run_training_iteration(self, config):
        """
        Run a single training iteration with the specified configuration.
        
        Args:
            config: Configuration to use
            
        Returns:
            Dictionary with results including MAE, R2, and training time
        """
        print(f"\n{'='*80}")
        print(f"Starting iteration {self.iteration+1}/{self.max_iterations} ({self.current_phase} phase)")
        print(f"Pipeline: {self.pipeline_module}")
        print(f"Configuration: {json.dumps(config, indent=2)}")
        print(f"{'='*80}\n")
        
        # Update config file
        self.update_config_file(config)
        
        try:
            # Import the pipeline module dynamically
            module = importlib.import_module(self.pipeline_module)
            
            # Extract the model parameters
            model_params = config.get("model_params", {})
            
            # Record start time
            start_time = time.time()
            
            # Run the pipeline's main function
            if hasattr(module, "main"):
                result = module.main(model_params=model_params)
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
            if isinstance(result, tuple) and len(result) >= 2:
                # If main returned (mae, r2, time)
                mae = result[0]
                r2 = result[1]
                
                return {
                    "mae": mae,
                    "r2": r2,
                    "training_time": training_time,
                    "success": True
                }
            else:
                # Fallback with placeholder values
                print("Warning: Could not get results from log file or return value, using placeholder values")
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
        print(f"Starting LGBMAutoTrainer with target MAE: {self.target_mae}")
        print(f"Maximum iterations: {self.max_iterations}")
        
        # Loop until we reach the target or max iterations
        while self.iteration < self.max_iterations:
            # Increment iteration counter
            self.iteration += 1
            
            # Select the next configuration to try
            config = self.select_next_config()
            
            # Run the training iteration
            results = self.run_training_iteration(config)
            
            # Check if the iteration was successful
            if results["success"]:
                mae = results["mae"]
                r2 = results["r2"]
                training_time = results["training_time"]
                additional_info = results.get("additional_info", {})
                
                # Log the iteration
                self.log_iteration(
                    config, mae, r2, training_time, additional_info
                )
                
                # Update best MAE if this iteration is better
                if mae < self.best_mae:
                    improvement = self.best_mae - mae
                    improvement_percent = (improvement / self.best_mae) * 100 if self.best_mae != float('inf') else 0
                    
                    self.best_mae = mae
                    self.best_config = config.copy()
                    
                    # Print improvement information
                    if self.best_mae != float('inf'):
                        print(f"New best MAE: {self.best_mae:.4f} (improved by {improvement:.4f}, {improvement_percent:.2f}%)")
                    else:
                        print(f"New best MAE: {self.best_mae:.4f}")
                    
                    # Trigger parameter space updates if significant improvement
                    if improvement_percent > 10 and self.current_phase == "refinement":
                        print("Significant improvement detected! Updating refinement space.")
                        self._create_refinement_space()
                    elif improvement_percent > 5 and self.current_phase == "exploitation":
                        print("Significant improvement detected! Updating exploitation space.")
                        self._create_exploitation_space()
                
                # Print summary
                print(f"\nIteration {self.iteration} summary:")
                print(f"MAE: {mae:.4f}")
                print(f"R²: {r2:.4f}")
                print(f"Training time: {training_time:.2f} seconds")
                print(f"Best MAE so far: {self.best_mae:.4f}")
                print(f"Current phase: {self.current_phase}")
                
                # Check if we've reached the target
                if mae <= self.target_mae:
                    print(f"\n{'='*80}")
                    print(f"Target MAE of {self.target_mae} reached!")
                    print(f"Final MAE: {mae:.4f}")
                    print(f"Configuration: {json.dumps(config, indent=2)}")
                    print(f"{'='*80}\n")
                    break
            else:
                # Log the failed iteration
                error_msg = results.get("error", "Unknown error")
                print(f"Iteration {self.iteration} failed: {error_msg}")
                
                # Log a placeholder entry
                self.log_iteration(
                    config, 999.0, 0.0, 0.0, 
                    {"error": error_msg, "success": False}
                )
            
            # Save checkpoint
            self.save_checkpoint()
            
            # Plot progress every few iterations
            if self.iteration % 5 == 0 or self.iteration == 1:
                self.plot_progress()
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"LGBMAutoTrainer finished after {self.iteration} iterations")
        print(f"Best MAE: {self.best_mae:.4f}")
        if self.best_config:
            print(f"Best configuration: {json.dumps(self.best_config, indent=2)}")
        print(f"{'='*80}\n")
        
        # Final plot
        self.plot_progress()
        
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
        phases = [entry["phase"] if "phase" in entry else "unknown" for entry in self.history]
        
        # Create unique colors for each phase
        unique_phases = list(set(phases))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_phases)))
        phase_colors = {phase: colors[i] for i, phase in enumerate(unique_phases)}
        
        # Plot MAE progress
        plt.figure(figsize=(12, 7))
        
        # Plot points colored by phase
        for phase in unique_phases:
            phase_indices = [i for i, p in enumerate(phases) if p == phase]
            phase_iterations = [iterations[i] for i in phase_indices]
            phase_maes = [maes[i] for i in phase_indices]
            
            # Skip points with excessive MAE for better visualization
            valid_indices = [(idx, mae) for idx, mae in enumerate(phase_maes) if mae < 10]
            if not valid_indices:
                continue
                
            valid_idx = [idx for idx, _ in valid_indices]
            valid_maes = [mae for _, mae in valid_indices]
            valid_iterations = [phase_iterations[idx] for idx in valid_idx]
            
            plt.scatter(valid_iterations, valid_maes, 
                       label=f"{phase.capitalize()}", 
                       color=phase_colors[phase], 
                       s=50, alpha=0.7)
        
        # Plot overall trend line
        valid_maes = [mae for mae in maes if mae < 10]
        valid_iterations = [iterations[i] for i, mae in enumerate(maes) if mae < 10]
        
        if len(valid_iterations) > 1:
            z = np.polyfit(valid_iterations, valid_maes, 1)
            p = np.poly1d(z)
            trend_x = range(min(valid_iterations), max(valid_iterations) + 1)
            plt.plot(trend_x, p(trend_x), "r--", alpha=0.8, label="Trend")
        
        # Plot target line
        plt.axhline(y=self.target_mae, color='g', linestyle='-', alpha=0.5, 
                   label=f"Target ({self.target_mae})")
        
        # Indicate phase transition points
        for phase_name, iteration in self.phase_transitions.items():
            if iteration <= max(iterations):
                plt.axvline(x=iteration, color='gray', linestyle='--', alpha=0.5)
                plt.text(iteration, max(valid_maes) * 0.9, phase_name.replace('_', ' ').title(), 
                        rotation=90, verticalalignment='top')
        
        plt.title('LightGBM MAE Progress Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Absolute Error (m)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Limit y-axis for better visualization of improvements
        max_y = min(1.0, max(valid_maes) * 1.1)
        min_y = max(0, min(valid_maes) * 0.9)
        plt.ylim(min_y, max_y)
        
        plot_path = os.path.join(plot_dir, "lgbm_mae_progress.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Plot MAE vs Training Time scatter
        plt.figure(figsize=(12, 7))
        
        # Remove any extremely high values for better visualization
        valid_indices = [(i, mae) for i, mae in enumerate(maes) if mae < 10]
        valid_idx = [idx for idx, _ in valid_indices]
        valid_maes = [mae for _, mae in valid_indices]
        valid_times = [times[idx] for idx in valid_idx]
        valid_phases = [phases[idx] for idx in valid_idx]
        
        # Plot points colored by phase
        for phase in unique_phases:
            phase_indices = [i for i, p in enumerate(valid_phases) if p == phase]
            phase_times = [valid_times[i] for i in phase_indices]
            phase_maes = [valid_maes[i] for i in phase_indices]
            
            plt.scatter(phase_times, phase_maes, 
                       label=f"{phase.capitalize()}", 
                       color=phase_colors[phase], 
                       s=50, alpha=0.7)
        
        plt.title('LightGBM MAE vs Training Time')
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Mean Absolute Error (m)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Limit y-axis for better visualization
        plt.ylim(min_y, max_y)
        
        plot_path = os.path.join(plot_dir, "lgbm_mae_vs_time.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Plot parameter relationships to MAE if enough data
        if len(valid_maes) > 5:
            self._plot_parameter_relationships()
    
    def _plot_parameter_relationships(self):
        """Plot the relationship between key parameters and MAE."""
        if not self.history:
            return
        
        # Create plots directory if it doesn't exist
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Filter out failed runs and excessive MAE values for better visualization
        valid_history = [entry for entry in self.history 
                        if entry.get("mae", 999) < 10 and "config" in entry]
        
        if len(valid_history) < 5:
            return
        
        # Extract key parameters and their values
        key_params = ["n_estimators", "max_depth", "learning_rate", 
                      "num_leaves", "feature_fraction", "lambda_l1", "lambda_l2"]
        
        param_values = defaultdict(list)
        maes = []
        
        for entry in valid_history:
            maes.append(entry["mae"])
            
            for param in key_params:
                value = entry.get("config", {}).get("model_params", {}).get(param, None)
                param_values[param].append(value)
        
        # Plot each parameter vs MAE
        for param in key_params:
            # Skip parameters with insufficient data
            values = param_values[param]
            if len(values) < 5 or all(v is None for v in values):
                continue
            
            # Convert None to -1 for plotting
            plot_values = [v if v is not None else -1 for v in values]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(plot_values, maes, s=50, alpha=0.7)
            
            # Add labels for None values
            if -1 in plot_values:
                none_indices = [i for i, v in enumerate(plot_values) if v == -1]
                none_maes = [maes[i] for i in none_indices]
                plt.scatter([-1] * len(none_indices), none_maes, label="None", marker='x')
                
            plt.title(f'LightGBM {param} vs MAE')
            plt.xlabel(param)
            plt.ylabel('Mean Absolute Error (m)')
            plt.grid(True, alpha=0.3)
            
            if -1 in plot_values:
                plt.legend()
            
            # Add trend line if possible
            if all(v != -1 for v in plot_values):
                try:
                    z = np.polyfit(plot_values, maes, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(plot_values), max(plot_values), 100)
                    plt.plot(x_range, p(x_range), "r--", alpha=0.8)
                except:
                    pass  # Skip trend line if there's an error
            
            plot_path = os.path.join(plot_dir, f"lgbm_{param}_vs_mae.png")
            plt.savefig(plot_path)
            plt.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Specialized LightGBM trainer for distance estimation")
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
        import matplotlib
        import yaml
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.call(["pip", "install", "matplotlib", "pyyaml"])
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize and run the LGBM auto trainer
    trainer = LGBMAutoTrainer(
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
    with open("lgbm_trainer_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to lgbm_trainer_results.json")