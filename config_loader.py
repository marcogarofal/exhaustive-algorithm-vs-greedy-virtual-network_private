# config_loader.py
"""
Configuration Loader - Handles loading and saving configuration files
"""

import json
import os


def load_config(config_path='config.json'):
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        dict with configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Convert string keys to integers for capacities if needed
    if 'capacities' in config and isinstance(config['capacities'], dict):
        if not isinstance(config['capacities'], dict) or 'default' not in config['capacities']:
            config['capacities'] = {int(k): v for k, v in config['capacities'].items()}

    return config


def save_config(config, config_path='config.json'):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        config_path: Path where to save configuration
    """
    # Convert integer keys to strings for JSON compatibility
    config_copy = config.copy()
    if 'capacities' in config_copy and isinstance(config_copy['capacities'], dict):
        config_copy['capacities'] = {str(k): v for k, v in config_copy['capacities'].items()}

    with open(config_path, 'w') as f:
        json.dump(config_copy, f, indent=2)

    print(f"Configuration saved to {config_path}")


def create_default_config(output_path='config.json'):
    """
    Create a default configuration file.

    Args:
        output_path: Path where to save the default configuration
    """
    default_config = {
        "graph_parameters": {
            "num_nodes": 8,
            "weak_ratio": 0.4,
            "mandatory_ratio": 0.2,
            "seed": 42,
            "comment": "Can also use: num_weak, num_mandatory, num_discretionary instead of num_nodes+ratios"
        },
        "capacities": {
            "default": 10,
            "custom": {
                "2": 30,
                "3": 2,
                "4": 1
            },
            "comment": "Options: {default: X}, {default: X, custom: {...}}, {random: {min: X, max: Y, seed: Z}}, or explicit {1: X, 2: Y, ...}"
        },
        "debug": {
            "plot_initial_graphs": False,
            "plot_intermediate": False,
            "plot_final": True,
            "save_plots": True,
            "verbose": False,
            "verbose_level2": False,
            "verbose_level3": False
        },
        "algorithm": {
            "weight_range": [1, 10]
        },
        "output": {
            "plots_dir": "plots",
            "results_dir": "results"
        }
    }

    save_config(default_config, output_path)
    return default_config


def validate_config(config):
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Returns:
        tuple (is_valid, error_messages)
    """
    errors = []

    # Check required sections
    required_sections = ['graph_parameters', 'capacities']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Validate graph_parameters
    if 'graph_parameters' in config:
        gp = config['graph_parameters']

        # Must have either num_nodes+ratios or absolute numbers
        has_ratios = 'num_nodes' in gp and 'weak_ratio' in gp and 'mandatory_ratio' in gp
        has_absolute = 'num_weak' in gp and 'num_mandatory' in gp and 'num_discretionary' in gp

        if not (has_ratios or has_absolute):
            errors.append("graph_parameters must have either (num_nodes + ratios) or (absolute node counts)")

        # Validate ratios if present
        if has_ratios:
            if not (0 <= gp['weak_ratio'] <= 1):
                errors.append("weak_ratio must be between 0 and 1")
            if not (0 <= gp['mandatory_ratio'] <= 1):
                errors.append("mandatory_ratio must be between 0 and 1")
            if gp['weak_ratio'] + gp['mandatory_ratio'] > 1:
                errors.append("Sum of weak_ratio and mandatory_ratio cannot exceed 1")

    # Validate capacities
    if 'capacities' in config:
        cap = config['capacities']
        if not isinstance(cap, dict):
            errors.append("capacities must be a dictionary")
        else:
            # Check different capacity formats
            has_default = 'default' in cap
            has_random = 'random' in cap
            has_custom = 'custom' in cap
            has_explicit = any(k not in ['default', 'random', 'custom', 'comment'] for k in cap.keys())

            if has_random:
                # Validate random config
                random_cfg = cap['random']
                if not isinstance(random_cfg, dict):
                    errors.append("capacities.random must be a dictionary")
                else:
                    if 'min' not in random_cfg or 'max' not in random_cfg:
                        errors.append("capacities.random must have 'min' and 'max' keys")
                    elif random_cfg['min'] >= random_cfg['max']:
                        errors.append("capacities.random.min must be less than max")

            if has_explicit and (has_default or has_random):
                errors.append("Cannot mix explicit node capacities with default/random format")

            # Check if all capacity values are positive
            for node, capacity in cap.items():
                if node not in ['default', 'random', 'custom', 'comment']:
                    if isinstance(capacity, (int, float)) and capacity <= 0:
                        errors.append(f"Capacity for node {node} must be positive")

            if has_custom:
                for node, capacity in cap.get('custom', {}).items():
                    if isinstance(capacity, (int, float)) and capacity <= 0:
                        errors.append(f"Custom capacity for node {node} must be positive")

    return len(errors) == 0, errors


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'create':
            # Create default config
            output = sys.argv[2] if len(sys.argv) > 2 else 'config.json'
            config = create_default_config(output)
            print(f"Created default configuration at {output}")
            print(json.dumps(config, indent=2))

        elif command == 'validate':
            # Validate existing config
            config_path = sys.argv[2] if len(sys.argv) > 2 else 'config.json'
            try:
                config = load_config(config_path)
                is_valid, errors = validate_config(config)

                if is_valid:
                    print(f"✓ Configuration at {config_path} is valid")
                else:
                    print(f"✗ Configuration at {config_path} has errors:")
                    for error in errors:
                        print(f"  - {error}")
            except Exception as e:
                print(f"Error loading configuration: {e}")

        elif command == 'show':
            # Show config
            config_path = sys.argv[2] if len(sys.argv) > 2 else 'config.json'
            try:
                config = load_config(config_path)
                print(json.dumps(config, indent=2))
            except Exception as e:
                print(f"Error loading configuration: {e}")
        else:
            print(f"Unknown command: {command}")
            print("Available commands: create, validate, show")
    else:
        print("Usage:")
        print("  python config_loader.py create [output_path]")
        print("  python config_loader.py validate [config_path]")
        print("  python config_loader.py show [config_path]")
