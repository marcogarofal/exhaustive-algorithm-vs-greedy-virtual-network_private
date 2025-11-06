# main.py
"""
Example usage of the tree optimization algorithm
Shows both standalone and library usage modes
"""

from tree_optimizer import run_algorithm
from graph_generator import generate_complete_config
from config_loader import load_config, create_default_config
import json
import os


def example_with_config_file():
    """Example: Run algorithm with configuration from JSON file"""
    print("=" * 60)
    print("EXAMPLE 1: Running with config.json")
    print("=" * 60)

    # Create default config if it doesn't exist
    if not os.path.exists('config.json'):
        print("Creating default config.json...")
        create_default_config('config.json')

    # Load configuration
    config = load_config('config.json')

    # Generate graph configuration
    graph_config = generate_complete_config(config)

    # Run algorithm
    result = run_algorithm(
        graph_config=graph_config,
        algorithm_config=config.get('algorithm', {}),
        debug_config=config.get('debug', {}),
        output_dir=config.get('output', {}).get('plots_dir', 'plots')
    )

    print("\nResults:")
    print(f"  Execution time: {result['execution_time']:.2f} seconds")
    print(f"  Number of nodes in best tree: {result['num_nodes']}")
    print(f"  Number of edges in best tree: {result['num_edges']}")

    return result


def example_programmatic():
    """Example: Run algorithm with programmatic configuration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Running with programmatic configuration")
    print("=" * 60)

    # Define configuration programmatically
    graph_config = {
        'weak_nodes': [1, 2, 3],
        'power_nodes_mandatory': [4, 5],
        'power_nodes_discretionary': [6, 7, 8],
        'capacities': {
            1: 10, 2: 30, 3: 2, 4: 1, 5: 10, 6: 4, 7: 5, 8: 5
        }
    }

    algorithm_config = {
        'seed': 42,
        'weight_range': [1, 10]
    }

    debug_config = {
        'plot_initial_graphs': False,
        'plot_intermediate': False,
        'plot_final': False,
        'save_plots': True,
        'verbose': False
    }

    # Run algorithm
    result = run_algorithm(
        graph_config=graph_config,
        algorithm_config=algorithm_config,
        debug_config=debug_config,
        output_dir='plots_example2'
    )

    print("\nResults:")
    print(f"  Execution time: {result['execution_time']:.2f} seconds")
    print(f"  Number of nodes in best tree: {result['num_nodes']}")
    print(f"  Number of edges in best tree: {result['num_edges']}")

    return result


def example_comparison_setup():
    """Example: Setup for algorithm comparison"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Setup for comparing different algorithms")
    print("=" * 60)

    # Load base configuration
    if not os.path.exists('config.json'):
        create_default_config('config.json')

    config = load_config('config.json')
    graph_config = generate_complete_config(config)

    # Save graph configuration for use with other algorithms
    comparison_config = {
        'graph_config': graph_config,
        'algorithm_config': config.get('algorithm', {}),
        'test_id': 'comparison_test_001',
        'seed': 42
    }

    output_path = 'comparison_config.json'
    with open(output_path, 'w') as f:
        json.dump(comparison_config, f, indent=2)

    print(f"\nComparison configuration saved to {output_path}")
    print("You can now use this configuration with different algorithms")
    print("\nGraph configuration:")
    print(f"  Total nodes: {len(graph_config['weak_nodes']) + len(graph_config['power_nodes_mandatory']) + len(graph_config['power_nodes_discretionary'])}")
    print(f"  Weak: {graph_config['weak_nodes']}")
    print(f"  Mandatory: {graph_config['power_nodes_mandatory']}")
    print(f"  Discretionary: {graph_config['power_nodes_discretionary']}")

    return comparison_config


def example_load_and_compare():
    """Example: Load saved configuration and run algorithm"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Load comparison config and run algorithm")
    print("=" * 60)

    # Check if comparison config exists
    if not os.path.exists('comparison_config.json'):
        print("Creating comparison config first...")
        example_comparison_setup()

    # Load comparison configuration
    with open('comparison_config.json', 'r') as f:
        comparison_config = json.load(f)

    print(f"Running test: {comparison_config['test_id']}")

    # Run algorithm with loaded configuration
    result = run_algorithm(
        graph_config=comparison_config['graph_config'],
        algorithm_config=comparison_config['algorithm_config'],
        debug_config={'plot_final': False, 'save_plots': True},
        output_dir='plots_comparison'
    )

    # Save results
    result_output = {
        'test_id': comparison_config['test_id'],
        'algorithm': 'tree_optimizer',
        'execution_time': result['execution_time'],
        'num_nodes': result['num_nodes'],
        'num_edges': result['num_edges']
    }

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    result_path = os.path.join(results_dir, f"{comparison_config['test_id']}_tree_optimizer.json")
    with open(result_path, 'w') as f:
        json.dump(result_output, f, indent=2)

    print(f"\nResults saved to {result_path}")
    print(f"Execution time: {result['execution_time']:.2f} seconds")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]

        if example == '1':
            example_with_config_file()
        elif example == '2':
            example_programmatic()
        elif example == '3':
            example_comparison_setup()
        elif example == '4':
            example_load_and_compare()
        elif example == 'all':
            example_with_config_file()
            example_programmatic()
            example_comparison_setup()
            example_load_and_compare()
        else:
            print(f"Unknown example: {example}")
            print("Available examples: 1, 2, 3, 4, all")
    else:
        print("Running all examples...\n")
        example_with_config_file()
        example_programmatic()
        example_comparison_setup()
        example_load_and_compare()
