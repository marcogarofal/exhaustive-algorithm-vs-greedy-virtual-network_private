# plot_benchmark_results.py
"""
Generate plots grouped by (ratio, capacity, strategy) combinations
Each plot shows all num_nodes values on X-axis
Includes cost difference analysis with graph regeneration and validation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import networkx as nx
import random


# ======================== CONFIGURABLE SETTINGS ========================

# Which raw results file to load (batch or merged)
RAW_RESULTS_FILE = 'benchmark_results/merged_results.json'

# Output settings
OUTPUT_DIR = 'benchmark_plots'
FIGURE_SIZE = (12, 10)  # Taller for 3 subplots
DPI = 300

# =============================================================================


def load_raw_results(filepath):
    """Load raw results from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'results' in data:
        results = data['results']
        print(f"✓ Loaded {len(results)} test results from batch")
    elif isinstance(data, list):
        results = data
        print(f"✓ Loaded {len(results)} test results")
    else:
        raise ValueError("Unknown file format")

    return results


def create_graph_with_strategy_from_test(test):
    """
    Regenerate the exact graph used in a test
    Uses same seed and configuration to guarantee identical graph
    """
    from graph_generator import generate_graph_config, generate_capacities

    config = test['config']
    seed = test['seed']

    # Generate graph structure
    graph_config_dict = generate_graph_config(
        num_nodes=config['num_nodes'],
        weak_ratio=config['weak_ratio'],
        mandatory_ratio=config['mandatory_ratio'],
        seed=seed
    )

    all_nodes = (graph_config_dict['weak_nodes'] +
                graph_config_dict['power_nodes_mandatory'] +
                graph_config_dict['power_nodes_discretionary'])

    # Parse capacity config from string
    cap_str = config['capacity_config']
    if 'default_' in cap_str:
        cap_value = int(cap_str.replace('default_', ''))
        capacity_config = {'default': cap_value}
    elif 'rand_' in cap_str or 'c' in cap_str:
        # Parse "rand_1-5" or "c1_5"
        parts = cap_str.replace('rand_', '').replace('c', '').replace('_', '-').split('-')
        if len(parts) == 2:
            capacity_config = {'random': {'min': int(parts[0]), 'max': int(parts[1]), 'seed': seed}}
        else:
            capacity_config = {'default': 10}  # Fallback
    else:
        capacity_config = {'default': 10}

    capacities = generate_capacities(all_nodes, capacity_config)

    # Determine weight strategy
    strategy_name = config['weight_strategy']
    if strategy_name == 'uniform':
        weight_strategy = {
            'default_range': (1, 10),
            'discretionary_range': (1, 10)
        }
    elif strategy_name == 'favor_discretionary':
        weight_strategy = {
            'default_range': (1, 10),
            'discretionary_range': (1, 5)
        }
    elif strategy_name == 'strong_favor':
        weight_strategy = {
            'default_range': (5, 10),
            'discretionary_range': (1, 3)
        }
    else:
        weight_strategy = {
            'default_range': (1, 10),
            'discretionary_range': (1, 10)
        }

    # Recreate graph
    random.seed(seed)

    G = nx.Graph()
    discretionary_set = set(graph_config_dict['power_nodes_discretionary'])

    # Add nodes
    for node in graph_config_dict['weak_nodes']:
        G.add_node(node, node_type='weak', capacity=capacities.get(node, 1))
    for node in graph_config_dict['power_nodes_mandatory']:
        G.add_node(node, node_type='power_mandatory', capacity=capacities.get(node, 10))
    for node in graph_config_dict['power_nodes_discretionary']:
        G.add_node(node, node_type='power_discretionary', capacity=capacities.get(node, 10))

    # Add edges
    for i in all_nodes:
        for j in all_nodes:
            if i < j:
                if i in discretionary_set or j in discretionary_set:
                    min_w, max_w = weight_strategy['discretionary_range']
                else:
                    min_w, max_w = weight_strategy['default_range']

                weight = random.randint(min_w, max_w)
                G.add_edge(i, j, weight=weight)

    return G, capacities, graph_config_dict


def calculate_solution_cost_normalized(graph, solution_edges, solution_nodes, capacities,
                                      power_nodes_mandatory, power_nodes_discretionary,
                                      ref_max_weight, ref_num_edges):
    """
    Calculate normalized cost using reference values (from exhaustive solution)

    Args:
        graph: NetworkX graph with weights
        solution_edges: list of edges in solution
        solution_nodes: list of nodes in solution
        capacities: dict of node capacities
        power_nodes_mandatory: list of mandatory nodes
        power_nodes_discretionary: list of discretionary nodes
        ref_max_weight: max weight from reference solution (exhaustive)
        ref_num_edges: number of edges from reference solution (exhaustive)

    Returns:
        total_cost, edge_cost, degree_cost
    """
    if not solution_edges:
        return 0, 0, 0

    # Edge cost - normalized using REFERENCE values
    total_weight = sum(graph[u][v]['weight'] for u, v in solution_edges)
    edge_cost = total_weight / (ref_max_weight * ref_num_edges)

    # Degree cost
    set_power_nodes = set(list(power_nodes_mandatory) + list(power_nodes_discretionary))

    # Build degree dict from edges
    degree_dict = defaultdict(int)
    for u, v in solution_edges:
        degree_dict[u] += 1
        degree_dict[v] += 1

    degree_cost = 0
    for node in solution_nodes:
        if node in set_power_nodes and node in capacities:
            degree_cost += degree_dict[node] / capacities[node]

    degree_cost = degree_cost / len(solution_nodes) if solution_nodes else 0

    total_cost = edge_cost + degree_cost

    return total_cost, edge_cost, degree_cost


def calculate_stats_with_cost_diff(test_list):
    """Calculate statistics including cost differences (absolute and percentage)"""
    stats = {
        'n_tests': len(test_list),
        'exhaustive_times': [],
        'greedy_times': [],
        'sa_times': [],
        'greedy_matches': 0,
        'sa_matches': 0,
        'greedy_cost_diffs_abs': [],
        'greedy_cost_diffs_pct': [],
        'sa_cost_diffs_abs': [],
        'sa_cost_diffs_pct': [],
        'validation_errors': 0
    }

    for test in test_list:
        # Skip tests with errors
        if 'error' in test.get('exhaustive', {}):
            continue

        # Collect times
        if 'error' not in test.get('exhaustive', {}):
            stats['exhaustive_times'].append(test['exhaustive']['time'])
        if 'error' not in test.get('greedy', {}):
            stats['greedy_times'].append(test['greedy']['time'])
            if test['greedy'].get('matches_exhaustive', False):
                stats['greedy_matches'] += 1
        if 'error' not in test.get('sa', {}):
            stats['sa_times'].append(test['sa']['time'])
            if test['sa'].get('matches_exhaustive', False):
                stats['sa_matches'] += 1

        # Calculate cost differences
        try:
            # Regenerate graph
            graph, capacities, graph_config_dict = create_graph_with_strategy_from_test(test)

            # Validate: check exhaustive edges exist in graph
            ex_edges = [tuple(e) for e in test['exhaustive']['solution_edges']]
            valid = all(graph.has_edge(u, v) or graph.has_edge(v, u) for u, v in ex_edges)

            if not valid:
                stats['validation_errors'] += 1
                continue  # Skip this test

            # Calculate exhaustive cost (establish reference normalization)
            ex_nodes = test['exhaustive']['solution_nodes']
            ex_edges = [tuple(e) for e in test['exhaustive']['solution_edges']]

            # Get exhaustive normalization parameters
            ex_max_weight = max(graph[u][v]['weight'] for u, v in ex_edges)
            ex_num_edges = len(ex_edges)

            # Calculate exhaustive cost with its own normalization
            ex_cost, ex_edge_cost, ex_degree_cost = calculate_solution_cost_normalized(
                graph, ex_edges, ex_nodes, capacities,
                graph_config_dict['power_nodes_mandatory'],
                graph_config_dict['power_nodes_discretionary'],
                ex_max_weight, ex_num_edges  # Uses its own values as reference
            )

            # Calculate greedy cost ALWAYS (even when matching)
            if 'error' not in test.get('greedy', {}):
                if test['greedy'].get('matches_exhaustive', False):
                    # Matches → cost difference is 0
                    cost_diff_abs = 0.0
                    cost_diff_pct = 0.0
                else:
                    # Doesn't match → calculate actual cost difference
                    gr_edges = [tuple(e) for e in test['greedy']['solution_edges']]
                    gr_nodes = test['greedy']['solution_nodes']

                    if all(graph.has_edge(u, v) or graph.has_edge(v, u) for u, v in gr_edges):
                        gr_cost, _, _ = calculate_solution_cost_normalized(
                            graph, gr_edges, gr_nodes, capacities,
                            graph_config_dict['power_nodes_mandatory'],
                            graph_config_dict['power_nodes_discretionary'],
                            ex_max_weight, ex_num_edges
                        )

                        cost_diff_abs = gr_cost - ex_cost
                        cost_diff_pct = (gr_cost - ex_cost) / ex_cost * 100 if ex_cost > 0 else 0
                    else:
                        stats['validation_errors'] += 1
                        continue

                stats['greedy_cost_diffs_abs'].append(cost_diff_abs)
                stats['greedy_cost_diffs_pct'].append(cost_diff_pct)

            # Calculate SA cost ALWAYS (even when matching)
            if 'error' not in test.get('sa', {}):
                if test['sa'].get('matches_exhaustive', False):
                    # Matches → cost difference is 0
                    cost_diff_abs = 0.0
                    cost_diff_pct = 0.0
                else:
                    # Doesn't match → calculate actual cost difference
                    sa_edges = [tuple(e) for e in test['sa']['solution_edges']]
                    sa_nodes = test['sa']['solution_nodes']

                    if all(graph.has_edge(u, v) or graph.has_edge(v, u) for u, v in sa_edges):
                        sa_cost, _, _ = calculate_solution_cost_normalized(
                            graph, sa_edges, sa_nodes, capacities,
                            graph_config_dict['power_nodes_mandatory'],
                            graph_config_dict['power_nodes_discretionary'],
                            ex_max_weight, ex_num_edges
                        )

                        cost_diff_abs = sa_cost - ex_cost
                        cost_diff_pct = (sa_cost - ex_cost) / ex_cost * 100 if ex_cost > 0 else 0
                    else:
                        stats['validation_errors'] += 1
                        continue

                stats['sa_cost_diffs_abs'].append(cost_diff_abs)
                stats['sa_cost_diffs_pct'].append(cost_diff_pct)

        except Exception as e:
            stats['validation_errors'] += 1
            continue

    # Calculate statistics
    result = {'n_tests': stats['n_tests']}

    # Time stats
    for algo in ['exhaustive', 'greedy', 'sa']:
        times = stats[f'{algo}_times']
        if times:
            result[f'{algo}_mean'] = np.mean(times)
            result[f'{algo}_std'] = np.std(times)
        else:
            result[f'{algo}_mean'] = 0
            result[f'{algo}_std'] = 0

    # Match rates
    result['greedy_match_rate'] = stats['greedy_matches'] / stats['n_tests'] if stats['n_tests'] > 0 else 0
    result['sa_match_rate'] = stats['sa_matches'] / stats['n_tests'] if stats['n_tests'] > 0 else 0
    result['greedy_match_count'] = stats['greedy_matches']
    result['sa_match_count'] = stats['sa_matches']

    # Cost difference stats (includes ALL cases, matching = 0)
    # Absolute differences
    if stats['greedy_cost_diffs_abs']:
        result['greedy_cost_diff_abs_mean'] = np.mean(stats['greedy_cost_diffs_abs'])
        result['greedy_cost_diff_abs_std'] = np.std(stats['greedy_cost_diffs_abs'])
        result['greedy_cost_diff_count'] = len(stats['greedy_cost_diffs_abs'])
    else:
        result['greedy_cost_diff_abs_mean'] = 0
        result['greedy_cost_diff_abs_std'] = 0
        result['greedy_cost_diff_count'] = 0

    if stats['sa_cost_diffs_abs']:
        result['sa_cost_diff_abs_mean'] = np.mean(stats['sa_cost_diffs_abs'])
        result['sa_cost_diff_abs_std'] = np.std(stats['sa_cost_diffs_abs'])
        result['sa_cost_diff_count'] = len(stats['sa_cost_diffs_abs'])
    else:
        result['sa_cost_diff_abs_mean'] = 0
        result['sa_cost_diff_abs_std'] = 0
        result['sa_cost_diff_count'] = 0

    # Percentage differences
    if stats['greedy_cost_diffs_pct']:
        result['greedy_cost_diff_pct_mean'] = np.mean(stats['greedy_cost_diffs_pct'])
        result['greedy_cost_diff_pct_std'] = np.std(stats['greedy_cost_diffs_pct'])
    else:
        result['greedy_cost_diff_pct_mean'] = 0
        result['greedy_cost_diff_pct_std'] = 0

    if stats['sa_cost_diffs_pct']:
        result['sa_cost_diff_pct_mean'] = np.mean(stats['sa_cost_diffs_pct'])
        result['sa_cost_diff_pct_std'] = np.std(stats['sa_cost_diffs_pct'])
    else:
        result['sa_cost_diff_pct_mean'] = 0
        result['sa_cost_diff_pct_std'] = 0

    result['validation_errors'] = stats['validation_errors']

    return result


def group_by_fixed_params(all_results):
    """Group results by (ratio, capacity, strategy) - varying only num_nodes"""
    grouped = defaultdict(lambda: defaultdict(list))

    for test in all_results:
        config = test['config']

        fixed_key = (
            config['weak_ratio'],
            config['mandatory_ratio'],
            config['capacity_config'],
            config['weight_strategy']
        )

        num_nodes = config['num_nodes']
        grouped[fixed_key][num_nodes].append(test)

    return grouped


def create_filename_from_fixed_params(fixed_key):
    """Create filename from fixed parameters"""
    weak_r, mand_r, cap, strategy = fixed_key
    discr = int((1 - weak_r - mand_r) * 100)

    filename = (f"w{int(weak_r*100)}_m{int(mand_r*100)}_d{discr}_"
               f"{cap.replace('default_', 'c').replace('rand_', 'c').replace('-', '_')}_"
               f"{strategy}")

    return filename


def create_title_from_fixed_params(fixed_key):
    """Create readable title from fixed parameters"""
    weak_r, mand_r, cap, strategy = fixed_key
    discr = int((1 - weak_r - mand_r) * 100)

    title = (f"{int(weak_r*100)}% Weak, {int(mand_r*100)}% Mandatory, {discr}% Discretionary | "
            f"Capacity: {cap} | Strategy: {strategy}")

    return title


def plot_configuration_group(fixed_key, nodes_data, output_dir):
    """
    Create plot for one (ratio, capacity, strategy) combination
    Shows all num_nodes values on X-axis with 3 subplots
    """

    sorted_nodes = sorted(nodes_data.keys())

    print(f"  Calculating statistics (including cost differences)...")
    all_stats = {}
    for num_nodes in sorted_nodes:
        all_stats[num_nodes] = calculate_stats_with_cost_diff(nodes_data[num_nodes])

        # Report validation errors
        if all_stats[num_nodes]['validation_errors'] > 0:
            print(f"    ⚠️ {num_nodes} nodes: {all_stats[num_nodes]['validation_errors']} validation errors")

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=FIGURE_SIZE,
                                        gridspec_kw={'height_ratios': [1.2, 1, 1]})

    x = np.arange(len(sorted_nodes))
    width = 0.25

    # ==================== SUBPLOT 1: EXECUTION TIMES ====================

    ex_means = [all_stats[n]['exhaustive_mean'] for n in sorted_nodes]
    ex_stds = [all_stats[n]['exhaustive_std'] for n in sorted_nodes]
    gr_means = [all_stats[n]['greedy_mean'] for n in sorted_nodes]
    gr_stds = [all_stats[n]['greedy_std'] for n in sorted_nodes]
    sa_means = [all_stats[n]['sa_mean'] for n in sorted_nodes]
    sa_stds = [all_stats[n]['sa_std'] for n in sorted_nodes]

    bars1 = ax1.bar(x - width, ex_means, width, yerr=ex_stds,
                    label='Exhaustive', capsize=5, color='#2E86AB', alpha=0.85,
                    edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x, gr_means, width, yerr=gr_stds,
                    label='Greedy', capsize=5, color='#A23B72', alpha=0.85,
                    edgecolor='black', linewidth=1)
    bars3 = ax1.bar(x + width, sa_means, width, yerr=sa_stds,
                    label='SA', capsize=5, color='#F18F01', alpha=0.85,
                    edgecolor='black', linewidth=1)

    for i in range(len(sorted_nodes)):
        y_offset = max(ex_means[i], gr_means[i], sa_means[i]) * 0.05
        ax1.text(i - width, ex_means[i] + ex_stds[i] + y_offset,
                f'{ex_means[i]:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i, gr_means[i] + gr_stds[i] + y_offset,
                f'{gr_means[i]:.4f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width, sa_means[i] + sa_stds[i] + y_offset,
                f'{sa_means[i]:.3f}', ha='center', va='bottom', fontsize=8)

    ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Execution Times (mean ± std over 100 seeds)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{n}' for n in sorted_nodes], fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # ==================== SUBPLOT 2: MATCH RATES ====================

    width2 = 0.35

    gr_rates = [all_stats[n]['greedy_match_rate'] * 100 for n in sorted_nodes]
    sa_rates = [all_stats[n]['sa_match_rate'] * 100 for n in sorted_nodes]

    bars_gr = ax2.bar(x - width2/2, gr_rates, width2, label='Greedy',
                     color='#A23B72', alpha=0.85, edgecolor='black', linewidth=1)
    bars_sa = ax2.bar(x + width2/2, sa_rates, width2, label='SA',
                     color='#F18F01', alpha=0.85, edgecolor='black', linewidth=1)

    for i, node in enumerate(sorted_nodes):
        gr_count = all_stats[node]['greedy_match_count']
        sa_count = all_stats[node]['sa_match_count']
        n_tests = all_stats[node]['n_tests']

        ax2.text(i - width2/2, gr_rates[i] + 2,
                f"{gr_rates[i]:.0f}%\n{gr_count}/{n_tests}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(i + width2/2, sa_rates[i] + 2,
                f"{sa_rates[i]:.0f}%\n{sa_count}/{n_tests}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel('Match Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Solution Quality (% matching Exhaustive)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{n}' for n in sorted_nodes], fontsize=10)
    ax2.set_ylim([0, 115])
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.6, linewidth=2)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # ==================== SUBPLOT 3: COST DIFFERENCE ====================

    # Use percentage for better interpretability
    gr_diff_means = [all_stats[n]['greedy_cost_diff_pct_mean'] for n in sorted_nodes]
    gr_diff_stds = [all_stats[n]['greedy_cost_diff_pct_std'] for n in sorted_nodes]
    sa_diff_means = [all_stats[n]['sa_cost_diff_pct_mean'] for n in sorted_nodes]
    sa_diff_stds = [all_stats[n]['sa_cost_diff_pct_std'] for n in sorted_nodes]

    bars_gr_diff = ax3.bar(x - width2/2, gr_diff_means, width2, yerr=gr_diff_stds,
                          label='Greedy', capsize=5, color='#A23B72', alpha=0.85,
                          edgecolor='black', linewidth=1)
    bars_sa_diff = ax3.bar(x + width2/2, sa_diff_means, width2, yerr=sa_diff_stds,
                          label='SA', capsize=5, color='#F18F01', alpha=0.85,
                          edgecolor='black', linewidth=1)

    # Add value labels (show both percentage and absolute)
    for i, node in enumerate(sorted_nodes):
        gr_count = all_stats[node]['greedy_cost_diff_count']
        sa_count = all_stats[node]['sa_cost_diff_count']

        gr_abs = all_stats[node]['greedy_cost_diff_abs_mean']
        sa_abs = all_stats[node]['sa_cost_diff_abs_mean']

        if gr_diff_means[i] > 0 or gr_abs > 0:
            ax3.text(i - width2/2, gr_diff_means[i] + gr_diff_stds[i],
                    f'{gr_diff_means[i]:.1f}%\n({gr_abs:.4f})',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        if sa_diff_means[i] > 0 or sa_abs > 0:
            ax3.text(i + width2/2, sa_diff_means[i] + sa_diff_stds[i],
                    f'{sa_diff_means[i]:.1f}%\n({sa_abs:.4f})',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax3.set_ylabel('Cost Difference from Optimal (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Average Cost Difference (% worse than exhaustive, over all 100 seeds)',
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('Number of Nodes', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{n} nodes' for n in sorted_nodes], fontsize=10)
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Optimal (0%)')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Main title
    fig.suptitle(create_title_from_fixed_params(fixed_key),
                fontsize=12, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    filename = create_filename_from_fixed_params(fixed_key) + '.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    """Main function"""

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("="*100)
    print("BENCHMARK PLOTTER - WITH COST DIFFERENCE ANALYSIS")
    print("="*100)

    print(f"\nLoading: {RAW_RESULTS_FILE}")
    all_results = load_raw_results(RAW_RESULTS_FILE)

    print("\nGrouping results...")
    grouped = group_by_fixed_params(all_results)

    print(f"\nFound {len(grouped)} unique (ratio, capacity, strategy) combinations")
    print("Each will generate 1 plot with:")
    print("  - Execution times")
    print("  - Match rates")
    print("  - Cost differences (for non-matching solutions)")
    print("\nNote: Graph regeneration for cost calculation may take a few seconds...\n")

    print("="*100)
    print("GENERATING PLOTS")
    print("="*100 + "\n")

    plot_count = 0
    for fixed_key, nodes_data in sorted(grouped.items()):
        plot_count += 1

        weak_r, mand_r, cap, strategy = fixed_key
        discr = int((1 - weak_r - mand_r) * 100)

        print(f"{plot_count}. w{int(weak_r*100)}%_m{int(mand_r*100)}%_d{discr}%, {cap}, {strategy}")
        print(f"   Nodes: {sorted(nodes_data.keys())}")

        output_path = plot_configuration_group(fixed_key, nodes_data, OUTPUT_DIR)
        print(f"   ✓ Saved: {os.path.basename(output_path)}\n")

    print("="*100)
    print(f"✅ Generated {plot_count} plots")
    print(f"   Location: {OUTPUT_DIR}/")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
