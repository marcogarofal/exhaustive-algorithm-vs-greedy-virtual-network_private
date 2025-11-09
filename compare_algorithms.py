# compare_algorithms.py
"""
Script to compare three algorithms: Exhaustive, Greedy, and Simulated Annealing
All algorithms receive the same graph configuration for fair comparison
"""

from tree_optimizer import run_algorithm as run_exhaustive
from greedy_algorithm_wrapper import run_greedy_algorithm
from sa_algorithm_wrapper import run_sa_algorithm
from graph_generator import generate_complete_config
from config_loader import load_config, create_default_config
import json
import os


def compare_three_algorithms(result_exhaustive, result_greedy, result_sa, output_file='comparison_results.json'):
    """
    Compare results from all three algorithms

    Args:
        result_exhaustive: result dict from exhaustive algorithm
        result_greedy: result dict from greedy algorithm
        result_sa: result dict from simulated annealing algorithm
        output_file: file to save comparison
    """
    comparison = {
        'exhaustive_algorithm': {
            'execution_time': result_exhaustive['execution_time'],
            'num_nodes': result_exhaustive['num_nodes'],
            'num_edges': result_exhaustive['num_edges'],
            'best_tree_nodes': list(result_exhaustive['best_tree'].nodes()) if result_exhaustive['best_tree'] else [],
            'best_tree_edges': [list(edge) for edge in result_exhaustive['best_tree'].edges()] if result_exhaustive['best_tree'] else []
        },
        'greedy_algorithm': {
            'execution_time': result_greedy['execution_time'],
            'num_nodes': result_greedy['num_nodes'],
            'num_edges': result_greedy['num_edges'],
            'best_tree_nodes': list(result_greedy['best_tree'].nodes()) if result_greedy['best_tree'] else [],
            'best_tree_edges': [list(edge) for edge in result_greedy['best_tree'].edges()] if result_greedy['best_tree'] else [],
            'acc_cost': result_greedy.get('acc_cost', 0),
            'aoc_cost': result_greedy.get('aoc_cost', 0),
            'score': result_greedy.get('score', 0),
            'alpha': result_greedy.get('alpha', 0.5)
        },
        'simulated_annealing_algorithm': {
            'execution_time': result_sa['execution_time'],
            'num_nodes': result_sa['num_nodes'],
            'num_edges': result_sa['num_edges'],
            'best_tree_nodes': list(result_sa['best_tree'].nodes()) if result_sa['best_tree'] else [],
            'best_tree_edges': [list(edge) for edge in result_sa['best_tree'].edges()] if result_sa['best_tree'] else [],
            'initial_cost': result_sa.get('initial_cost', 0),
            'final_cost': result_sa.get('final_cost', 0),
            'cost_improvement': result_sa.get('cost_improvement', 0),
            'acc': result_sa.get('acc', 0),
            'aoc': result_sa.get('aoc', 0)
        },
        'comparison': {
            'fastest_algorithm': min(['exhaustive', 'greedy', 'simulated_annealing'],
                                   key=lambda x: result_exhaustive['execution_time'] if x == 'exhaustive'
                                   else result_greedy['execution_time'] if x == 'greedy'
                                   else result_sa['execution_time']),
            'speedup_greedy_vs_exhaustive': result_exhaustive['execution_time'] / result_greedy['execution_time'] if result_greedy['execution_time'] > 0 else float('inf'),
            'speedup_sa_vs_exhaustive': result_exhaustive['execution_time'] / result_sa['execution_time'] if result_sa['execution_time'] > 0 else float('inf'),
            'speedup_sa_vs_greedy': result_greedy['execution_time'] / result_sa['execution_time'] if result_sa['execution_time'] > 0 else float('inf'),
            'same_nodes_ex_gr': result_exhaustive['num_nodes'] == result_greedy['num_nodes'],
            'same_nodes_ex_sa': result_exhaustive['num_nodes'] == result_sa['num_nodes'],
            'same_nodes_gr_sa': result_greedy['num_nodes'] == result_sa['num_nodes']
        }
    }

    # Calculate identical solutions AFTER creating the dict above
    ex_nodes = set(comparison['exhaustive_algorithm']['best_tree_nodes'])
    ex_edges = set(tuple(sorted(edge)) for edge in comparison['exhaustive_algorithm']['best_tree_edges'])

    gr_nodes = set(comparison['greedy_algorithm']['best_tree_nodes'])
    gr_edges = set(tuple(sorted(edge)) for edge in comparison['greedy_algorithm']['best_tree_edges'])

    sa_nodes = set(comparison['simulated_annealing_algorithm']['best_tree_nodes'])
    sa_edges = set(tuple(sorted(edge)) for edge in comparison['simulated_annealing_algorithm']['best_tree_edges'])

    # Add identical solution checks
    comparison['comparison']['identical_solution_greedy_exhaustive'] = (ex_nodes == gr_nodes and ex_edges == gr_edges) if ex_nodes and gr_nodes else False
    comparison['comparison']['identical_solution_sa_exhaustive'] = (ex_nodes == sa_nodes and ex_edges == sa_edges) if ex_nodes and sa_nodes else False
    comparison['comparison']['identical_solution_greedy_sa'] = (gr_nodes == sa_nodes and gr_edges == sa_edges) if gr_nodes and sa_nodes else False

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print detailed comparison
    print(f"\n{'='*100}")
    print("THREE ALGORITHMS COMPARISON RESULTS")
    print(f"{'='*100}")

    print(f"\n{'Algorithm':<30} {'Time (s)':<15} {'Nodes':<10} {'Edges':<10} {'Special Metrics':<30}")
    print(f"{'-'*100}")

    # Exhaustive
    print(f"{'EXHAUSTIVE':<30} {result_exhaustive['execution_time']:<15.4f} {result_exhaustive['num_nodes']:<10} {result_exhaustive['num_edges']:<10} {'Guaranteed optimal':<30}")

    # Greedy
    greedy_metrics = f"ACC:{result_greedy.get('acc_cost', 0):.4f} AOC:{result_greedy.get('aoc_cost', 0):.4f}"
    greedy_alpha = result_greedy.get('alpha', 0.5)
    greedy_label = f"GREEDY (a={greedy_alpha})"
    print(f"{greedy_label:<30} {result_greedy['execution_time']:<15.4f} {result_greedy['num_nodes']:<10} {result_greedy['num_edges']:<10} {greedy_metrics:<30}")

    # Simulated Annealing
    sa_metrics = f"Cost improved:{result_sa.get('cost_improvement', 0):.4f}"
    print(f"{'SIMULATED ANNEALING':<30} {result_sa['execution_time']:<15.4f} {result_sa['num_nodes']:<10} {result_sa['num_edges']:<10} {sa_metrics:<30}")

    print(f"\n{'='*100}")
    print("SPEEDUP ANALYSIS")
    print(f"{'='*100}")

    speedup_greedy = comparison['comparison']['speedup_greedy_vs_exhaustive']
    speedup_sa = comparison['comparison']['speedup_sa_vs_exhaustive']
    speedup_sa_greedy = comparison['comparison']['speedup_sa_vs_greedy']

    print(f"  Greedy vs Exhaustive: {speedup_greedy:.2f}x faster")
    print(f"  SA vs Exhaustive: {speedup_sa:.2f}x faster")
    print(f"  SA vs Greedy: {speedup_sa_greedy:.2f}x {'faster' if speedup_sa_greedy > 1 else 'slower'}")

    print(f"\n{'='*100}")
    print("SOLUTION QUALITY")
    print(f"{'='*100}")

    # Node count comparison
    print(f"\n  Node Count Comparison:")
    print(f"    Exhaustive vs Greedy: {'✅ SAME' if comparison['comparison']['same_nodes_ex_gr'] else '❌ DIFFERENT'}")
    print(f"    Exhaustive vs SA: {'✅ SAME' if comparison['comparison']['same_nodes_ex_sa'] else '❌ DIFFERENT'}")
    print(f"    Greedy vs SA: {'✅ SAME' if comparison['comparison']['same_nodes_gr_sa'] else '❌ DIFFERENT'}")

    # Identical solution comparison (nodes AND edges)
    print(f"\n  🎯 IDENTICAL SOLUTIONS (same nodes AND same edges):")

    greedy_identical = comparison['comparison']['identical_solution_greedy_exhaustive']
    sa_identical = comparison['comparison']['identical_solution_sa_exhaustive']
    greedy_sa_identical = comparison['comparison']['identical_solution_greedy_sa']

    print(f"    🚀 Greedy = 📊 Exhaustive: {'✅ YES - IDENTICAL!' if greedy_identical else '❌ NO - DIFFERENT SOLUTIONS'}")
    if greedy_identical:
        print(f"       → Greedy found the OPTIMAL solution {speedup_greedy:.2f}x faster!")

    print(f"    🔥 SA = 📊 Exhaustive: {'✅ YES - IDENTICAL!' if sa_identical else '❌ NO - DIFFERENT SOLUTIONS'}")
    if sa_identical:
        print(f"       → SA found the OPTIMAL solution {speedup_sa:.2f}x faster!")

    print(f"    🚀 Greedy = 🔥 SA: {'✅ YES - IDENTICAL!' if greedy_sa_identical else '❌ NO - DIFFERENT SOLUTIONS'}")

    # Summary
    if greedy_identical and sa_identical:
        print(f"\n  🎉 ALL THREE ALGORITHMS FOUND THE SAME OPTIMAL SOLUTION!")
    elif greedy_identical:
        print(f"\n  ⭐ Greedy matched exhaustive optimum!")
    elif sa_identical:
        print(f"\n  ⭐ SA matched exhaustive optimum!")
    else:
        print(f"\n  ⚠️  Heuristics found different solutions than exhaustive")
        print(f"     (Exhaustive is guaranteed optimal)")

    print(f"\n💾 Detailed comparison saved to: {output_file}")
    print(f"{'='*100}\n")

    return comparison


if __name__ == "__main__":
    print("="*100)
    print("THREE ALGORITHMS COMPARISON: EXHAUSTIVE vs GREEDY vs SIMULATED ANNEALING")
    print("="*100)

    # 1. Load or create configuration
    if not os.path.exists('config.json'):
        print("\nCreating default config.json...")
        create_default_config('config.json')

    config = load_config('config.json')

    # 2. Generate graph configuration (same for all algorithms)
    print("\nGenerating graph configuration...")
    graph_config = generate_complete_config(config)

    seed = config.get('graph_parameters', {}).get('seed', 42)
    alpha = config.get('algorithm', {}).get('alpha', 0.5)

    print(f"\nGraph Configuration:")
    total_nodes = len(graph_config['weak_nodes']) + len(graph_config['power_nodes_mandatory']) + len(graph_config['power_nodes_discretionary'])
    print(f"  Total nodes: {total_nodes}")
    print(f"  Weak nodes: {len(graph_config['weak_nodes'])} -> {graph_config['weak_nodes']}")
    print(f"  Mandatory nodes: {len(graph_config['power_nodes_mandatory'])} -> {graph_config['power_nodes_mandatory']}")
    print(f"  Discretionary nodes: {len(graph_config['power_nodes_discretionary'])} -> {graph_config['power_nodes_discretionary']}")
    print(f"  Random seed: {seed} (ensures same graph structure for all)")

    # 3. Run EXHAUSTIVE algorithm
    print(f"\n{'='*100}")
    print("Running EXHAUSTIVE algorithm...")
    print(f"{'='*100}")

    algorithm_config_exhaustive = {'seed': seed}
    debug_config_exhaustive = {
        'plot_initial_graphs': False,
        'plot_intermediate': False,
        'plot_final': False,
        'save_plots': True,
        'verbose': False
    }

    result_exhaustive = run_exhaustive(
        graph_config=graph_config,
        algorithm_config=algorithm_config_exhaustive,
        debug_config=debug_config_exhaustive,
        output_dir='plots_exhaustive'
    )

    print(f"Exhaustive completed in {result_exhaustive['execution_time']:.4f}s")
    print(f"   Solution: {result_exhaustive['num_nodes']} nodes, {result_exhaustive['num_edges']} edges")

    # 4. Run GREEDY algorithm
    print(f"\n{'='*100}")
    print("Running GREEDY algorithm...")
    print(f"{'='*100}")

    algorithm_config_greedy = {
        'seed': seed,
        'alpha': alpha
    }

    result_greedy = run_greedy_algorithm(
        graph_config=graph_config,
        algorithm_config=algorithm_config_greedy,
        output_dir='plots_greedy'
    )

    print(f"Greedy completed in {result_greedy['execution_time']:.4f}s")
    print(f"   Solution: {result_greedy['num_nodes']} nodes, {result_greedy['num_edges']} edges")
    print(f"   Score: {result_greedy.get('score', 0):.2f}")

    # 5. Run SIMULATED ANNEALING algorithm
    print(f"\n{'='*100}")
    print("Running SIMULATED ANNEALING algorithm...")
    print(f"{'='*100}")

    algorithm_config_sa = {
        'seed': seed,
        'initial_temperature': 120,
        'k_factor': 12
    }

    result_sa = run_sa_algorithm(
        graph_config=graph_config,
        algorithm_config=algorithm_config_sa,
        output_dir='plots_sa'
    )

    print(f"Simulated Annealing completed in {result_sa['execution_time']:.4f}s")
    print(f"   Solution: {result_sa['num_nodes']} nodes, {result_sa['num_edges']} edges")
    print(f"   Cost: {result_sa['initial_cost']:.4f} -> {result_sa['final_cost']:.4f} (improved by {result_sa['cost_improvement']:.4f})")

    # 6. Compare all three results
    comparison = compare_three_algorithms(result_exhaustive, result_greedy, result_sa)

    # 7. Additional insights
    print("\nINSIGHTS:")

    # Check exhaustive scores.json
    scores_path = 'plots_exhaustive/scores.json'
    if os.path.exists(scores_path):
        with open(scores_path, 'r') as f:
            exhaustive_scores = json.load(f)

        print(f"\n  Exhaustive algorithm:")
        print(f"     - Explored {len(exhaustive_scores['trees'])} tree configurations")
        print(f"     - Guaranteed to find optimal solution")

    print(f"\n  Greedy algorithm:")
    print(f"     - Very fast heuristic approach")
    print(f"     - Uses custom cost function (a={alpha})")
    print(f"     - Tested only 2 configurations")

    print(f"\n  Simulated Annealing:")
    print(f"     - Metaheuristic optimization")
    improvement_pct = (result_sa['cost_improvement']/result_sa['initial_cost']*100 if result_sa['initial_cost'] > 0 else 0)
    print(f"     - Improved solution quality by {improvement_pct:.1f}%")
    print(f"     - Balances exploration vs exploitation")

    # Recommendation
    print(f"\nRECOMMENDATION:")
    fastest = comparison['comparison']['fastest_algorithm']
    print(f"   - Fastest: {fastest.upper()}")
    print(f"   - For small graphs (<15 nodes): Use EXHAUSTIVE for guaranteed optimality")
    print(f"   - For medium graphs (15-30 nodes): Use SIMULATED ANNEALING for good trade-off")
    print(f"   - For large graphs (>30 nodes): Use GREEDY for speed")

    print(f"\n{'='*100}")
    print("COMPARISON COMPLETE")
    print(f"{'='*100}\n")
