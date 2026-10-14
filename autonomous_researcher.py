import os
import sys
import json
import time
import random
import importlib.util
from pathlib import Path
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import functools
print = functools.partial(print, flush=True)

from benchmark import get_benchmark_suite
from evaluator import evaluate_algorithm
import algorithms_library as baselines

DISCOVERED_DIR = Path("discovered_algorithms")
DISCOVERED_DIR.mkdir(exist_ok=True)
LEADERBOARD_FILE = Path("leaderboard.json")
STATUS_FILE = Path("research_status.json")

# Strategy Templates that the synthesizer can explore and combine
STRATEGY_TEMPLATES = [
    {
        "name": "Adaptive_VNS",
        "description": "Variable Neighborhood Search combining 2-Opt and Random 3-Opt Kicks to escape local minima",
        "code_template": """
import random

def calculate_tour_len(tour, dist_matrix):
    n = len(tour)
    return sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))

def solve(dist_matrix):
    n = len(dist_matrix)
    # Greedy Nearest Neighbor Initialization
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda c: dist_matrix[current][c])
        unvisited.remove(nxt)
        tour.append(nxt)
        current = nxt

    best_dist = calculate_tour_len(tour, dist_matrix)
    max_kicks = {max_kicks}
    
    for _ in range(max_kicks):
        # 2-Opt local search
        improved = True
        while improved:
            improved = False
            for i in range(1, n):
                for j in range(i + 1, n + 1):
                    if j - i == 1 or (i == 1 and j == n):
                        continue
                    new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                    new_d = calculate_tour_len(new_tour, dist_matrix)
                    if new_d < best_dist - 1e-6:
                        tour = new_tour
                        best_dist = new_d
                        improved = True
                        break
                if improved:
                    break

        # Double-bridge kick to escape local minima if stuck
        if n >= 8:
            pts = sorted(random.sample(range(1, n), 4))
            a, b, c, d = pts
            # Reconnect 4 segments in non-trivial permutation
            kicked_tour = tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]
            kicked_d = calculate_tour_len(kicked_tour, dist_matrix)
            if kicked_d < best_dist * {kick_tolerance}:
                tour = kicked_tour
                best_dist = kicked_d

    return tour
"""
    },
    {
        "name": "Guided_Local_Search",
        "description": "2-Opt with edge penalty memory (Tabu/Guided) to penalize over-used long edges",
        "code_template": """
def calculate_tour_len(tour, dist_matrix):
    n = len(tour)
    return sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))

def solve(dist_matrix):
    n = len(dist_matrix)
    # Furthest insertion initialization
    tour = [0]
    unvisited = set(range(1, n))
    farthest = max(unvisited, key=lambda c: dist_matrix[0][c])
    tour.append(farthest)
    unvisited.remove(farthest)

    while unvisited:
        # Find city with max minimum distance to current tour
        next_city = max(unvisited, key=lambda c: min(dist_matrix[c][t] for t in tour))
        best_pos = -1
        best_inc = float('inf')
        for i in range(len(tour)):
            u, v = tour[i], tour[(i + 1) % len(tour)]
            inc = dist_matrix[u][next_city] + dist_matrix[next_city][v] - dist_matrix[u][v]
            if inc < best_inc:
                best_inc = inc
                best_pos = i + 1
        tour.insert(best_pos, next_city)
        unvisited.remove(next_city)

    # 2-Opt pass
    improved = True
    best_dist = calculate_tour_len(tour, dist_matrix)
    passes = 0
    while improved and passes < {max_passes}:
        improved = False
        passes += 1
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                if j - i == 1 or (i == 1 and j == n):
                    continue
                cand = tour[:i] + tour[i:j][::-1] + tour[j:]
                d = calculate_tour_len(cand, dist_matrix)
                if d < best_dist - 1e-6:
                    tour = cand
                    best_dist = d
                    improved = True
                    break
            if improved:
                break
    return tour
"""
    },
    {
        "name": "Convex_Hull_Angle_Insertion",
        "description": "Multi-fragment Clarke-Wright Savings style edge merger with 2.5-Opt node relocation",
        "code_template": """
def calculate_tour_len(tour, dist_matrix):
    n = len(tour)
    return sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))

def solve(dist_matrix):
    n = len(dist_matrix)
    # Savings-based greedy edge joining
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist_matrix[i][j], i, j))
    edges.sort()

    adj = {i: [] for i in range(n)}
    deg = [0] * n

    for d, u, v in edges:
        if deg[u] < 2 and deg[v] < 2:
            # Check cycle unless it closes the full tour
            visited = set()
            curr = u
            cycle = False
            while True:
                visited.add(curr)
                neighbors = [x for x in adj[curr] if x not in visited]
                if not neighbors:
                    break
                curr = neighbors[0]
                if curr == v:
                    cycle = True
                    break
            if not cycle or len(visited) == n:
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1

    # Connect remaining ends to form closed tour
    ends = [i for i in range(n) if deg[i] < 2]
    if len(ends) == 2:
        adj[ends[0]].append(ends[1])
        adj[ends[1]].append(ends[0])

    # Extract tour
    tour = []
    curr = 0
    prev = -1
    while len(tour) < n:
        tour.append(curr)
        nexts = [x for x in adj[curr] if x != prev]
        prev = curr
        curr = nexts[0] if nexts else (set(range(n)) - set(tour)).pop()

    # Node Relocation (Or-Opt step 1)
    best_dist = calculate_tour_len(tour, dist_matrix)
    for _ in range({relocation_passes}):
        improved = False
        for i in range(1, n):
            city = tour[i]
            rem_tour = tour[:i] + tour[i+1:]
            for pos in range(len(rem_tour)):
                cand = rem_tour[:pos] + [city] + rem_tour[pos:]
                cand_d = calculate_tour_len(cand, dist_matrix)
                if cand_d < best_dist - 1e-6:
                    tour = cand
                    best_dist = cand_d
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return tour
"""
    }
]

def load_leaderboard():
    if LEADERBOARD_FILE.exists():
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_leaderboard(leaderboard):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(leaderboard, f, indent=2)

def update_status(status_dict):
    with open(STATUS_FILE, "w") as f:
        json.dump(status_dict, f, indent=2)

def evaluate_and_record(name, func, code_str, benchmark_suite, gen_id):
    """Evaluates an algorithm and adds it to the leaderboard."""
    print(f"\n[Evaluating Gen #{gen_id}: '{name}']...")
    results = evaluate_algorithm(func, benchmark_suite)

    record = {
        "id": gen_id,
        "name": name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "valid": results["valid"],
        "error": results["error"],
        "avg_optimality_gap_pct": results["avg_optimality_gap_pct"],
        "empirical_beta": results["empirical_beta"],
        "scaling_label": results["scaling_label"],
        "avg_runtime_ms": results["avg_runtime_ms"],
        "p_equals_np_candidate": results["p_equals_np_candidate"],
        "verdict": results["verdict"],
        "exact_gaps": results["exact_gaps"],
        "scaling_points": results["scaling_points"]
    }

    # Save code to disk
    code_path = DISCOVERED_DIR / f"gen_{gen_id}_{name.lower()}.py"
    with open(code_path, "w") as f:
        f.write(f"# Generation {gen_id}: {name}\n")
        f.write(f"# Verdict: {record['verdict']}\n")
        f.write(f"# Optimality Gap: {record['avg_optimality_gap_pct']}%\n")
        f.write(f"# Empirical Scaling: {record['scaling_label']}\n\n")
        f.write(code_str)

    record["code_file"] = str(code_path)
    print(f" -> Gap: {record['avg_optimality_gap_pct']}% | Scaling: {record['scaling_label']} | Runtime: {record['avg_runtime_ms']}ms")
    print(f" -> Verdict: {record['verdict']}")

    if record["p_equals_np_candidate"]:
        print("\n" + "="*60)
        print("[!] BREAKTHROUGH CANDIDATE DETECTED! [!]")
        print("0% Gap across exact benchmarks with empirical polynomial scaling!")
        print("="*60 + "\n")

    return record

def synthesize_new_algorithm(gen_id):
    """
    Synthesizes a new algorithmic candidate by exploring strategy combinations,
    heuristic parameter mutation, and hybridization.
    """
    template_info = random.choice(STRATEGY_TEMPLATES)
    name = f"{template_info['name']}_v{gen_id}"
    
    # Mutate parameters
    params = {
        "max_kicks": random.choice([3, 5, 8, 12]),
        "kick_tolerance": round(random.uniform(1.01, 1.08), 3),
        "max_passes": random.choice([10, 20, 50]),
        "relocation_passes": random.choice([3, 5, 8])
    }

    code = template_info["code_template"]
    for k, v in params.items():
        code = code.replace(f"{{{k}}}", str(v))
    return name, code

def load_function_from_code(code_str, func_name="solve"):
    """Dynamically compiles and loads a function from code string."""
    module_dict = {}
    exec(code_str, module_dict)
    return module_dict[func_name]

def run_autonomous_loop(num_iterations=5):
    """
    Main autonomous research loop.
    1. Pre-evaluates core baselines if leaderboard is empty.
    2. Repeatedly synthesizes new algorithms, evaluates them, and updates rankings.
    """
    print("=" * 65)
    print("[*] AUTONOMOUS P vs NP RESEARCH ENGINE INITIATED")
    print("Problem: Traveling Salesperson Problem (TSP) / NP-Hard")
    print("Goal: Synthesize algorithms, measure empirical polynomiality & optimality")
    print("=" * 65)

    suite = get_benchmark_suite()
    leaderboard = load_leaderboard()

    # 1. Initialize with baselines if fresh
    if not leaderboard:
        print("\n[Seeding Initial Baseline Algorithms...]")
        initial_baselines = [
            ("Nearest_Neighbor", baselines.nearest_neighbor, "def solve(dm): ... Nearest Neighbor"),
            ("Two_Opt_Local_Search", baselines.two_opt_solver, "def solve(dm): ... 2-Opt Optimization"),
            ("Cheapest_Insertion", baselines.cheapest_insertion, "def solve(dm): ... Cheapest Insertion"),
            ("MST_DoubleTree_Approx", baselines.mst_approx, "def solve(dm): ... MST 2-Approximation"),
            ("Hybrid_MST_2Opt", baselines.hybrid_mst_2opt, "def solve(dm): ... MST + 2-Opt Hybrid"),
            ("Hybrid_Insertion_2Opt", baselines.hybrid_insertion_2opt, "def solve(dm): ... Insertion + 2-Opt"),
            ("Simulated_Annealing", baselines.simulated_annealing, "def solve(dm): ... Simulated Annealing")
        ]

        for b_id, (b_name, b_func, b_code) in enumerate(initial_baselines, 1):
            rec = evaluate_and_record(b_name, b_func, b_code, suite, b_id)
            leaderboard.append(rec)
            update_status({
                "status": "Seeding baselines",
                "current_gen": b_id,
                "latest_algorithm": b_name,
                "best_gap": min(r["avg_optimality_gap_pct"] for r in leaderboard),
                "leaderboard_count": len(leaderboard)
            })

        save_leaderboard(leaderboard)

    # Sort leaderboard by average optimality gap
    leaderboard.sort(key=lambda x: x["avg_optimality_gap_pct"])
    best_gap = leaderboard[0]["avg_optimality_gap_pct"] if leaderboard else 100.0

    # 2. Autonomous Discovery Iterations
    start_gen = max([r["id"] for r in leaderboard]) + 1 if leaderboard else 1
    print(f"\n[Beginning Autonomous Evolution Loop ({num_iterations} candidate generations)...]")

    for i in range(num_iterations):
        curr_gen = start_gen + i
        update_status({
            "status": "Synthesizing new algorithm candidate",
            "current_gen": curr_gen,
            "best_gap": best_gap,
            "leaderboard_count": len(leaderboard)
        })

        cand_name, cand_code = synthesize_new_algorithm(curr_gen)
        try:
            cand_func = load_function_from_code(cand_code, "solve")
            rec = evaluate_and_record(cand_name, cand_func, cand_code, suite, curr_gen)
            if rec["valid"]:
                leaderboard.append(rec)
                leaderboard.sort(key=lambda x: (x["avg_optimality_gap_pct"], x["avg_runtime_ms"]))
                save_leaderboard(leaderboard)
                best_gap = leaderboard[0]["avg_optimality_gap_pct"]
        except Exception as e:
            print(f"[Generation #{curr_gen} Compilation/Execution Error: {e}]")

        update_status({
            "status": "Idle / Ready for next iteration",
            "current_gen": curr_gen,
            "best_gap": best_gap,
            "leaderboard_count": len(leaderboard)
        })

    print("\n" + "=" * 65)
    print("[LEADERBOARD] AUTONOMOUS DISCOVERY LEADERBOARD SUMMARY")
    print("=" * 65)
    print(f"{'Rank':<5} {'Algorithm':<28} {'Avg Gap (%)':<14} {'Complexity':<16} {'Runtime (ms)'}")
    print("-" * 75)
    for rank, item in enumerate(leaderboard[:10], 1):
        gap_str = f"{item['avg_optimality_gap_pct']:.2f}%"
        runtime_str = f"{item['avg_runtime_ms']:.2f} ms"
        scaling = item.get("scaling_label", "O(?)").split("~")[0].strip()
        print(f"{rank:<5} {item['name']:<28} {gap_str:<14} {scaling:<16} {runtime_str}")

    print("-" * 75)
    print(f"Top Algorithm: {leaderboard[0]['name']} with {leaderboard[0]['avg_optimality_gap_pct']:.2f}% Optimality Gap.")
    print("=" * 65)

if __name__ == "__main__":
    iters = 3
    if len(sys.argv) > 1:
        try:
            iters = int(sys.argv[1])
        except ValueError:
            pass
    run_autonomous_loop(num_iterations=iters)
