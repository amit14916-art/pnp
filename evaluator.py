import time
import numpy as np

def calculate_tour_length(tour, dist_matrix):
    n = len(tour)
    return sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))

def verify_valid_tour(tour, expected_n):
    if not isinstance(tour, (list, tuple, np.ndarray)):
        return False, "Tour is not an iterable list or array."
    if len(tour) != expected_n:
        return False, f"Tour length ({len(tour)}) does not match number of cities ({expected_n})."
    tour_set = set(tour)
    if len(tour_set) != expected_n:
        return False, f"Duplicate cities detected in tour (unique count {len(tour_set)} != {expected_n})."
    if min(tour_set) != 0 or max(tour_set) != expected_n - 1:
        return False, "City indices must range from 0 to N-1."
    return True, "Valid tour."

def evaluate_algorithm(algo_func, benchmark_suite, timeout_per_test=5.0):
    """
    Evaluates a candidate algorithm across the benchmark suite:
    1. Solution validity (Hamiltonian cycle)
    2. Optimality Gap vs Held-Karp ground truth (%)
    3. Empirical scaling exponent (beta in T ~ N^beta)
    4. Adversarial trap resistance
    5. P = NP criteria check (0% gap & polynomial scaling)
    """
    results = {
        "valid": True,
        "error": None,
        "exact_gaps": [],
        "avg_optimality_gap_pct": 0.0,
        "scaling_points": [],
        "empirical_beta": None,
        "scaling_label": "Unknown",
        "avg_runtime_ms": 0.0,
        "adversarial_scores": [],
        "p_equals_np_candidate": False,
        "verdict": ""
    }

    total_time_ms = 0.0
    run_count = 0

    # 1. Exact Suite Testing (Ground Truth Comparison)
    for test in benchmark_suite["exact"]:
        n = test["num_cities"]
        dm = test["dist_matrix"]
        opt_cost = test["optimal_cost"]

        t0 = time.perf_counter()
        try:
            tour = algo_func(dm)
        except Exception as e:
            results["valid"] = False
            results["error"] = f"Runtime error on {test['name']}: {str(e)}"
            return results

        dt_ms = (time.perf_counter() - t0) * 1000.0
        total_time_ms += dt_ms
        run_count += 1

        is_valid, msg = verify_valid_tour(tour, n)
        if not is_valid:
            results["valid"] = False
            results["error"] = f"Invalid tour on {test['name']}: {msg}"
            return results

        cost = calculate_tour_length(tour, dm)
        gap = ((cost - opt_cost) / opt_cost) * 100.0
        results["exact_gaps"].append({
            "test": test["name"],
            "n": n,
            "cost": round(cost, 2),
            "opt_cost": round(opt_cost, 2),
            "gap_pct": round(gap, 2),
            "time_ms": round(dt_ms, 2)
        })

    avg_gap = np.mean([g["gap_pct"] for g in results["exact_gaps"]])
    results["avg_optimality_gap_pct"] = round(avg_gap, 2)

    # 2. Scaling Suite Testing (Empirical Time Complexity)
    scaling_ns = []
    scaling_times = []

    for test in benchmark_suite["scaling"]:
        n = test["num_cities"]
        dm = test["dist_matrix"]

        t0 = time.perf_counter()
        try:
            tour = algo_func(dm)
        except Exception as e:
            results["valid"] = False
            results["error"] = f"Runtime error on {test['name']}: {str(e)}"
            return results

        dt_sec = max(time.perf_counter() - t0, 1e-6)
        total_time_ms += dt_sec * 1000.0
        run_count += 1

        is_valid, _ = verify_valid_tour(tour, n)
        if not is_valid:
            results["valid"] = False
            results["error"] = f"Invalid tour on {test['name']}"
            return results

        scaling_ns.append(n)
        scaling_times.append(dt_sec)
        results["scaling_points"].append({
            "n": n,
            "time_ms": round(dt_sec * 1000.0, 3)
        })

    # Linear regression in log-log space: log(Time) = beta * log(N) + C
    if len(scaling_ns) >= 3:
        log_n = np.log(scaling_ns)
        log_t = np.log(scaling_times)
        # Polyfit slope is beta
        slope, _ = np.polyfit(log_n, log_t, 1)
        results["empirical_beta"] = round(float(slope), 2)
        if slope < 1.4:
            results["scaling_label"] = f"O(N^{results['empirical_beta']}) ~ Linear/Quasilinear"
        elif slope < 2.4:
            results["scaling_label"] = f"O(N^{results['empirical_beta']}) ~ Quadratic"
        elif slope < 3.4:
            results["scaling_label"] = f"O(N^{results['empirical_beta']}) ~ Cubic"
        else:
            results["scaling_label"] = f"O(N^{results['empirical_beta']}) ~ High Polynomial / Exponential"

    # 3. Adversarial Suite Testing
    for test in benchmark_suite["adversarial"]:
        n = test["num_cities"]
        dm = test["dist_matrix"]
        t0 = time.perf_counter()
        tour = algo_func(dm)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        cost = calculate_tour_length(tour, dm)
        results["adversarial_scores"].append({
            "test": test["name"],
            "n": n,
            "cost": round(cost, 2),
            "time_ms": round(dt_ms, 2)
        })

    results["avg_runtime_ms"] = round(total_time_ms / max(run_count, 1), 2)

    # 4. P vs NP Verdict
    # An algorithm would prove P = NP if it is:
    # a) Exactly optimal (gap <= 0.01%) across all exact tests
    # b) Scaled polynomially (beta <= 4.0) with no exponential blowup
    if results["avg_optimality_gap_pct"] <= 0.01 and (results["empirical_beta"] is not None and results["empirical_beta"] <= 4.0):
        results["p_equals_np_candidate"] = True
        results["verdict"] = "PROBABLE POLYNOMIAL EXACT SOLVER FOUND! (Requires formal proof verification)"
    elif results["avg_optimality_gap_pct"] <= 1.0:
        results["verdict"] = "High-Precision Polynomial Heuristic (Near-Optimal)"
    elif results["avg_optimality_gap_pct"] <= 5.0:
        results["verdict"] = "Competitive Polynomial Heuristic"
    else:
        results["verdict"] = "Fast Greedy / Approximate Heuristic (Suboptimal)"

    return results
