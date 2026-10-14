import itertools
import numpy as np

def held_karp_exact(dist_matrix):
    """
    Exact TSP solver using Held-Karp dynamic programming with combinatorial subsets.
    Guarantees mathematically optimal tour length for ground truth benchmarks.
    """
    n = len(dist_matrix)
    if n <= 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(dist_matrix[0][1] + dist_matrix[1][0])

    memo = {(1 << i, i): (dist_matrix[0][i], 0) for i in range(1, n)}

    for r in range(2, n):
        for combo in itertools.combinations(range(1, n), r):
            mask = sum(1 << i for i in combo)
            for u in combo:
                prev_mask = mask ^ (1 << u)
                best_cost = float('inf')
                best_parent = -1
                for v in combo:
                    if v == u:
                        continue
                    cost = memo[(prev_mask, v)][0] + dist_matrix[v][u]
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = v
                memo[(mask, u)] = (best_cost, best_parent)

    # Return to city 0
    final_mask = (1 << n) - 2
    best_cost = float('inf')
    last_city = -1
    for u in range(1, n):
        cost = memo[(final_mask, u)][0] + dist_matrix[u][0]
        if cost < best_cost:
            best_cost = cost
            last_city = u

    # Reconstruct optimal tour
    curr_mask = final_mask
    curr_city = last_city
    path = []
    while curr_city != 0:
        path.append(curr_city)
        _, parent = memo[(curr_mask, curr_city)]
        curr_mask = curr_mask ^ (1 << curr_city)
        curr_city = parent

    tour = [0] + list(reversed(path))
    return tour, float(best_cost)

def create_dist_matrix(cities):
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def generate_adversarial_instance(num_cities, seed=42):
    np.random.seed(seed)
    hubs = np.array([[20, 20], [80, 20], [50, 80]])
    cities = []
    cities_per_hub = num_cities // 3
    for i, hub in enumerate(hubs):
        count = cities_per_hub + (num_cities % 3 if i == 0 else 0)
        cluster = hub + np.random.randn(count, 2) * 5.0
        cities.append(cluster)
    return np.vstack(cities)

# Cache so benchmark is generated once
_CACHED_BENCHMARK_SUITE = None

def get_benchmark_suite():
    global _CACHED_BENCHMARK_SUITE
    if _CACHED_BENCHMARK_SUITE is not None:
        return _CACHED_BENCHMARK_SUITE

    suite = {
        "exact": [],
        "scaling": [],
        "adversarial": []
    }

    # 1. Exact Suite (ground truth optimal)
    for n in [8, 10, 12]:
        np.random.seed(100 + n)
        cities = np.random.rand(n, 2) * 100
        dm = create_dist_matrix(cities)
        opt_tour, opt_cost = held_karp_exact(dm)
        suite["exact"].append({
            "name": f"Exact_N{n}",
            "num_cities": n,
            "cities": cities,
            "dist_matrix": dm,
            "optimal_tour": opt_tour,
            "optimal_cost": opt_cost
        })

    # 2. Scaling Suite (for empirical polynomial vs exponential time measurement)
    for n in [20, 40, 80, 160]:
        np.random.seed(200 + n)
        cities = np.random.rand(n, 2) * 100
        dm = create_dist_matrix(cities)
        suite["scaling"].append({
            "name": f"Scale_N{n}",
            "num_cities": n,
            "cities": cities,
            "dist_matrix": dm
        })

    # 3. Adversarial Suite
    for n in [24, 48]:
        cities = generate_adversarial_instance(n, seed=300 + n)
        dm = create_dist_matrix(cities)
        suite["adversarial"].append({
            "name": f"Adversarial_Cluster_N{n}",
            "num_cities": n,
            "cities": cities,
            "dist_matrix": dm
        })

    _CACHED_BENCHMARK_SUITE = suite
    return suite
