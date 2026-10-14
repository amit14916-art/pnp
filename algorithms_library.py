import numpy as np
import random
import math

def calculate_tour_length(tour, dist_matrix):
    n = len(tour)
    return sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))

# 1. Nearest Neighbor
def nearest_neighbor(dist_matrix):
    n = len(dist_matrix)
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda c: dist_matrix[current][c])
        unvisited.remove(next_city)
        tour.append(next_city)
        current = next_city
    return tour

# 2. 2-Opt Optimizer
def two_opt_solver(dist_matrix, initial_tour=None):
    n = len(dist_matrix)
    if initial_tour is None:
        tour = nearest_neighbor(dist_matrix)
    else:
        tour = list(initial_tour)

    best_dist = calculate_tour_length(tour, dist_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                if j - i == 1:
                    continue
                if i == 1 and j == n:
                    continue
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_dist = calculate_tour_length(new_tour, dist_matrix)
                if new_dist < best_dist - 1e-6:
                    tour = new_tour
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return tour

# 3. Cheapest Insertion Heuristic (O(N^2) greedy construction)
def cheapest_insertion(dist_matrix):
    n = len(dist_matrix)
    if n <= 3:
        return list(range(n))

    # Start with a triangle of 3 cities
    tour = [0, 1, 2]
    unvisited = set(range(3, n))

    while unvisited:
        best_city = -1
        best_idx = -1
        min_extra_dist = float('inf')

        for city in unvisited:
            for i in range(len(tour)):
                u = tour[i]
                v = tour[(i + 1) % len(tour)]
                extra = dist_matrix[u][city] + dist_matrix[city][v] - dist_matrix[u][v]
                if extra < min_extra_dist:
                    min_extra_dist = extra
                    best_city = city
                    best_idx = i + 1

        tour.insert(best_idx, best_city)
        unvisited.remove(best_city)

    return tour

# 4. Simulated Annealing (Metaheuristic to escape local minima)
def simulated_annealing(dist_matrix, max_iters=2500, initial_temp=100.0, cooling_rate=0.995):
    n = len(dist_matrix)
    current_tour = nearest_neighbor(dist_matrix)
    current_dist = calculate_tour_length(current_tour, dist_matrix)
    best_tour = list(current_tour)
    best_dist = current_dist

    temp = initial_temp

    for _ in range(max_iters):
        if temp < 1e-3:
            break
        i = random.randint(1, n - 2)
        j = random.randint(i + 1, n - 1)
        
        cand_tour = current_tour[:i] + current_tour[i:j][::-1] + current_tour[j:]
        cand_dist = calculate_tour_length(cand_tour, dist_matrix)

        delta = cand_dist - current_dist
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-5)):
            current_tour = cand_tour
            current_dist = cand_dist
            if current_dist < best_dist:
                best_tour = list(current_tour)
                best_dist = current_dist

        temp *= cooling_rate

    return best_tour

# 5. MST + Double-Tree Shortcut (2-Approximation)
def mst_approx(dist_matrix):
    n = len(dist_matrix)
    # Prim's Algorithm to find MST
    visited = [False] * n
    min_edge = [float('inf')] * n
    parent = [-1] * n
    min_edge[0] = 0

    for _ in range(n):
        u = -1
        for i in range(n):
            if not visited[i] and (u == -1 or min_edge[i] < min_edge[u]):
                u = i
        visited[u] = True

        for v in range(n):
            if not visited[v] and dist_matrix[u][v] < min_edge[v]:
                min_edge[v] = dist_matrix[u][v]
                parent[v] = u

    # Adjacency list of MST
    adj = {i: [] for i in range(n)}
    for v in range(1, n):
        u = parent[v]
        adj[u].append(v)
        adj[v].append(u)

    # DFS traversal for shortcutting
    tour = []
    seen = set()
    def dfs(node):
        seen.add(node)
        tour.append(node)
        for nxt in adj[node]:
            if nxt not in seen:
                dfs(nxt)

    dfs(0)
    return tour

# 6. Hybrid: MST Initialization + 2-Opt Refinement
def hybrid_mst_2opt(dist_matrix):
    init = mst_approx(dist_matrix)
    return two_opt_solver(dist_matrix, initial_tour=init)

# 7. Hybrid: Cheapest Insertion + 2-Opt Refinement
def hybrid_insertion_2opt(dist_matrix):
    init = cheapest_insertion(dist_matrix)
    return two_opt_solver(dist_matrix, initial_tour=init)
