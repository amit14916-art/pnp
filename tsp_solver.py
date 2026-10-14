import numpy as np
import matplotlib.pyplot as plt

def generate_cities(num_cities=25, seed=42):
    """Random 2D coordinates generate karta hai cities ke liye."""
    np.random.seed(seed)
    return np.random.rand(num_cities, 2) * 100

def distance_matrix(cities):
    """Har do city ke beech Euclidean distance calculate karta hai."""
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def calculate_tour_length(tour, dist_matrix):
    """Pure tour ki total distance calculate karta hai (loop included)."""
    return sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def nearest_neighbor(dist_matrix):
    """Greedy approach: Sabse nazdeeki unvisited city chunta hai."""
    n = len(dist_matrix)
    unvisited = set(range(1, n))
    tour = [0]
    
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda city: dist_matrix[current][city])
        unvisited.remove(next_city)
        tour.append(next_city)
        current = next_city
        
    return tour

def two_opt(tour, dist_matrix):
    """
    2-Opt: Edges ko cross/uncross karke optimize karta hai.
    Slicing note: j ko n+1 tak le jaate hain taaki last element (closing edge) 
    bhi reverse hokar uncross ho sake.
    """
    improved = True
    best_tour = list(tour)
    best_distance = calculate_tour_length(best_tour, dist_matrix)
    n = len(best_tour)

    iteration = 0
    while improved:
        improved = False
        iteration += 1
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                if j - i == 1:
                    continue
                if i == 1 and j == n:
                    continue  # Puray tour ko reverse karna identical distance dega
                
                # Segment reverse karke check karo
                new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
                new_distance = calculate_tour_length(new_tour, dist_matrix)
                
                if new_distance < best_distance - 1e-6:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    print(f"[2-Opt Completed in {iteration} outer iterations. Final Distance: {best_distance:.2f}]")
    return best_tour

def plot_comparison(cities, greedy_tour, opt_tour, dist_matrix, save_path="tsp_comparison.png", show=True):
    """Dono solutions ko side-by-side plot karta hai comparison ke liye."""
    d_greedy = calculate_tour_length(greedy_tour, dist_matrix)
    d_opt = calculate_tour_length(opt_tour, dist_matrix)
    savings_pct = ((d_greedy - d_opt) / d_greedy) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Nearest Neighbor (Greedy)
    axes[0].scatter(cities[:, 0], cities[:, 1], c='crimson', zorder=3, s=50)
    for idx, (x, y) in enumerate(cities):
        axes[0].annotate(str(idx), (x + 1, y + 1), fontsize=8, alpha=0.7)
    g_path = greedy_tour + [greedy_tour[0]]
    axes[0].plot(cities[g_path, 0], cities[g_path, 1], c='steelblue', linestyle='--', linewidth=1.5)
    axes[0].scatter([cities[greedy_tour[0], 0]], [cities[greedy_tour[0], 1]], c='gold', s=120, edgecolors='black', zorder=4, label='Start City (0)')
    axes[0].set_title(f"1. Nearest Neighbor (Greedy)\nTotal Distance: {d_greedy:.2f}", fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc='upper right')

    # Plot 2: 2-Opt Optimization
    axes[1].scatter(cities[:, 0], cities[:, 1], c='crimson', zorder=3, s=50)
    for idx, (x, y) in enumerate(cities):
        axes[1].annotate(str(idx), (x + 1, y + 1), fontsize=8, alpha=0.7)
    o_path = opt_tour + [opt_tour[0]]
    axes[1].plot(cities[o_path, 0], cities[o_path, 1], c='forestgreen', linewidth=2)
    axes[1].scatter([cities[opt_tour[0], 0]], [cities[opt_tour[0], 1]], c='gold', s=120, edgecolors='black', zorder=4, label='Start City (0)')
    axes[1].set_title(f"2. After 2-Opt Refinement\nTotal Distance: {d_opt:.2f} (Savings: {savings_pct:.1f}%)", fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc='upper right')

    plt.suptitle("Traveling Salesperson Problem (TSP) Solver & Visualizer", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save image for non-interactive / report viewing
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot saved to '{save_path}']")
    
    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[Notice: Interactive window could not open: {e}. Saved to file instead.]")

import sys

if __name__ == "__main__":
    num_cities = 30
    cities = generate_cities(num_cities=num_cities, seed=10)
    dist_matrix = distance_matrix(cities)

    print(f"--- Solving TSP for {num_cities} cities ---")
    # Solutions
    greedy_tour = nearest_neighbor(dist_matrix)
    d_greedy = calculate_tour_length(greedy_tour, dist_matrix)
    print(f"Nearest Neighbor Tour Length: {d_greedy:.2f}")

    opt_tour = two_opt(greedy_tour, dist_matrix)
    d_opt = calculate_tour_length(opt_tour, dist_matrix)
    print(f"2-Opt Optimized Tour Length: {d_opt:.2f} (Improvement: {((d_greedy-d_opt)/d_greedy)*100:.1f}%)")

    # Display comparison
    should_show = "--no-show" not in sys.argv
    plot_comparison(cities, greedy_tour, opt_tour, dist_matrix, save_path="tsp_comparison.png", show=should_show)

