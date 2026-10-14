import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tsp_solver import generate_cities, distance_matrix, calculate_tour_length, nearest_neighbor

def two_opt_with_history(tour, dist_matrix):
    """
    2-Opt run karta hai aur har successful swap ke baad tour ka snapshot save karta hai
    taaki visually animate kiya ja sake.
    """
    best_tour = list(tour)
    best_distance = calculate_tour_length(best_tour, dist_matrix)
    n = len(best_tour)

    history = [(list(best_tour), best_distance, 0, "Initial Greedy Tour")]
    step = 0
    improved = True

    while improved:
        improved = False
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                if j - i == 1:
                    continue
                if i == 1 and j == n:
                    continue
                new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
                new_distance = calculate_tour_length(new_tour, dist_matrix)

                if new_distance < best_distance - 1e-6:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
                    step += 1
                    history.append((list(best_tour), best_distance, step, f"Step {step}: Uncrossed edges ({i}, {j})"))
                    break
            if improved:
                break

    history.append((list(best_tour), best_distance, step, "Finished (2-Optimal Local Minimum)"))
    return best_tour, history

def animate_tsp(cities, history, save_gif=True, gif_path="tsp_optimization.gif", show=True):
    """Matplotlib animation ke through 2-Opt optimization ko step-by-step dikhata hai."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Static city markers
    ax.scatter(cities[:, 0], cities[:, 1], c='crimson', zorder=4, s=60)
    for idx, (x, y) in enumerate(cities):
        ax.annotate(str(idx), (x + 1, y + 1), fontsize=8, alpha=0.7)
    
    line, = ax.plot([], [], c='forestgreen', linewidth=2, zorder=3)
    start_point = ax.scatter([], [], c='gold', s=140, edgecolors='black', zorder=5, label='Start (0)')
    title = ax.set_title("", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc='upper right')

    ax.set_xlim(np.min(cities[:, 0]) - 5, np.max(cities[:, 0]) + 5)
    ax.set_ylim(np.min(cities[:, 1]) - 5, np.max(cities[:, 1]) + 5)

    def init():
        line.set_data([], [])
        return line,

    def update(frame_idx):
        tour, dist, step_num, desc = history[frame_idx]
        path = tour + [tour[0]]
        xs = cities[path, 0]
        ys = cities[path, 1]
        line.set_data(xs, ys)
        
        # Color line steelblue on initial frame, then green
        if frame_idx == 0:
            line.set_color('steelblue')
            line.set_linestyle('--')
        else:
            line.set_color('forestgreen')
            line.set_linestyle('-')

        start_point.set_offsets([[cities[tour[0], 0], cities[tour[0], 1]]])
        title.set_text(f"2-Opt Live Optimization\n{desc} | Distance: {dist:.2f}")
        return line, start_point, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(history), init_func=init, blit=False, interval=400, repeat_delay=2000
    )

    if save_gif:
        try:
            print(f"[Rendering animation to '{gif_path}'...]")
            ani.save(gif_path, writer='pillow', fps=2.5)
            print(f"[Animation successfully saved to '{gif_path}']")
        except Exception as e:
            print(f"[Could not save GIF: {e}]")

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[Notice: GUI window could not open: {e}]")

import sys

if __name__ == "__main__":
    num_cities = 25
    cities = generate_cities(num_cities=num_cities, seed=42)
    dist_matrix = distance_matrix(cities)

    greedy_tour = nearest_neighbor(dist_matrix)
    best_tour, history = two_opt_with_history(greedy_tour, dist_matrix)
    
    print(f"Total 2-Opt Improvement Steps recorded: {len(history) - 2}")
    should_show = "--no-show" not in sys.argv
    animate_tsp(cities, history, save_gif=True, gif_path="tsp_optimization.gif", show=should_show)
