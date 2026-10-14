# Autonomous P vs NP Research Engine & TSP Lab

Yeh ek autonomous research system hai jo **P vs NP** (specifically Travelling Salesperson Problem - TSP) ke liye automatically naye algorithmic heuristics synthesize karta hai, unhein mathematically benchmark karta hai, aur unki **Empirical Time Complexity** $\mathcal{O}(N^\beta)$ aur **Optimality Gap** measure karta hai.

---

## 🔬 Core Components

1. **[`benchmark.py`](file:///c:/Users/pc/OneDrive/Desktop/P=NP/benchmark.py)**:
   - **Exact Ground Truth**: Held-Karp Dynamic Programming ($\mathcal{O}(2^N N^2)$) ke zariye exact optimal tour length calculate karta hai ($N=8, 10, 12$).
   - **Scaling Suite**: $N=20, 40, 80, 160$ par time complexity curve measure karta hai.
   - **Adversarial Suite**: Clustered aur circular traps jo greedy heuristics ko fool karne ke liye design kiye gaye hain.

2. **[`evaluator.py`](file:///c:/Users/pc/OneDrive/Desktop/P=NP/evaluator.py)**:
   - Validates Hamiltonian cycles (no duplicate/missing cities).
   - Computes **Optimality Gap (%)** vs exact ground truth.
   - Computes **Empirical Scaling Exponent** ($\beta$ via $\log(T) \approx \beta \log(N) + C$).
   - Flags breakthrough candidates if Gap is 0.00% across all exact instances with polynomial scaling ($\beta \le 4.0$).

3. **[`autonomous_researcher.py`](file:///c:/Users/pc/OneDrive/Desktop/P=NP/autonomous_researcher.py)**:
   - **Autonomous Evolution Engine**:
     - Naye algorithmic candidates banata hai (`Adaptive_VNS`, `Guided_Local_Search`, `Convex_Hull_Angle_Insertion`, etc.).
     - Har candidate ko sandbox me test karke [`discovered_algorithms/`](file:///c:/Users/pc/OneDrive/Desktop/P=NP/discovered_algorithms) me save karta hai.
     - Results aur metrics ko [`leaderboard.json`](file:///c:/Users/pc/OneDrive/Desktop/P=NP/leaderboard.json) me live update karta hai.

4. **[`index.html`](file:///c:/Users/pc/OneDrive/Desktop/P=NP/index.html)**:
   - Interactive Visualizer aur Web Lab jisme:
     - Discovered algorithms ko direct test kar sakte hain (Cheapest Insertion, 2-Opt, Adaptive VNS, Simulated Annealing, MST).
     - Live Leaderboard modal.
     - Custom city placement and real-time topology adjustments.

---

## 🏆 Current Discovered Leaderboard

| Rank | Algorithm Name | Avg Optimality Gap | Empirical Scaling | Runtime (ms) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Cheapest Insertion** | **0.00%** | $\mathcal{O}(N^{2.60})$ (Cubic) | 173.92 ms | Near-Optimal Exact Heuristic |
| **2** | **Hybrid Insertion + 2Opt** | **0.00%** | $\mathcal{O}(N^{3.47})$ | 1403.05 ms | Near-Optimal Exact Heuristic |
| **3** | **2-Opt Local Search** | **0.00%** | $\mathcal{O}(N^{3.78})$ | 3025.68 ms | Near-Optimal Exact Heuristic |
| **4** | **Adaptive VNS (Gen #8)** | **0.00%** | $\mathcal{O}(N^{3.40})$ | 8318.12 ms | Near-Optimal Exact Heuristic |
| **5** | **Hybrid MST + 2Opt** | **0.97%** | $\mathcal{O}(N^{3.87})$ | 7067.79 ms | High-Precision Heuristic |
| **6** | **Convex Hull Angle (Gen #9)**| **3.61%** | $\mathcal{O}(N^{0.80})$ (Linear) | 92.85 ms | Competitive Linear Heuristic |
| **7** | **Simulated Annealing** | **4.41%** | $\mathcal{O}(N^{0.89})$ (Linear) | 73.51 ms | Competitive Linear Heuristic |
| **8** | **Nearest Neighbor** | **7.37%** | $\mathcal{O}(N^{1.81})$ (Quadratic) | 1.51 ms | Fast Suboptimal Heuristic |
| **9** | **MST Double-Tree** | **12.62%** | $\mathcal{O}(N^{1.86})$ (Quadratic) | 2.80 ms | Guaranteed 2-Approximation |

---

## 🚀 Autonomous Loop Kaise Run Karein?

Terminal me naye algorithms generate aur test karne ke liye:

```powershell
# 5 naye algorithms synthesize aur evaluate karein
.venv\Scripts\python autonomous_researcher.py 5
```

Har candidate automatically [`discovered_algorithms/`](file:///c:/Users/pc/OneDrive/Desktop/P=NP/discovered_algorithms) folder me save hota hai aur [`leaderboard.json`](file:///c:/Users/pc/OneDrive/Desktop/P=NP/leaderboard.json) me rank hota hai!
