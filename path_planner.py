"""
scripts/run_rover_simulation.py
================================
Extended simulation runner with comparison mode:
  - Runs both A* and Dijkstra
  - Runs multiple seeds for statistical comparison
  - Prints a side-by-side performance table
  - Saves all results

Usage
-----
    python scripts/run_rover_simulation.py --grid_size 96 --seeds 42 7 13
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import run_simulation


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(rows: list, headers: list) -> None:
    col_w = [max(len(str(h)), max(len(str(r[i])) for r in rows))
             for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


# ---------------------------------------------------------------------------
# Single run wrapper
# ---------------------------------------------------------------------------

def _run_one(grid_size, seed, algorithm, n_craters, n_waypoints,
             results_dir, verbose):
    t0 = time.time()
    result = run_simulation(
        grid_size   = grid_size,
        seed        = seed,
        algorithm   = algorithm,
        n_craters   = n_craters,
        n_waypoints = n_waypoints,
        results_dir = results_dir,
        verbose     = verbose,
    )
    elapsed = round(time.time() - t0, 2)
    m = result["metrics"]
    r = result["mission_report"]
    return {
        "seed"       : seed,
        "algorithm"  : algorithm.upper(),
        "path_len"   : int(m.get("Path Length (cells)", 0)),
        "tortuosity" : round(m.get("Tortuosity", 0), 3),
        "energy_wh"  : round(m.get("Energy Used (Wh)", 0), 2),
        "coverage"   : round(m.get("Coverage (%)", 0), 1),
        "science"    : round(m.get("Science Score", 0), 1),
        "wps"        : r.get("waypoints_completed", 0),
        "time_s"     : elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="LunaRover AI — Extended Multi-Run Simulation Script"
    )
    p.add_argument("--grid_size",   type=int,   default=96)
    p.add_argument("--seeds",       type=int,   nargs="+", default=[42])
    p.add_argument("--algorithm",   type=str,   default="both",
                   choices=["astar", "dijkstra", "both"])
    p.add_argument("--n_craters",   type=int,   default=20)
    p.add_argument("--n_waypoints", type=int,   default=5)
    p.add_argument("--results_dir", type=str,   default="results")
    p.add_argument("--quiet",       action="store_true")
    args = p.parse_args()

    algorithms = (["astar", "dijkstra"] if args.algorithm == "both"
                  else [args.algorithm])

    all_results = []
    run_id = 0

    for seed in args.seeds:
        for algo in algorithms:
            run_id += 1
            sub_dir = os.path.join(
                args.results_dir, f"run_{run_id:03d}_seed{seed}_{algo}"
            )
            print(f"\n{'='*60}")
            print(f"  Run {run_id} | Seed={seed} | Algo={algo.upper()}")
            print(f"{'='*60}")
            result = _run_one(
                grid_size   = args.grid_size,
                seed        = seed,
                algorithm   = algo,
                n_craters   = args.n_craters,
                n_waypoints = args.n_waypoints,
                results_dir = sub_dir,
                verbose     = not args.quiet,
            )
            all_results.append(result)

    # ------------------------------------------------------------------ #
    # Comparison table
    # ------------------------------------------------------------------ #
    print("\n\n" + "=" * 70)
    print("  MULTI-RUN COMPARISON TABLE")
    print("=" * 70)
    headers = ["Seed", "Algo", "PathLen", "Tortuous",
               "Energy(Wh)", "Cover%", "Science", "WPs", "Time(s)"]
    rows = [
        [r["seed"], r["algorithm"], r["path_len"], r["tortuosity"],
         r["energy_wh"], r["coverage"], r["science"],
         r["wps"], r["time_s"]]
        for r in all_results
    ]
    _print_table(rows, headers)

    # ------------------------------------------------------------------ #
    # Aggregate stats (if more than one run)
    # ------------------------------------------------------------------ #
    if len(all_results) > 1:
        print("\n  Aggregate Statistics")
        print("-" * 40)
        for key, label in [
            ("path_len",   "Path Length (cells)"),
            ("tortuosity", "Tortuosity"),
            ("energy_wh",  "Energy (Wh)"),
            ("coverage",   "Coverage (%)"),
            ("science",    "Science Score"),
        ]:
            vals = [r[key] for r in all_results]
            print(f"  {label:<24}: "
                  f"mean={np.mean(vals):.2f}  "
                  f"std={np.std(vals):.2f}  "
                  f"min={np.min(vals):.2f}  "
                  f"max={np.max(vals):.2f}")

    print(f"\n  Total runs: {run_id}")
    print(f"  Results in: {args.results_dir}/\n")


if __name__ == "__main__":
    main()
