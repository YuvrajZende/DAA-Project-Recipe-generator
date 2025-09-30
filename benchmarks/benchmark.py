
from __future__ import annotations
import time, csv, json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from combopt.utils import Result, seed_everything
from combopt import tsp, knapsack, matching

HERE = Path(__file__).parent
DATA = HERE / "datasets"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True, parents=True)

def run_tsp_all():
    pts = []
    with open(DATA/"tsp_cities.csv", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append((float(row["x"]), float(row["y"])))
    algos = [
        tsp.tsp_greedy_nn_2opt,
        tsp.tsp_divide_conquer,
        tsp.tsp_dp_held_karp,
        tsp.tsp_backtracking,
        tsp.tsp_branch_and_bound,
    ]
    results: List[Result] = [a(pts) for a in algos]
    return results

def run_knapsack_all(capacity: int = 50):
    items = []
    with open(DATA/"knapsack_items.csv", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            items.append((int(row["value"]), int(row["weight"])))
    algos = [
        ("greedy", lambda: knapsack.knapsack_greedy(items, capacity)),
        ("dnc",    lambda: knapsack.knapsack_divide_conquer(items, capacity)),
        ("dp",     lambda: knapsack.knapsack_dp(items, capacity)),
        ("bt",     lambda: knapsack.knapsack_backtracking(items, capacity)),
        ("bnb",    lambda: knapsack.knapsack_branch_and_bound(items, capacity)),
    ]
    out = []
    for name, fn in algos:
        take, val, meta = fn()
        out.append((name, val, meta))
    return out

def run_assignment_all():
    mat = []
    with open(DATA/"assignment_cost.csv", newline="") as f:
        r = csv.reader(f)
        for row in r:
            mat.append([float(x) for x in row])
    algos = [
        ("greedy", matching.assignment_greedy),
        ("dnc", matching.assignment_divide_conquer),
        ("dp", matching.assignment_dp),
        ("bt", matching.assignment_backtracking),
        ("hungarian", matching.assignment_hungarian),
    ]
    out = []
    for name, fn in algos:
        a, v, meta = fn(mat)
        out.append((name, v, meta))
    return out

def barplot(values, title, ylabel, fname):
    names = [x[0] for x in values]
    vals = [x[1] for x in values]
    plt.figure()
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(OUT/fname, bbox_inches="tight")
    plt.close()

def main():
    seed_everything(42)

    # === TSP ===
    tsp_res = run_tsp_all()
    with open(OUT/"tsp_results.json", "w") as f:
        json.dump([r.__dict__ for r in tsp_res], f, indent=2)

    print("\n=== Traveling Salesman Problem (TSP) ===")
    for r in tsp_res:
        time_str = f"{r.time:.4f}s" if hasattr(r, "time") else "N/A"
        print(f"{r.name:20s} | Tour Length = {r.objective:.2f} | Time = {time_str}")


    barplot([(r.name, r.objective) for r in tsp_res],
            "TSP Tour Length by Strategy", "Tour Length", "tsp_lengths.png")

    # === Knapsack ===
    knap = run_knapsack_all(capacity=50)
    with open(OUT/"knapsack_results.json", "w") as f:
        json.dump([{"name": n, "value": v, "meta": m} for (n,v,m) in knap], f, indent=2)

    print("\n=== Knapsack Problem (capacity=50) ===")
    for n, v, m in knap:
        print(f"{n:10s} | Value = {v} | Meta = {m}")

    barplot(knap,
            "Knapsack Value by Strategy (capacity=50)", "Total Value", "knapsack_values.png")

    # === Assignment ===
    assign = run_assignment_all()
    with open(OUT/"assignment_results.json", "w") as f:
        json.dump([{"name": n, "value": v, "meta": m} for (n,v,m) in assign], f, indent=2)

    print("\n=== Assignment Problem ===")
    for n, v, m in assign:
        print(f"{n:10s} | Cost = {v:.2f} | Meta = {m}")

    barplot([(n, -v) for (n,v,_) in assign],
            "Assignment (lower cost better) — negative for visualization", "- Cost", "assignment_costs.png")

    print("\n✅ Benchmark complete. See 'benchmarks/out/' for plots and JSON.\n")

if __name__ == "__main__":
    main()
