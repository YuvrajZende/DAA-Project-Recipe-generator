
from __future__ import annotations
import argparse, csv, json, sys, os
from typing import List, Tuple
from combopt.utils import Result, seed_everything
from combopt import tsp, knapsack, matching

def load_tsp_csv(path: str):
    pts = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append((float(row["x"]), float(row["y"])))
    return pts

def load_knapsack_csv(path: str):
    items = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            items.append((int(row["value"]), int(row["weight"])))
    return items

def load_assignment_csv(path: str):
    # CSV with n rows, n columns of costs (no header)
    mat = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        for row in r:
            mat.append([float(x) for x in row])
    # ensure square
    n = len(mat)
    assert all(len(row)==n for row in mat), "Cost matrix must be square"
    return mat

def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Strategy Combinatorial Optimization")
    parser.add_argument("--problem", choices=["tsp","knapsack","assignment"], required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--capacity", type=int, default=None, help="Knapsack capacity (required for knapsack)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    if args.problem == "tsp":
        pts = load_tsp_csv(args.data)
        algos = {
            "greedy": tsp.tsp_greedy_nn_2opt,
            "dnc": tsp.tsp_divide_conquer,
            "dp": tsp.tsp_dp_held_karp,
            "bt": tsp.tsp_backtracking,
            "bnb": tsp.tsp_branch_and_bound,
        }
        if args.algo not in algos:
            sys.exit(f"Unknown TSP algo: {args.algo}")

        res = algos[args.algo](pts)

        # --- fix: support Result or tuple return ---
        if hasattr(res, "__dict__"):   # Result object
            out = res.__dict__
        elif isinstance(res, tuple) and len(res) == 3:
            sol, obj, meta = res
            out = {"solution": sol, "objective": obj, "meta": meta}
        else:
            out = {"result": res}

        print(json.dumps(out, indent=2))


    elif args.problem == "knapsack":
        assert args.capacity is not None, "Provide --capacity for knapsack"
        items = load_knapsack_csv(args.data)
        algos = {
            "greedy": knapsack.knapsack_greedy,
            "dnc": knapsack.knapsack_divide_conquer,
            "dp": knapsack.knapsack_dp,
            "bt": knapsack.knapsack_backtracking,
            "bnb": knapsack.knapsack_branch_and_bound,
        }
        if args.algo not in algos: sys.exit(f"Unknown Knapsack algo: {args.algo}")
        take, val, meta = algos[args.algo](items, args.capacity)
        print(json.dumps({"take": take, "value": val, "meta": meta}, indent=2))

    else: # assignment
        mat = load_assignment_csv(args.data)
        algos = {
            "greedy": matching.assignment_greedy,
            "dnc": matching.assignment_divide_conquer,
            "dp": matching.assignment_dp,
            "bt": matching.assignment_backtracking,
            "hungarian": matching.assignment_hungarian,
        }
        if args.algo not in algos: sys.exit(f"Unknown Assignment algo: {args.algo}")
        assign, val, meta = algos[args.algo](mat)
        print(json.dumps({"assign": assign, "value": val, "meta": meta}, indent=2))

if __name__ == "__main__":
    main()
