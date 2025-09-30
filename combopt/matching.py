
from __future__ import annotations
from typing import List, Tuple, Dict, Any

# We implement the Assignment (minimum cost bipartite matching) problem.
# Strategies: Greedy (row-wise minimal with conflict resolution), Divide & Conquer (block split),
# Dynamic Programming (bitmask DP), Backtracking, and Hungarian (optimal, O(n^3)).

def assignment_greedy(cost: List[List[float]]):
    n = len(cost)
    assigned_cols = set()
    assign = [-1]*n
    for i in range(n):
        best_j, best_c = None, float("inf")
        for j in range(n):
            if j in assigned_cols: continue
            if cost[i][j] < best_c:
                best_c, best_j = cost[i][j], j
        assign[i] = best_j
        assigned_cols.add(best_j)
    val = sum(cost[i][assign[i]] for i in range(n))
    return assign, val, {"strategy": "greedy_row_min"}

def assignment_divide_conquer(cost: List[List[float]]):
    # simple recursive split: solve top-left and bottom-right blocks separately if they are "cheap",
    # fallback to greedy if blocks mismatch; heuristic, not guaranteed optimal.
    n = len(cost)
    if n <= 2:
        return assignment_greedy(cost)
    mid = n//2
    top = [row[:mid] for row in cost[:mid]]
    bottom = [row[mid:] for row in cost[mid:]]
    a1, v1, _ = assignment_greedy(top)
    a2, v2, _ = assignment_greedy(bottom)
    # combine
    assign = [-1]*n
    for i in range(mid):
        assign[i] = a1[i]
    for i in range(mid, n):
        assign[i] = mid + a2[i-mid]
    val = sum(cost[i][assign[i]] for i in range(n))
    return assign, val, {"strategy": "divide_and_conquer_block_heuristic"}

def assignment_dp(cost: List[List[float]]):
    n = len(cost)
    DP = [float("inf")] * (1<<n)
    parent = [(-1,-1)] * (1<<n)  # (prev_mask, chosen_j)
    DP[0] = 0.0
    for i in range(n):
        for mask in range(1<<n):
            if bin(mask).count("1") != i: 
                continue
            if DP[mask] == float("inf"):
                continue
            for j in range(n):
                if not (mask & (1<<j)):
                    m2 = mask | (1<<j)
                    val = DP[mask] + cost[i][j]
                    if val < DP[m2]:
                        DP[m2] = val
                        parent[m2] = (mask, j)
    full = (1<<n)-1
    # reconstruct
    assign = [-1]*n
    mask = full
    i = n-1
    while mask:
        pm, j = parent[mask]
        assign[i] = j
        i -= 1
        mask = pm
    return assign, DP[full], {"strategy": "dp_bitmask"}

def assignment_backtracking(cost: List[List[float]]):
    n = len(cost)
    best = {"v": float("inf"), "a": None}
    used = [False]*n
    assign = [-1]*n
    def bound(i, curr):
        # optimistic: add minimal in each remaining row
        add = 0.0
        for r in range(i, n):
            add += min(cost[r][c] for c in range(n) if not used[c])
        return curr + add
    def dfs(i, curr):
        if curr >= best["v"]:
            return
        if i == n:
            best["v"] = curr
            best["a"] = assign[:]
            return
        rows = list(range(n))
        # choose row i; try cheapest columns first
        cols = sorted([c for c in range(n) if not used[c]], key=lambda c: cost[i][c])
        for c in cols:
            used[c] = True
            assign[i] = c
            b = bound(i+1, curr + cost[i][c]) if i+1 < n else curr + cost[i][c]
            if b < best["v"]:
                dfs(i+1, curr + cost[i][c])
            used[c] = False
            assign[i] = -1
    dfs(0, 0.0)
    return best["a"], best["v"], {"strategy": "backtracking_with_bound"}

def assignment_hungarian(cost: List[List[float]]):
    # Standard Hungarian algorithm for square cost matrices (minimization).
    n = len(cost)
    u = [0]*(n+1)
    v = [0]*(n+1)
    p = [0]*(n+1)
    way = [0]*(n+1)
    for i in range(1, n+1):
        p[0] = i
        j0 = 0
        minv = [float("inf")]*(n+1)
        used = [False]*(n+1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n+1):
                if not used[j]:
                    cur = cost[i0-1][j-1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = [-1]*n
    for j in range(1, n+1):
        assign[p[j]-1] = j-1
    val = sum(cost[i][assign[i]] for i in range(n))
    return assign, val, {"strategy": "hungarian"}
