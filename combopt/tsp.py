
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import math, heapq
from .utils import timed, euclid, path_length

# ---------- GREEDY: Nearest Neighbor + 2-Opt polish ----------
def _two_opt(points, tour):
    improved = True
    n = len(tour)
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[j], tour[(j + 1) % n]
                old = euclid(points[a], points[b]) + euclid(points[c], points[d])
                new = euclid(points[a], points[c]) + euclid(points[b], points[d])
                if new + 1e-9 < old:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
    return tour

@timed
def tsp_greedy_nn_2opt(points: List[Tuple[float, float]], start: int = 0):
    n = len(points)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    curr = start
    while unvisited:
        nxt = min(unvisited, key=lambda u: euclid(points[curr], points[u]))
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    tour = _two_opt(points, tour)
    obj = path_length(points, tour)
    return tour, obj, {"strategy": "nearest_neighbor+2opt"}

# ---------- D&C: Split by median x, solve halves, stitch + 2-Opt ----------
@timed
def tsp_divide_conquer(points: List[Tuple[float, float]]):
    n = len(points)
    idx = list(range(n))
    idx.sort(key=lambda i: points[i][0])
    mid = n // 2
    left, right = idx[:mid], idx[mid:]
    def nn(seq):
        if not seq: return []
        unv = set(seq[1:])
        t = [seq[0]]
        curr = seq[0]
        while unv:
            nxt = min(unv, key=lambda u: euclid(points[curr], points[u]))
            t.append(nxt)
            unv.remove(nxt)
            curr = nxt
        return t
    left_tour = nn(left)
    right_tour = nn(right)
    # stitch: connect ends minimizing cross edges
    best = None
    for i in range(len(left_tour)):
        for j in range(len(right_tour)):
            a, b = left_tour[i], left_tour[(i+1)%len(left_tour)]
            c, d = right_tour[j], right_tour[(j+1)%len(right_tour)]
            base = euclid(points[a], points[b]) + euclid(points[c], points[d])
            cross = euclid(points[a], points[c]) + euclid(points[b], points[d])
            if best is None or cross - base < best[0]:
                best = (cross - base, i, j)
    _, i, j = best
    # build combined tour
    left_cycle = left_tour[i+1:] + left_tour[:i+1]
    right_cycle = right_tour[j+1:] + right_tour[:j+1]
    tour = left_cycle + right_cycle
    tour = _two_opt(points, tour)
    obj = path_length(points, tour)
    return tour, obj, {"strategy": "divide_and_conquer+stitch+2opt"}

# ---------- DP: Held-Karp (exact) ----------
@timed
def tsp_dp_held_karp(points: List[Tuple[float, float]]):
    n = len(points)
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = euclid(points[i], points[j])
    # DP[mask][i] = best cost to reach i with visited set mask
    DP = { (1,0): 0.0 }
    parent = {}
    for mask in range(1, 1<<n):
        if not (mask & 1):  # ensure start=0 included
            continue
        for j in range(1, n):
            if not (mask & (1<<j)): continue
            if (mask, j) not in DP: continue
            cost = DP[(mask, j)]
            for k in range(1, n):
                if mask & (1<<k): continue
                new_mask = mask | (1<<k)
                new_cost = cost + dist[j][k]
                if (new_mask, k) not in DP or new_cost < DP[(new_mask, k)]:
                    DP[(new_mask, k)] = new_cost
                    parent[(new_mask, k)] = (j, mask)
    # close the tour
    best_cost, best_end = float("inf"), None
    full_mask = (1<<n) - 1
    for j in range(1,n):
        if (full_mask, j) in DP:
            c = DP[(full_mask, j)] + dist[j][0]
            if c < best_cost:
                best_cost, best_end = c, j
    # reconstruct
    tour = [0]
    mask, j = full_mask, best_end
    seq = []
    while j is not None and ((mask, j) in parent or j == best_end):
        seq.append(j)
        if (mask, j) in parent:
            pj, pmask = parent[(mask, j)]
            mask, j = pmask, pj
        else:
            break
    seq.reverse()
    tour.extend(seq)
    obj = best_cost
    return tour, obj, {"strategy": "held-karp", "n": n}

# ---------- Backtracking with pruning ----------
@timed
def tsp_backtracking(points):
    n = len(points)
    best = {"obj": float("inf"), "tour": None}
    used = [False]*n
    used[0] = True
    order = [0]
    def lower_bound(curr_len, last, remaining):
        # optimistic: add two nearest edges for each remaining node (half since double counted)
        add = 0.0
        for u in remaining:
            # nearest neighbor distance
            d = min(euclid(points[u], points[v]) for v in range(n) if v != u)
            add += d
        return curr_len + add
    def dfs(last, curr_len, depth):
        if curr_len >= best["obj"]: return
        if depth == n:
            total = curr_len + euclid(points[last], points[0])
            if total < best["obj"]:
                best["obj"] = total
                best["tour"] = order[:]
            return
        remaining = [i for i in range(n) if not used[i]]
        if lower_bound(curr_len, last, remaining) >= best["obj"]:
            return
        # try nearest-first heuristic
        remaining.sort(key=lambda u: euclid(points[last], points[u]))
        for u in remaining:
            used[u] = True
            order.append(u)
            dfs(u, curr_len + euclid(points[last], points[u]), depth+1)
            order.pop()
            used[u] = False
    dfs(0, 0.0, 1)
    return best["tour"], best["obj"], {"strategy": "backtracking+pruning"}

# ---------- Branch & Bound (priority queue on bound) ----------
@timed
def tsp_branch_and_bound(points):
    n = len(points)
    # bound: current path length + MST(lower) over remaining + return edge approx (simple nn-based bound)
    # For speed we use simpler bound: current + sum(min outgoing edge) over remaining.
    def min_out_sum(nodes):
        s = 0.0
        for u in nodes:
            best = float("inf")
            for v in range(n):
                if v != u:
                    d = euclid(points[u], points[v])
                    if d < best:
                        best = d
            s += best
        return s
    start_state = (0.0, 0, [0], set(range(1,n)))  # (bound, length, path, remaining)
    best_cost, best_path = float("inf"), None
    pq = [start_state]
    while pq:
        bound, length, path, rem = heapq.heappop(pq)
        if bound >= best_cost: 
            continue
        last = path[-1]
        if not rem:
            cost = length + euclid(points[last], points[0])
            if cost < best_cost:
                best_cost, best_path = cost, path[:]
            continue
        # expand in NN order
        nxts = sorted(rem, key=lambda u: euclid(points[last], points[u]))
        for u in nxts:
            new_len = length + euclid(points[last], points[u])
            new_path = path + [u]
            new_rem = rem - {u}
            new_bound = new_len + 0.5*min_out_sum(new_rem)  # optimistic
            if new_bound < best_cost:
                heapq.heappush(pq, (new_bound, new_len, new_path, new_rem))
    return best_path, best_cost, {"strategy": "branch_and_bound"}
