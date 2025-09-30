
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import heapq

Item = Tuple[int, int]  # (value, weight)

def _value(items: List[Item], take: List[int]):
    v = 0
    for i, x in enumerate(take):
        if x: v += items[i][0]
    return v

# ---------- Greedy by value/weight ratio (approx) ----------
def knapsack_greedy(items: List[Item], capacity: int):
    idx = list(range(len(items)))
    idx.sort(key=lambda i: items[i][0]/items[i][1], reverse=True)
    take = [0]*len(items)
    w = 0
    for i in idx:
        if w + items[i][1] <= capacity:
            take[i] = 1
            w += items[i][1]
    val = _value(items, take)
    return take, val, {"strategy": "greedy_ratio"}

# ---------- Divide & Conquer (meet-in-the-middle) ----------
def knapsack_divide_conquer(items: List[Item], capacity: int):
    n = len(items)
    A = items[:n//2]
    B = items[n//2:]
    def enum_half(arr):
        res = []
        m = len(arr)
        for mask in range(1<<m):
            w = v = 0
            for i in range(m):
                if mask & (1<<i):
                    v += arr[i][0]; w += arr[i][1]
            res.append((w, v, mask))
        res.sort()
        # prune dominated
        pruned = []
        best_v = -1
        for w, v, msk in res:
            if v > best_v:
                pruned.append((w,v,msk))
                best_v = v
        return pruned
    EA, EB = enum_half(A), enum_half(B)
    best = (0, 0, 0)  # value, maskA, maskB
    j = len(EB)-1
    for wa, va, ma in EA:
        if wa > capacity: continue
        rem = capacity - wa
        # find best EB with weight <= rem
        lo, hi = 0, len(EB)-1
        pos = -1
        while lo <= hi:
            mid = (lo+hi)//2
            if EB[mid][0] <= rem:
                pos = mid
                lo = mid+1
            else:
                hi = mid-1
        if pos >= 0:
            vb = EB[pos][1]
            if va + vb > best[0]:
                best = (va+vb, ma, EB[pos][2])
    # build take vector
    take = [0]*n
    ma, mb = best[1], best[2]
    for i in range(len(A)):
        if ma & (1<<i): take[i] = 1
    for i in range(len(B)):
        if mb & (1<<i): take[len(A)+i] = 1
    return take, best[0], {"strategy": "divide_and_conquer_meet_in_middle"}

# ---------- Dynamic Programming (0/1 exact) ----------
def knapsack_dp(items: List[Item], capacity: int):
    n = len(items)
    DP = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        v, w = items[i-1]
        for c in range(capacity+1):
            DP[i][c] = DP[i-1][c]
            if w <= c:
                DP[i][c] = max(DP[i][c], DP[i-1][c-w] + v)
    # reconstruct
    take = [0]*n
    c = capacity
    for i in range(n, 0, -1):
        if DP[i][c] != DP[i-1][c]:
            take[i-1] = 1
            c -= items[i-1][1]
    return take, DP[n][capacity], {"strategy": "dp_table"}

# ---------- Backtracking with fractional bound ----------
def knapsack_backtracking(items: List[Item], capacity: int):
    n = len(items)
    idx = list(range(n))
    idx.sort(key=lambda i: items[i][0]/items[i][1], reverse=True)
    items_sorted = [items[i] for i in idx]
    best_val, best_take = 0, [0]*n
    take = [0]*n
    def bound(i, w, v):
        # fractional knapsack bound
        cap = capacity - w
        bound_val = v
        while i < n and items_sorted[i][1] <= cap:
            bound_val += items_sorted[i][0]
            cap -= items_sorted[i][1]
            i += 1
        if i < n:
            bound_val += items_sorted[i][0] * cap / items_sorted[i][1]
        return bound_val
    def dfs(i, w, v):
        nonlocal best_val, best_take
        if w > capacity: return
        if i == n:
            if v > best_val:
                best_val = v
                best_take = take[:]
            return
        if bound(i, w, v) <= best_val: return
        # choose item i
        take[idx[i]] = 1
        dfs(i+1, w + items_sorted[i][1], v + items_sorted[i][0])
        take[idx[i]] = 0
        dfs(i+1, w, v)
    dfs(0, 0, 0)
    return best_take, best_val, {"strategy": "backtracking_fractional_bound"}

# ---------- Branch & Bound using priority queue ----------
def knapsack_branch_and_bound(items: List[Item], capacity: int):
    n = len(items)
    idx = list(range(n))
    idx.sort(key=lambda i: items[i][0]/items[i][1], reverse=True)
    items_sorted = [items[i] for i in idx]
    def bound(i, w, v):
        cap = capacity - w
        bound_val = v
        while i < n and items_sorted[i][1] <= cap:
            bound_val += items_sorted[i][0]
            cap -= items_sorted[i][1]
            i += 1
        if i < n and items_sorted[i][1] > 0:
            bound_val += items_sorted[i][0] * cap / items_sorted[i][1]
        return bound_val
    best_val, best_take = 0, [0]*n
    # node: (-bound, i, w, v, take_copy)
    pq = [(-bound(0,0,0), 0, 0, 0, [0]*n)]
    while pq:
        nb, i, w, v, take_vec = heapq.heappop(pq)
        b = -nb
        if b <= best_val: 
            continue
        if i == n:
            if v > best_val:
                best_val, best_take = v, take_vec
            continue
        # include
        if w + items_sorted[i][1] <= capacity:
            t1 = take_vec[:]
            t1[idx[i]] = 1
            v1 = v + items_sorted[i][0]
            w1 = w + items_sorted[i][1]
            heapq.heappush(pq, (-bound(i+1, w1, v1), i+1, w1, v1, t1))
        # exclude
        heapq.heappush(pq, (-bound(i+1, w, v), i+1, w, v, take_vec))
    return best_take, best_val, {"strategy": "branch_and_bound"}
