
from __future__ import annotations
import math, time, random
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict, Any

@dataclass
class Result:
    name: str
    objective: float
    solution: Any
    elapsed_sec: float
    meta: Dict[str, Any]

def timed(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        sol, obj, meta = func(*args, **kwargs)
        t1 = time.perf_counter()
        return Result(name=func.__name__, objective=obj, solution=sol, elapsed_sec=t1 - t0, meta=meta or {})
    wrapper.__name__ = func.__name__
    return wrapper

def euclid(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])

def path_length(points, tour):
    n = len(tour)
    return sum(euclid(points[tour[i]], points[tour[(i+1)%n]]) for i in range(n))

def seed_everything(seed: int = 42):
    random.seed(seed)
