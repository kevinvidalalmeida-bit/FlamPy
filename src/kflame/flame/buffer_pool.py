"""Reusable float buffers for one solve, with a hard memory bound.

Large NumPy arrays are mapped fresh from the operating system and every page
is zero-filled on first touch (on Windows about 0.8 s per GB). Repeated
allocations of the same shape -- the Jacobians and pivot inverses of one
grid, factorization chunks, per-thread chemistry tiles -- are therefore
served from buffers kept for the duration of a solve.

Memory is never traded for speed: buffers have the exact requested size, a
buffer is lent again only when no view of it is alive (so concurrent users
never share memory), idle large buffers are released before a new one is
allocated, and the idle buffers retained never exceed ``MAX_FREE_BYTES``
(least recently used released first). Contents are never assumed: callers
overwrite everything they read.
"""
import sys
import threading

import numpy as np

# Idle buffers of at least this size are released before any new allocation.
LARGE_BYTES = 64*1024**2
# Upper bound of the memory held by idle buffers.
MAX_FREE_BYTES = 512*1024**2


def _idle(buffer):
    # An idle buffer is referred to only by the pool list, the caller's loop
    # variable, this parameter and getrefcount's own argument; every live
    # view of it holds one more reference (its .base).
    return sys.getrefcount(buffer) <= 4


class Pool:
    def __init__(self):
        self._buffers = []  # least recently used first
        self._lock = threading.Lock()

    def take(self, shape, dtype=np.float64):
        size = int(np.prod(shape))
        dtype = np.dtype(dtype)
        with self._lock:
            # A comprehension keeps no loop variable alive in this frame.
            best = next((b for b in self._buffers if b.size == size and b.dtype == dtype and _idle(b)), None)
            if best is None:
                # Release idle large buffers before allocating, so a previous
                # grid's storage never coexists with the new one.
                self._buffers = [b for b in self._buffers if not (b.nbytes >= LARGE_BYTES and _idle(b))]
                best = np.empty(size, dtype=dtype)
            self._buffers = [b for b in self._buffers if b is not best]+[best]
            # Bound the idle memory, releasing the least recently used first.
            idle = [b for b in self._buffers if b is not best and _idle(b)]
            held = sum(b.nbytes for b in idle)
            released = []
            for candidate in idle:
                if held <= MAX_FREE_BYTES:
                    break
                held -= candidate.nbytes
                released.append(id(candidate))
            self._buffers = [b for b in self._buffers if id(b) not in released]
            del idle, released
            return best.reshape(shape)

    def stats(self):
        """(buffers, total bytes, idle bytes) currently held."""
        with self._lock:
            idle = sum(b.nbytes for b in self._buffers if _idle(b))
            return len(self._buffers), sum(b.nbytes for b in self._buffers), idle

    def release(self):
        with self._lock:
            self._buffers = []


def pool_of(problem):
    """The problem's pool, created on first use."""
    pool = getattr(problem, '_buffer_pool', None)
    if pool is None:
        pool = problem._buffer_pool = Pool()
    return pool
