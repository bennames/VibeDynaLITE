import os
import numba
import numpy as np
import time

print("Numba Version:", numba.__version__)
print("NUMBA_THREADING_LAYER Env:", os.environ.get("NUMBA_THREADING_LAYER"))
print("OMP_NUM_THREADS Env:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS Env:", os.environ.get("MKL_NUM_THREADS"))
print("Default threads:", numba.config.NUMBA_DEFAULT_NUM_THREADS)

try:
    print("Active threads:", numba.get_num_threads())
except Exception as e:
    print("get_num_threads failed:", e)

# Let's run a simple prange test to see if it triggers multi-threading
@numba.njit(parallel=True)
def test_parallel(x):
    s = 0.0
    for i in numba.prange(len(x)):
        s += np.sin(x[i])
    return s

x = np.random.rand(50_000_000)
print("Compiling test_parallel...")
test_parallel(x[:10])

print("Running test_parallel on 50M elements...")
t0 = time.perf_counter()
res = test_parallel(x)
t1 = time.perf_counter()
print(f"Finished in {t1-t0:.4f} seconds.")

try:
    print("Threading layer used:", numba.threading_layer())
except Exception as e:
    print("threading_layer() failed:", e)
