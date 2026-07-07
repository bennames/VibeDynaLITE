import numpy as np
import taichi as ti

try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)

n = 100
a = ti.Vector.field(3, dtype=ti.f32)
a_block = ti.root.dense(ti.i, n)
for i in range(3):
    a_block.place(a.get_scalar_field(i, 0))

b = ti.Vector.field(3, dtype=ti.f32)
b_block = ti.root.dense(ti.i, n)
for i in range(3):
    b_block.place(b.get_scalar_field(i, 0))

indices = ti.field(dtype=ti.i32, shape=(10,))

@ti.kernel
def test_kernel():
    ti.block_local(a)
    ti.block_local(b)
    for i in range(10):
        idx = indices[i]
        b[idx] += a[idx]

a.from_numpy(np.ones((n, 3), dtype=np.float32))
indices.from_numpy(np.arange(10, dtype=np.int32) * 5)
test_kernel()
print("Success! b[0] =", b[0])
