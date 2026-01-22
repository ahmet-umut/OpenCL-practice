import pyopencl as opencl
import numpy as np
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

ctx = opencl.create_some_context(answers=[""])
queue = opencl.CommandQueue(ctx)

print("Testing if USE_HOST_PTR provides true zero-copy...\n")

# Test 1: Modify host array AFTER buffer creation and see if GPU sees it
print("=" * 70)
print("Test 1: CPU writes → GPU reads (without explicit copy)")
print("=" * 70)

host_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
cl_buf = opencl.Buffer(
    ctx,
    opencl.mem_flags.READ_WRITE | opencl.mem_flags.USE_HOST_PTR,
    hostbuf=host_array
)

# Modify host array AFTER buffer creation
print(f"Original host array: {host_array}")
host_array[:] = [10, 20, 30, 40, 50]
print(f"Modified host array: {host_array}")

# Create kernel to read and print values
kernel_code = """
kernel void test_read(global int* data, global int* output) {
    int i = get_global_id(0);
    output[i] = data[i];
}
"""

output_array = np.zeros(5, dtype=np.int32)
output_buf = opencl.Buffer(
    ctx,
    opencl.mem_flags.WRITE_ONLY | opencl.mem_flags.USE_HOST_PTR,
    hostbuf=output_array
)

prg = opencl.Program(ctx, kernel_code).build()
prg.test_read(queue, (5,), None, cl_buf, output_buf)
queue.finish()

# Map to read results
opencl.enqueue_map_buffer(queue, output_buf, opencl.map_flags.READ,
                         0, output_array.nbytes, output_array.dtype, is_blocking=True)

print(f"GPU read values:     {output_array}")

if np.array_equal(output_array, [10, 20, 30, 40, 50]):
    print("✓ SUCCESS: GPU saw modified values WITHOUT explicit copy!")
    print("  → This indicates TRUE ZERO-COPY")
else:
    print("✗ FAIL: GPU saw original values")
    print("  → Driver is copying data internally (not true zero-copy)")

# Test 2: GPU writes → CPU reads (without explicit copy)
print("\n" + "=" * 70)
print("Test 2: GPU writes → CPU reads (without explicit copy)")
print("=" * 70)

host_array2 = np.zeros(5, dtype=np.int32)
cl_buf2 = opencl.Buffer(
    ctx,
    opencl.mem_flags.READ_WRITE | opencl.mem_flags.USE_HOST_PTR,
    hostbuf=host_array2
)

kernel_code2 = """
kernel void test_write(global int* data) {
    int i = get_global_id(0);
    data[i] = i * 100;
}
"""

prg2 = opencl.Program(ctx, kernel_code2).build()
print(f"Host array before GPU write: {host_array2}")

prg2.test_write(queue, (5,), None, cl_buf2)
queue.finish()

# Try to read from host array WITHOUT explicit map/copy
# (Just ensure coherency with a finish)
print(f"Host array after GPU write:  {host_array2}")

if np.array_equal(host_array2, [0, 100, 200, 300, 400]):
    print("✓ SUCCESS: CPU saw GPU-written values WITHOUT explicit copy!")
    print("  → This indicates TRUE ZERO-COPY")
else:
    print("✗ FAIL: CPU didn't see GPU writes")
    print("  → Need explicit map/copy for coherency")

    # Try with explicit map
    opencl.enqueue_map_buffer(queue, cl_buf2, opencl.map_flags.READ,
                             0, host_array2.nbytes, host_array2.dtype, is_blocking=True)
    print(f"After explicit map:          {host_array2}")

    if np.array_equal(host_array2, [0, 100, 200, 300, 400]):
        print("  → Works with explicit map (coarse-grained SVM behavior)")

# Test 3: Memory address comparison
print("\n" + "=" * 70)
print("Test 3: Memory address analysis")
print("=" * 70)

host_array3 = np.array([1, 2, 3], dtype=np.int32)
print(f"Host array memory address: {host_array3.__array_interface__['data'][0]:x}")

# Create buffer with USE_HOST_PTR
cl_buf3 = opencl.Buffer(
    ctx,
    opencl.mem_flags.READ_WRITE | opencl.mem_flags.USE_HOST_PTR,
    hostbuf=host_array3
)

# Create buffer WITHOUT USE_HOST_PTR for comparison
host_array4 = np.array([1, 2, 3], dtype=np.int32)
cl_buf4 = opencl.Buffer(
    ctx,
    opencl.mem_flags.READ_WRITE | opencl.mem_flags.COPY_HOST_PTR,
    hostbuf=host_array4
)

print(f"Buffer with USE_HOST_PTR created")
print(f"Buffer with COPY_HOST_PTR created (control)")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
