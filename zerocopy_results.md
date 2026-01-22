# Zero-Copy Test Results for Intel Iris Xe iGPU

## Summary
**`USE_HOST_PTR` does NOT provide true zero-copy on your Intel Iris Xe GPU.**

## Test Results

### Test 1: CPU writes → GPU reads
- **Result**: ✗ FAILED
- CPU modified array from `[1,2,3,4,5]` → `[10,20,30,40,50]`
- GPU still read original values `[1,2,3,4,5]`
- **Conclusion**: Driver copied data at buffer creation time

### Test 2: GPU writes → CPU reads
- **Result**: ✗ FAILED (without map), ✓ SUCCESS (with map)
- GPU wrote `[0,100,200,300,400]`
- CPU read `[0,0,0,0,0]` without explicit map
- CPU read `[0,100,200,300,400]` after explicit `enqueue_map_buffer`
- **Conclusion**: Need explicit map for coherency (coarse-grained SVM behavior)

## What This Means

1. **Driver behavior**: Intel's OpenCL driver creates an INTERNAL copy of the data when using `USE_HOST_PTR`
   - It doesn't use the host pointer directly
   - Data is copied to a separate GPU-accessible region

2. **Why?**: Even though it's an iGPU sharing physical RAM:
   - Driver may use different memory regions for better performance
   - Virtual address spaces are still separate
   - Cache coherency management

3. **Actual memory flow**:
   ```
   Buffer creation with USE_HOST_PTR:
   Host array → [COPY] → GPU memory region

   Kernel execution:
   GPU reads/writes its copy

   Reading results:
   GPU memory → [MAP/COPY] → Host array
   ```

4. **Performance impact**:
   - Still faster than discrete GPUs (same physical RAM, no PCIe)
   - But NOT true zero-copy
   - Each kernel invocation requires coherency operations

## Comparison with True Zero-Copy (SVM)

Your GPU reports SVM capability = 1 (coarse-grained buffer SVM), which would provide:
- ✓ Unified virtual addresses
- ✓ Same memory location accessed by CPU and GPU
- ✗ Still requires explicit synchronization (map/unmap or barriers)

But PyOpenCL's SVM API is not well-exposed, making it impractical to use.

## Bottom Line

Your current implementation with `USE_HOST_PTR`:
- ✓ Simpler API than explicit `enqueue_copy`
- ✓ Better than discrete GPU (same RAM chip)
- ✗ NOT true zero-copy
- ✗ Driver still copies data internally
- ✗ Requires explicit `enqueue_map_buffer` for CPU to see GPU writes
