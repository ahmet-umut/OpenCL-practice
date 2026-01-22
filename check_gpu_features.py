import pyopencl as cl
import sys

# Force UTF-8 output for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Get all platforms and devices
platforms = cl.get_platforms()

for platform in platforms:
    print(f"\n{'='*70}")
    print(f"Platform: {platform.name}")
    print(f"Vendor: {platform.vendor}")
    print(f"Version: {platform.version}")
    print(f"{'='*70}")

    for device in platform.get_devices():
        print(f"\n  Device: {device.name}")
        print(f"  Type: {cl.device_type.to_string(device.type)}")
        print(f"  OpenCL Version: {device.version}")
        print(f"  Driver Version: {device.driver_version}")

        # Get OpenCL version as number
        version_str = device.version.split()[1]  # "OpenCL 2.0" -> "2.0"
        major, minor = map(int, version_str.split('.'))
        opencl_version = major * 10 + minor

        print(f"\n  --- Memory ---")
        print(f"  Global Memory: {device.global_mem_size / (1024**3):.2f} GB")
        print(f"  Local Memory: {device.local_mem_size / 1024:.2f} KB")
        print(f"  Max Allocation: {device.max_mem_alloc_size / (1024**3):.2f} GB")

        print(f"\n  --- Compute ---")
        print(f"  Max Compute Units: {device.max_compute_units}")
        print(f"  Max Work Group Size: {device.max_work_group_size}")
        print(f"  Max Work Item Dimensions: {device.max_work_item_dimensions}")
        print(f"  Max Work Item Sizes: {device.max_work_item_sizes}")

        print(f"\n  --- Advanced Features ---")

        # Device Enqueue (kernel spawning from kernels) - OpenCL 2.0+
        if opencl_version >= 20:
            try:
                max_on_device_queues = device.max_on_device_queues
                max_on_device_events = device.max_on_device_events
                queue_on_device_properties = device.queue_on_device_properties

                print(f"  ✓ Device Enqueue (Kernel spawning): SUPPORTED")
                print(f"    - Max on-device queues: {max_on_device_queues}")
                print(f"    - Max on-device events: {max_on_device_events}")
                print(f"    - Queue properties: {queue_on_device_properties}")
            except:
                print(f"  ✗ Device Enqueue (Kernel spawning): NOT SUPPORTED")
        else:
            print(f"  ✗ Device Enqueue: Requires OpenCL 2.0+ (you have {version_str})")

        # Pipes - OpenCL 2.0+
        if opencl_version >= 20:
            try:
                max_pipe_args = device.max_pipe_args
                print(f"  ✓ Pipes: SUPPORTED (max args: {max_pipe_args})")
            except:
                print(f"  ✗ Pipes: NOT SUPPORTED")

        # SVM (Shared Virtual Memory) - OpenCL 2.0+
        if opencl_version >= 20:
            try:
                svm_caps = device.svm_capabilities
                svm_supported = svm_caps != 0
                if svm_supported:
                    print(f"  ✓ SVM (Shared Virtual Memory): SUPPORTED")
                    print(f"    - Capabilities: {svm_caps}")
                else:
                    print(f"  ✗ SVM: NOT SUPPORTED")
            except:
                print(f"  ✗ SVM: NOT SUPPORTED")

        # Extensions
        print(f"\n  --- Extensions ---")
        extensions = device.extensions.split()

        interesting_extensions = {
            'cl_khr_fp16': 'Half precision (fp16)',
            'cl_khr_fp64': 'Double precision (fp64)',
            'cl_khr_int64_base_atomics': 'int64 atomic operations',
            'cl_khr_global_int32_base_atomics': 'Global int32 atomics',
            'cl_khr_local_int32_base_atomics': 'Local int32 atomics',
            'cl_khr_subgroups': 'Subgroups',
            'cl_intel_subgroups': 'Intel subgroups',
            'cl_amd_device_attribute_query': 'AMD device attributes',
            'cl_nv_device_attribute_query': 'NVIDIA device attributes',
        }

        for ext_name, ext_desc in interesting_extensions.items():
            if ext_name in extensions:
                print(f"  ✓ {ext_desc} ({ext_name})")

        # Show all extensions
        print(f"\n  All Extensions ({len(extensions)}):")
        for i, ext in enumerate(sorted(extensions)):
            if i % 2 == 0:
                print(f"    {ext}")
            else:
                print(f"    {ext}")

print("\n" + "="*70)
