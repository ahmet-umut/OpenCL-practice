import pyopencl as opencl
import pyopencl.cltypes as clypes
import numpy
from math import *

ctx = opencl.create_some_context(answers=[""])
queue = opencl.CommandQueue(ctx)

string = "Biz "
volary = list(set(string))
mbedin = {char: volary.index(char) for char in string}

#coarse SVM zero-copy buffer
def rm_ar_zerocopy(shape, random=False, dtype=clypes.half):
    np_dtype = numpy.float16 if dtype == clypes.half else numpy.int32
    host_array = opencl.csvm_empty(ctx, shape, np_dtype, queue=queue)
    if random:
        host_array[:] = numpy.random.rand(*shape).astype(np_dtype)
    cl_buf = opencl.SVM(host_array)
    cl_buf.shape = shape
    cl_buf.dtype = dtype
    cl_buf.host_array = host_array
    return cl_buf

n=9
rank=2
opspe=len(volary)
weights = rm_ar_zerocopy((rank,opspe,2), 1)
output = rm_ar_zerocopy((1,), 0, clypes.int)

class Agumen:
    def __init__(self, cl_buf, globa=False):
        self.array = cl_buf
        self.pecer = "global" if globa else "constant"
        self.type = "half" if cl_buf.dtype == clypes.half else "int"
        for name, value in globals().items():
            if value is self.array:
                self.name = name
                break
        def access(self, indice_text):
            indices = indice_text.split(',')
            brackets_text = indices[-1]
            ine_o = len(indices)
            for i in range(ine_o-1):
                brackets_text += f"+{indices[ine_o-i-2]}"
                for j in range(i+1):
                    brackets_text += f"*{self.array.shape[j]}"
            return f"{self.name}[{brackets_text}]"
        self.access = access
        self.array.access = access.__get__(self)
    def __getattr__(self, name):
        return getattr(self.array, name)


iputs = Agumen(weights), Agumen(output, True)

kernel = f"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void program({', '.join(f"{iput.pecer} {iput.type}* {iput.name}" for iput in iputs)}, int input)
{{
    int locl0 = get_local_id(0);
    int locl1 = get_local_id(1);
    int locl = locl1*{rank} + locl0;

    local half preactivation[{rank}][{opspe}];
    preactivation[locl0][locl1] = {weights.access("locl0,input,0")} * {weights.access("locl0,locl1,1")};
    local half activations[{(activationsz := 2**ceil(log2(opspe)))}];

    // activation
    if (locl0)
    {{
        barrier(CLK_LOCAL_MEM_FENCE);
        half sum = 0.0f;
        for (int i = 0; i < {rank}; i++)
            sum += preactivation[i][locl1];
        activations[locl1] = sum / (1+fabs(sum));
    }}
    else if (locl1 < {activationsz - opspe})
        activations[{opspe}+locl1] = -2;

    // argmax
    local uchar indis[{opspe//2+1}];
    if (locl1%2 == 0 && locl0)
    {{
        barrier(CLK_LOCAL_MEM_FENCE);
        if (activations[locl1] < activations[locl1+1])
            indis[locl1/2] = locl1+1;
        else
            indis[locl1/2] = locl1;

        //{( depth := ceil(log2(opspe))-1 )}
        {"".join(f"""
        if (locl1%{2**(n+1)} == 0 && locl0)
        {{
            barrier(CLK_LOCAL_MEM_FENCE);
            if (activations[indis[locl1/{2**n}]] < activations[indis[locl1/{2**n}+1]])
                indis[locl1/{2**(n+1)}] = indis[locl1/{2**n}+1];
            else
                indis[locl1/{2**(n+1)}] = indis[locl1/{2**n}];
            """ for n in range(1, depth+1))}
            output[0] = indis[0];
        {"}"*depth}
    }}
}}
"""

print(kernel)

# Build and run
import struct
prg = opencl.Program(ctx, kernel).build()

#run kernel with zero-copy buffers
prg.program(queue, (rank,opspe), None, weights, output, struct.pack('i', mbedin[string[0]]))
queue.finish()

print(f"Input {string}")
print(f"Weights \n{weights.host_array}")
print(f"Result {output.host_array[0]}")

#print("".join(volary[token] for token in rsult))
