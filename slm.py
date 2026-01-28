from numpy.random import f
import pyopencl as opencl
import pyopencl.cltypes as clypes
import numpy
from math import *

coext = opencl.create_some_context(answers=[open("anser.txt").read()])
queue = opencl.CommandQueue(coext)

string = "Biz "
volary = "".join(list(set(string)))
mbedin = {char: volary.index(char) for char in string}

#zero-copy array
def zerocopa(shape, adiid=False, dtype=clypes.half):
    np_dtype = numpy.float16 if dtype == clypes.half else numpy.int32
    host_array = opencl.csvm_empty(coext, shape, np_dtype, queue=queue)
    if adiid:
        host_array[:] = numpy.random.rand(*shape).astype(np_dtype)
    cl_buf = opencl.SVM(host_array)
    cl_buf.shape = shape
    cl_buf.dtype = dtype
    cl_buf.host_array = host_array
    return cl_buf

optlt=3
rank=2
vo_se=len(volary)

weights = zerocopa((rank,vo_se,2), 1)
output = zerocopa((optlt+1,), 0, clypes.int)
error = zerocopa((vo_se,))

output.host_array[0] = mbedin[string[0]]

class Agumen:
    def __init__(self, cl_buf):
        self.array = cl_buf
        self.type = "half" if cl_buf.dtype == clypes.half else "int"
        for name, value in globals().items():
            if value is self.array:
                self.name = name
                break
        def text(self, globa=0):
            return f"{"global" if globa else "constant"} {self.type}* {self.name}"
        self.text = text
        self.array.text = text.__get__(self)

    def __getattr__(self, name):
        return getattr(self.array, name)

Agumen(weights)
Agumen(output)
Agumen(error)

alpha = 1.1
irie = f"""
    int locl0 = get_local_id(0);
    int locl1 = get_local_id(1);
    int locl = locl1*{rank} + locl0;

    //{( cel_o := 2**ceil(log2(vo_se)) )} //next power of 2 from vo_se
    local half preactivation[{rank}][{vo_se}];
    local half linop[{cel_o}];
    local half reddt[{vo_se//2+1}];
    local half tau,taumn,taumx, z[{cel_o}]; //for finding tau in alpha-entmax 
    """
# Reduction macro. Reduces to reddt[0] in log(opspe steps.)
redue = lambda souce, auval, fn: f"""
    if (locl1 < {cel_o - vo_se})
        {souce}[{vo_se}+locl1] = {auval};
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl0 && locl1%2 == 0)
    {{
        reddt[locl1/2] = {fn}({souce}[locl1], {souce}[locl1+1]);

        //{( depth := ceil(log2(vo_se))-1 )}
        {"".join(f"""
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locl1%{2**(n+1)} == 0)
        {{
            reddt[locl1/{2**(n+1)}] = {fn}(reddt[locl1/{2**n}], reddt[locl1/{2**n}+1]);
            """ for n in range(1, depth+1)) }
        { "}"*depth }
    }}
    """
irlop = lambda ipnex=None: f"""
    preactivation[locl0][locl1] = weights[locl0][{"input" if ipnex is None else f"output[{ipnex}]"}][0] * weights[locl0][locl1][1];

    // linear output:

    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl0)
    {{
        half sum = 0.0f;
        for (int i = 0; i < {rank}; i++)
            sum += preactivation[i][locl1];
        linop[locl1] = sum;
    }}

    // entmax:

    // find tau (t)
    {redue("linop", "-HUGE_VAL", "max")}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl)
    {{
        taumn = reddt[0] - 1;
        taumx = reddt[0] - pow({vo_se}.h, {1-alpha}h);
    }}

    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl)
        tau = (taumn + taumx) / 2;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl0)
        z[locl1] = pow(max(0.h, linop[locl1] - tau), {1/(alpha - 1)}h);
    {redue("z", "0.0h", "add")}
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl0)
        linop[locl1] = pow(max(0.h, linop[locl1] - tau), {1/(alpha - 1)}h) / reddt[0];
    """

kernel = f"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define add(a,b) ((a)+(b))

// kernel void infer ({", ".join([ weights.text(), output.text(1) ])})
kernel void infer (constant half* _weights, global int* output)
{{
    constant half (*weights){"".join(f"[{dimen}]" for dimen in weights.shape[1:])} = (constant half (*){"".join(f"[{dimen}]" for dimen in weights.shape[1:])})_weights;
    {irie}
    local char cf_rt[{optlt}];
    {{{ "} barrier(CLK_LOCAL_MEM_FENCE); {".join(
        irlop(ipnex) + 
        f"""
        if (locl1 < {cel_o - vo_se})
            linop[{vo_se}+locl1] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locl0 && locl1%2 == 0)
        {{
            reddt[locl1/2] = linop[locl1] + linop[locl1+1];
            cf_rt[locl1/2] = {numpy.random.rand():f}h * reddt[locl1/2] < linop[locl1] ? locl1 : locl1+1;

            //{( depth := ceil(log2(vo_se))-1 )}
            {"".join(f"""
            barrier(CLK_LOCAL_MEM_FENCE);
            if (locl1%{2**(n+1)} == 0)
            {{
                reddt[locl1/{2**(n+1)}] = reddt[locl1/{2**n}] + reddt[locl1/{2**n}+1];
                cf_rt[locl1/{2**(n+1)}] = {numpy.random.rand():f}h * reddt[locl1/{2**(n+1)}] < reddt[locl1/{2**n}] ? cf_rt[locl1/{2**n}] : cf_rt[locl1/{2**n}+1];
                """ for n in range(1, depth+1)) }
            { "}"*depth }
        }}
        if (!locl)
            output[{ipnex+1}] = cf_rt[0];
        """
    for ipnex in range(optlt))}}}
}}
kernel void error (constant half* _weights, {error.text(1)}, int input, int coect)
{{
    constant half (*weights){"".join(f"[{dimen}]" for dimen in weights.shape[1:])} = (constant half (*){"".join(f"[{dimen}]" for dimen in weights.shape[1:])})_weights;
    {irie}
    {irlop()}
    if (locl0)
    {{
        error[locl1] = linop[locl1];
    }}
}}
"""

print(kernel)

# Build and run
prg = opencl.Program(coext, kernel).build()

#run kernels with zero-copy buffers
prg.infer(queue, (rank,vo_se), None, weights, output)
prg.error(queue, (rank,vo_se), None, weights, error, numpy.int32(mbedin[string[0]]), numpy.int32(mbedin[string[1]]))
queue.finish()

print(f"Weights \n{weights.host_array}")
print(f"volary: *{volary}*")
print(f"inference sequence {output.host_array}")
print(f"decoded: {''.join(volary[output.host_array[i]] for i in range(optlt+1))}")
print(f"Error {error.host_array}")
