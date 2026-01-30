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

# lower bound: 1.0003 - upper bound: 2.3414
alpha = 1.5
irie = f"""
    int locl0 = get_local_id(0);
    int locl1 = get_local_id(1);
    int locl = locl1*{max(rank,3)} + locl0;

    //{( cel_o := 2**ceil(log2(vo_se)) )} //next power of 2 from vo_se
    local half preactivation[{rank}][{vo_se}];
    local half linop[{cel_o}];
    local half reddt[3][{vo_se//2+1}];
    local half tau,taumn,taumx, z[3][{cel_o}], Z; //for finding tau in alpha-entmax
    local int converged;
    """
# Reduction macro. Reduces to reddt[0] in log(opspe steps.)
redue = lambda souce, auval, fn: f"""
    if (locl1 < {cel_o - vo_se})
        {souce}[{vo_se}+locl1] = {auval};
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl1%2 == 0)
    {{
        reddt[locl0][locl1/2] = {fn}({souce}[locl1], {souce}[locl1+1]);

        //{( depth := ceil(log2(vo_se))-1 )}
        {"".join(f"""
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locl1%{2**(n+1)} == 0)
        {{
            reddt[locl0][locl1/{2**(n+1)}] = {fn}(reddt[locl0][locl1/{2**n}], reddt[locl0][locl1/{2**n}+1]);
            """ for n in range(1, depth+1)) }
        { "}"*depth }
    }}
    """
irlop = lambda ipnex=None: f"""
    if (locl0 < {rank})
        preactivation[locl0][locl1] = weights[locl0][{"input" if ipnex is None else f"output[{ipnex}]"}][0] * weights[locl0][locl1][1];

    // linear output:

    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl0)
    {{
        half sum = 0.0f;
        for (int i = 0; i < {rank}; i++)
            sum += preactivation[i][locl1];
        linop[locl1] = sum * {alpha-1}h;  // pre-scale for entmax
    }}

    // find tau for alpha entmax:

    //initialize
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl0)
    {{
        {redue("linop", "-HUGE_VAL", "max")}
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl)
    {{
        taumn = reddt[0][0] - 1;
        taumx = reddt[0][0] - pow({vo_se}.h, {1-alpha}h);
        tau = (taumn + taumx) / 2;
        converged = 0;
    }}

    for (int iter = 0; iter < 1; iter++)
    {{
        //compute f, f', f''
        barrier(CLK_LOCAL_MEM_FENCE);
        z[locl0][locl1] = pow(max(0.h, linop[locl1] - tau), {1/(alpha - 1)}h - locl0);
        barrier(CLK_LOCAL_MEM_FENCE);
        {redue("z[locl0]", "0.0h", "add")}

        barrier(CLK_LOCAL_MEM_FENCE);
        if (!locl)
        {{
            half f = reddt[0][0] - 1;
            if (f != 0)
            {{
                if (f < 0)
                    taumx = tau;  // sum < 1 means tau too high
                else
                    taumn = tau;  // sum > 1 means tau too low

                half ff = reddt[1][0] / {1-alpha}h;
                half fff = reddt[2][0] / {(2-alpha) / (alpha-1)**2}h;

                half h = tau - 2*f*ff / (2*ff*ff - f*fff);

                if (h < taumn || h > taumx)
                    tau = (taumn+taumx)/2;
                else
                    tau = h;
            }}
            else
                converged = 1;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        if (converged) break;
    }}
    """

kernel = f"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define add(a,b) ((a)+(b))

constant half weights{"".join(f"[{dimen}]" for dimen in weights.shape)} = {{ {", ".join(f"{i:f}h" for i in weights.host_array.flatten())} }};

kernel void infer (global int* output)
{{
    constant half random[] = {{{", ".join(f"{numpy.random.rand():f}h" for _ in range(optlt))}}};
    //constant half (*weights){"".join(f"[{dimen}]" for dimen in weights.shape[1:])} = (constant half (*){"".join(f"[{dimen}]" for dimen in weights.shape[1:])})_weights;
    {irie}
    local char cf_rt[{optlt}];
    {{{ "} barrier(CLK_LOCAL_MEM_FENCE); {".join(
        irlop(ipnex) + 
        f"""
        if (!locl0 && locl1 < {cel_o - vo_se})
            z[0][{vo_se}+locl1] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (!locl0 && locl1%2 == 0)
        {{
            reddt[0][locl1/2] = z[0][locl1] + z[0][locl1+1];
            cf_rt[locl1/2] = random[{ipnex}] * reddt[0][locl1/2] < z[0][locl1] ? locl1 : locl1+1;

            //{( depth := ceil(log2(vo_se))-1 )}
            {"".join(f"""
            barrier(CLK_LOCAL_MEM_FENCE);
            if (locl1%{2**(n+1)} == 0)
            {{
                reddt[0][locl1/{2**(n+1)}] = reddt[0][locl1/{2**n}] + reddt[0][locl1/{2**n}+1];
                cf_rt[locl1/{2**(n+1)}] = random[{ipnex}] * reddt[0][locl1/{2**(n+1)}] < reddt[0][locl1/{2**n}] ? cf_rt[locl1/{2**n}] : cf_rt[locl1/{2**n}+1];
                """ for n in range(1, depth+1)) }
            { "}"*depth }
        }}
        if (!locl)
            output[{ipnex+1}] = cf_rt[0];
        """
    for ipnex in range(optlt))}}}
}}
kernel void error (global half* error)
{{
    constant char input = {mbedin[string[0]]}, coect = {mbedin[string[1]]};
    {irie}
    {irlop()}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl0 == 0)
    {{
        /*
        half relu = max(0.h, linop[coect] - tau);
        if (relu > 0.h)
            error[0] = -log(relu) / {alpha - 1}h;   // cross-entropy loss
        */
        error[locl1] = z[0][locl1] - (locl1==coect);
    }}
}}
"""

print(kernel)

# Build and run
prg = opencl.Program(coext, kernel).build()

#run kernels with zero-copy buffers
prg.infer(queue, (max(rank,3), vo_se), None, output)
prg.error(queue, (max(rank,3), vo_se), None, error)
queue.finish()

print(f"Weights \n{weights.host_array}")
print(f"volary: *{volary}*")
print(f"inference sequence {output.host_array}")
print(f"decoded: {''.join(volary[output.host_array[i]] for i in range(optlt+1))}")
print(f"Error {error.host_array}")
