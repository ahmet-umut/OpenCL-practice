import pyopencl as opencl
import pyopencl.cltypes as clypes
import numpy
from math import *

coext = opencl.create_some_context(answers=[""])
queue = opencl.CommandQueue(coext)

string = "Biz "
volary = "".join(list(set(string)))
mbedin = {char: volary.index(char) for char in string}

#zero-copy array
def rcopa(shape, adiid=False, dtype=clypes.half):
    np_dtype = numpy.float16 if dtype == clypes.half else numpy.int32
    host_array = opencl.csvm_empty(coext, shape, np_dtype, queue=queue)
    if adiid:
        host_array[:] = numpy.random.rand(*shape).astype(np_dtype)
    cl_buf = opencl.SVM(host_array)
    cl_buf.shape = shape
    cl_buf.dtype = dtype
    cl_buf.host_array = host_array
    return cl_buf

oplt=3
rank=2
opspe=len(volary)

weights = rcopa((rank,opspe,2), 1)
output = rcopa((oplt+1,), 0, clypes.int)
error = rcopa((opspe,))

output.host_array[0] = mbedin[string[0]]

class Agumen:
    def __init__(self, cl_buf):
        self.array = cl_buf
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

    local uchar indis[{opspe//2+1}];
    local half noled[{(cel_o := 2**ceil(log2(opspe)))}];
    local half activations[{cel_o}];

    local half preactivation[{rank}][{opspe}];
    local half linop[{cel_o}];
    //local half preactivation[{opspe}];

    local half maxer[{opspe//2+1}];
    local half sumer[{opspe//2+1}];

    local half reddt[{opspe//2+1}];
    local half t,tmin,tmax;

    """
redue = lambda souce, auval, fn: f"""
    if (locl1 < {cel_o - opspe})
        {souce}[{opspe}+locl1] = {auval};
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl0 && locl1%2 == 0)
    {{
        reddt[locl1/2] = {fn}({souce}[locl1], {souce}[locl1+1]);

        //{( depth := ceil(log2(opspe))-1 )}
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
    preactivation[locl0][locl1] = {weights.access(f"locl0, {"input" if ipnex is None else f"output[{ipnex}]"} ,0")} * {weights.access("locl0,locl1,1")};

    // linear output
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl0)
    {{
        half sum = 0.0f;
        for (int i = 0; i < {rank}; i++)
            sum += preactivation[i][locl1];
        linop[locl1] = sum;
    }}

    // entmax

    {redue("linop", "-HUGE_VAL", "max")}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl)
    {{
        tmin = reddt[0] - 1;
        tmax = reddt[0] - pow({opspe}.h, {1-alpha}h);
    }}

    {"""
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl)
        if (reddt[0] > 1)   tmin = t;
        else                tmax = t;
    """.join(f"""
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl)
        t = (tmin + tmax) / 2;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl0)
        activations[locl1] = pow(max(0.h, linop[locl1] - t), {1/(alpha - 1)}h);
    {redue("activations", "0.0h", "add")}
    """ for _ in range(1))}

    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl0)
        linop[locl1] = pow(max(0.h, linop[locl1] - t), {1/(alpha - 1)}h) / reddt[0];

    // argmax
    /*
    {"" if ipnex is None else f"""
    else if (locl1 < {cel_o - opspe})
        activations[{opspe}+locl1] = -2;

    if (locl1%2 == 0 && locl0)
    {{
        barrier(CLK_LOCAL_MEM_FENCE);
        if (activations[locl1] < activations[locl1+1])
            indis[locl1/2] = locl1+1;
        else
            indis[locl1/2] = locl1;

        //{( depth := ceil(log2(opspe))-1 )}
        {"".join(f"""
        if (locl1%{2**(n+1)} == 0)
        {{
            barrier(CLK_LOCAL_MEM_FENCE);
            if (activations[indis[locl1/{2**n}]] < activations[indis[locl1/{2**n}+1]])
                indis[locl1/{2**(n+1)}] = indis[locl1/{2**n}+1];
            else
                indis[locl1/{2**(n+1)}] = indis[locl1/{2**n}];
            """ for n in range(1, depth+1))}
            output[{ipnex+1}] = indis[0];
        {"}"*depth}
    }}
    """}
    */
    """

kernel = f"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define add(a,b) ((a)+(b))
kernel void infer ({", ".join([ weights.text(), output.text(1) ])})
{{
    {irie}
    {{{"} barrier(CLK_LOCAL_MEM_FENCE); {".join(irlop(ipnex) for ipnex in range(oplt))}}}
}}
kernel void error ({", ".join([ weights.text(), error.text(1), "int input, int coect" ])})
{{
    {irie}
    {irlop()}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl0)
    {{
        error[locl1] = linop[locl1];
    }}
}}
"""

print(kernel)

# Build and run
# import struct
prg = opencl.Program(coext, kernel).build()

#run kernel with zero-copy buffers
prg.infer(queue, (rank,opspe), None, weights, output)
prg.error(queue, (rank,opspe), None, weights, error, numpy.int32(mbedin[string[0]]), numpy.int32(mbedin[string[1]]))
queue.finish()

print(f"Weights \n{weights.host_array}")
print(f"volary: *{volary}*")
print(f"inference sequence {output.host_array}")
print(f"decoded: {''.join(volary[output.host_array[i]] for i in range(oplt+1))}")
print(f"Error {error.host_array}")
