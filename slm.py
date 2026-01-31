import pyopencl as opencl
import pyopencl.cltypes as clypes
import numpy
from math import *

coext = opencl.create_some_context(answers=[open("anser.txt").read()])

string = "Biz "
volary = "".join(list(set(string)))
mbedin = {char: volary.index(char) for char in string}

#zero-copy array
def zerocopa(shape, adiid=False, dtype=clypes.half):
    np_dtype = numpy.float16 if dtype == clypes.half else numpy.int32
    host_array = opencl.csvm_empty(coext, shape, np_dtype)  # no queue = uses internal in-order queue
    if adiid:
        host_array[:] = numpy.random.rand(*shape).astype(np_dtype)
    cl_buf = opencl.SVM(host_array)
    cl_buf.shape = shape
    cl_buf.dtype = dtype
    cl_buf.c_ray = host_array
    return cl_buf

opt_length=2
rank=2
vo_size=len(volary)

weights = zerocopa((rank,vo_size,2), 1)
output = zerocopa((opt_length+1,), 0, clypes.int)
error = zerocopa((1,))
debug = zerocopa((vo_size,2))

output.c_ray[0] = mbedin[string[0]]

# Create out-of-order queue after SVM allocations
queue = opencl.CommandQueue(coext, properties=opencl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)

# lower bound: 1.0003 - upper bound: 2.3414
alpha = 1.5
# Reduction macro. Reduces to reddt[0] in log(opspe steps.)
redue = lambda souce, value, fn: f"""
    if (locl1 < {e_of2 - vo_size})
        {souce}[{vo_size}+locl1] = {value};
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl1%2 == 0)
    {{
        reddt[locl0][locl1] = {fn}({souce}[locl1], {souce}[locl1+1]);

        //{( depth := ceil(log2(vo_size)) )}
        {"".join(f"""
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locl1%{2**(n+1)} == 0)
        {{
            reddt[locl0][locl1] = {fn}(reddt[locl0][locl1], reddt[locl0][locl1+{2**n}]);
            """ for n in range(1, depth)) }
        { "}"*(depth-1) }
    }}
    """

ir_init = f"""
    global half (*weights){"".join(f"[{dimen}]" for dimen in weights.shape[1:])} = (global half (*){"".join(f"[{dimen}]" for dimen in weights.shape[1:])})_weights;

    int locl0 = get_local_id(0);
    int locl1 = get_local_id(1);
    int locl = locl1*{max(rank,3)} + locl0;

    //{( e_of2 := 2**ceil(log2(vo_size)) )} //next power of 2 from vo_se
    local half preactivation[{rank}][{vo_size}];
    local half linop[{e_of2}], scors[{e_of2}];
    local half reddt[3][{vo_size//2+1}];
    local half tau,taumn,taumx, z[3][{e_of2}], Z; //for finding tau in alpha-entmax
    local int converged;
    """
irloop = lambda iptnex=None: f"""
    // linear layer:

    if (locl0 < {rank})
        preactivation[locl0][locl1] = weights[locl0][{"input" if iptnex is None else f"output[{iptnex}]"}][0] * weights[locl0][locl1][1];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl0)
        linop[locl1] = (scors[locl1] = {"+".join(f"preactivation[{i}][locl1]" for i in range(rank))}) * {alpha-1}h;  // pre-scaled sum of inputs for entmax

    // find tau for alpha entmax:

    //initialize
    if (!locl0)
    {{
        barrier(CLK_LOCAL_MEM_FENCE);
        {redue("linop", "-HUGE_VAL", "max")}
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl)
    {{
        taumn = reddt[0][0] - 1;
        taumx = reddt[0][0] - pow({vo_size}.h, {1-alpha}h);
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

kernel void infer (global int* _weights, global int* output)
{{
    constant half random[] = {{{", ".join(f"{numpy.random.rand():f}h" for _ in range(opt_length))}}};
    {ir_init}
    local int cf_rt[{e_of2//2}];
    {{{ "} barrier(CLK_LOCAL_MEM_FENCE); {".join(
        irloop(iptnex) + 
        f"""
        if (!locl0 && locl1 < {e_of2 - vo_size})
            z[0][{vo_size}+locl1] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (!locl0 && locl1%2 == 0)
        {{
            reddt[0][locl1] = z[0][locl1] + z[0][locl1+1];
            cf_rt[locl1] = random[{iptnex}] * reddt[0][locl1] < z[0][locl1] ? locl1 : locl1+1;

            //{( depth := ceil(log2(vo_size))-1 )}
            {"".join(f"""
            barrier(CLK_LOCAL_MEM_FENCE);
            if (locl1%{2**(n+1)} == 0)
            {{
                reddt[0][locl1] = reddt[0][locl1] + reddt[0][locl1+{2**n}];
                cf_rt[locl1] = random[{iptnex}] * reddt[0][locl1] > reddt[0][locl1+{2**n}] ? cf_rt[locl1] : cf_rt[locl1+{2**n}];
                """ for n in range(1, depth+1)) }
            { "}"*depth }
        }}
        if (!locl)
            output[{iptnex+1}] = cf_rt[0];
        """
    for iptnex in range(opt_length))}}}
}}
kernel void loss (global int* _weights, char input, char coect, global half* error)
{{
    //constant char input = {mbedin[string[0]]}, coect = {mbedin[string[1]]};
    {ir_init}
    {irloop()}
    barrier(CLK_LOCAL_MEM_FENCE);

    // alpha-entmax loss
    local half array[{vo_size}];
    if (!locl0)
    {{
        array[locl1] = (z[0][locl1] - (locl1==coect)) * scors[locl1] + (z[0][locl1] - pow(z[0][locl1], {alpha}h));
        barrier(CLK_LOCAL_MEM_FENCE);
        {redue("array", "0.0h", "add")}
        if (!locl)
            error[0] = reddt[0][0] / {alpha * (alpha - 1)}h;
    }}
}}
kernel void train (global int* _weights, char input, char coect, global half* _debug)
{{
    {ir_init}
    //constant char input = {mbedin[string[0]]}, coect = {mbedin[string[1]]};
    {irloop()}
    global half (*debug){"".join(f"[{dimen}]" for dimen in debug.shape[1:])} = (global half (*){"".join(f"[{dimen}]" for dimen in debug.shape[1:])})_debug;
    barrier(CLK_LOCAL_MEM_FENCE);

    local half gadien[{vo_size}];
    if (!locl0)
    {{
        //debug[locl1][0] = z[0][locl1] - (locl1==coect);
        debug[locl1][0] = z[0][locl1];
        debug[locl1][1] = weights[1][locl1][1];

        gadien[locl1] = z[0][locl1] - (locl1==coect);
        weights[1][locl1][1] -= gadien[locl1] * weights[1][input][0];
    }}
    /*
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!locl)
    {{
        debug[0][1] = weights[1][input][0];
        debug[1][1] = weights[1][2][1];

        //weights[1][2][1] -= gadien[2] * weights[1][input][0];

        debug[2][1] = weights[1][2][1];
        debug[3][1] = z[0][coect];
    }}
    */
}}
"""

print(kernel)

prg = opencl.Program(coext, kernel).build()
infer = prg.infer
loss = prg.loss
train = prg.train

#print(f"Weights \n{weights.host_array}")
print(f"volary: *{volary}*")

for _ in range(99):
    #run kernels
    infer(queue, (max(rank,3), vo_size), None, weights, output)
    loss(queue, (max(rank,3), vo_size), None, weights, clypes.char(mbedin[string[0]]), clypes.char(mbedin[string[1]]), error)
    train(queue, (max(rank,3), vo_size), None, weights, clypes.char(mbedin[string[0]]), clypes.char(mbedin[string[1]]), debug)
    queue.finish()
    ##
    decoded = ''.join(volary[output.c_ray[i]] for i in range(opt_length+1))
    #print(f"inference sequence {output.c_ray}")
    print(f"decoded output: {decoded}")
    print(f"Error: {error.c_ray}")
    print(f"train Debug: \n{debug.c_ray}")
    #print(f"Weights after training \n{weights.c_ray}")
    if error.c_ray[0] < 0 and decoded[1]!="i":
        #print(f"train Debug: \n{debug.c_ray}")
        pass
    if error.c_ray[0] < -1:
        break