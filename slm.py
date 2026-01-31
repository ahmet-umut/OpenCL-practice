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
    cl_buf.host_array = host_array
    return cl_buf

optlt=2
rank=2
vo_se=len(volary)

weights = zerocopa((rank,vo_se,2), 1)
output = zerocopa((optlt+1,), 0, clypes.int)
error = zerocopa((1,))
debug = zerocopa((vo_se,2))

output.host_array[0] = mbedin[string[0]]

# Create out-of-order queue after SVM allocations
queue = opencl.CommandQueue(coext, properties=opencl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)

# lower bound: 1.0003 - upper bound: 2.3414
alpha = 1.5
# Reduction macro. Reduces to reddt[0] in log(opspe steps.)
redue = lambda souce, auval, fn: f"""
    if (locl1 < {cel_o - vo_se})
        {souce}[{vo_se}+locl1] = {auval};
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locl1%2 == 0)
    {{
        reddt[locl0][locl1] = {fn}({souce}[locl1], {souce}[locl1+1]);

        //{( depth := ceil(log2(vo_se)) )}
        {"".join(f"""
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locl1%{2**(n+1)} == 0)
        {{
            reddt[locl0][locl1] = {fn}(reddt[locl0][locl1], reddt[locl0][locl1+{2**n}]);
            """ for n in range(1, depth)) }
        { "}"*(depth-1) }
    }}
    """

irie = f"""
    global half (*weights){"".join(f"[{dimen}]" for dimen in weights.shape[1:])} = (global half (*){"".join(f"[{dimen}]" for dimen in weights.shape[1:])})_weights;

    int locl0 = get_local_id(0);
    int locl1 = get_local_id(1);
    int locl = locl1*{max(rank,3)} + locl0;

    //{( cel_o := 2**ceil(log2(vo_se)) )} //next power of 2 from vo_se
    local half preactivation[{rank}][{vo_se}];
    local half linop[{cel_o}], scors[{cel_o}];
    local half reddt[3][{vo_se//2+1}];
    local half tau,taumn,taumx, z[3][{cel_o}], Z; //for finding tau in alpha-entmax
    local int converged;
    """
irlop = lambda ipnex=None: f"""
    // linear layer:

    if (locl0 < {rank})
        preactivation[locl0][locl1] = weights[locl0][{"input" if ipnex is None else f"output[{ipnex}]"}][0] * weights[locl0][locl1][1];

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

kernel void infer (global int* _weights, global int* output)
{{
    constant half random[] = {{{", ".join(f"{numpy.random.rand():f}h" for _ in range(optlt))}}};
    {irie}
    local int cf_rt[{cel_o//2}];
    {{{ "} barrier(CLK_LOCAL_MEM_FENCE); {".join(
        irlop(ipnex) + 
        f"""
        if (!locl0 && locl1 < {cel_o - vo_se})
            z[0][{vo_se}+locl1] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (!locl0 && locl1%2 == 0)
        {{
            reddt[0][locl1] = z[0][locl1] + z[0][locl1+1];
            cf_rt[locl1] = random[{ipnex}] * reddt[0][locl1] < z[0][locl1] ? locl1 : locl1+1;
            if ({ipnex}==0)
                printf("randomized: %f, element: %f, picked %d\\n", random[{ipnex}] * reddt[0][locl1], z[0][locl1], cf_rt[locl1]);

            //{( depth := ceil(log2(vo_se))-1 )}
            {"".join(f"""
            barrier(CLK_LOCAL_MEM_FENCE);
            if (locl1%{2**(n+1)} == 0)
            {{
                reddt[0][locl1] = reddt[0][locl1] + reddt[0][locl1+{2**n}];
                cf_rt[locl1] = random[{ipnex}] * reddt[0][locl1] > reddt[0][locl1+{2**n}] ? cf_rt[locl1] : cf_rt[locl1+{2**n}];
                if ({ipnex}==0)
                    printf("randomized: %f, element: %f, picked %d\\n", random[{ipnex}] * reddt[0][locl1], reddt[0][locl1+{2**n}], cf_rt[locl1]);
                """ for n in range(1, depth+1)) }
            { "}"*depth }
        }}
        if (!locl)
            output[{ipnex+1}] = cf_rt[0];
        """
    for ipnex in range(optlt))}}}
}}
kernel void loss (global int* _weights, global half* error)
{{
    constant char input = {mbedin[string[0]]}, coect = {mbedin[string[1]]};
    {irie}
    {irlop()}
    barrier(CLK_LOCAL_MEM_FENCE);

    // alpha-entmax loss
    local half array[{vo_se}];
    if (!locl0)
    {{
        array[locl1] = (z[0][locl1] - (locl1==coect)) * scors[locl1] + (z[0][locl1] - pow(z[0][locl1], {alpha}h));
        barrier(CLK_LOCAL_MEM_FENCE);
        {redue("array", "0.0h", "add")}
        if (!locl)
            error[0] = reddt[0][0] / {alpha * (alpha - 1)}h;
    }}
}}
kernel void train (global int* _weights, global half* _debug)
{{
    {irie}
    constant char input = {mbedin[string[0]]}, coect = {mbedin[string[1]]};
    {irlop()}
    global half (*debug){"".join(f"[{dimen}]" for dimen in debug.shape[1:])} = (global half (*){"".join(f"[{dimen}]" for dimen in debug.shape[1:])})_debug;
    barrier(CLK_LOCAL_MEM_FENCE);

    local half gadien[{vo_se}];
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
    infer(queue, (max(rank,3), vo_se), None, weights, output)
    loss(queue, (max(rank,3), vo_se), None, weights, error)
    train(queue, (max(rank,3), vo_se), None, weights, debug)
    queue.finish()
    ##
    decoded = ''.join(volary[output.host_array[i]] for i in range(optlt+1))
    #print(f"inference sequence {output.host_array}")
    print(f"decoded output: *{decoded}*")
    print(f"Error: {error.host_array}")
    print(f"train Debug: \n{debug.host_array}")
    #print(f"Weights after training \n{weights.host_array}")
    if error.host_array[0] < 0 and decoded[1]!="i":
        #print(f"train Debug: \n{debug.host_array}")
        pass
    if error.host_array[0] < -1:
        break
