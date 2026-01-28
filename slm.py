import pyopencl as cl
import pyopencl.cltypes as cltypes
import numpy as np
from math import ceil, log2

cl_context = cl.create_some_context(answers=["1"])
cl_queue = cl.CommandQueue(cl_context)

seed_text = "Biz "
vocab = "".join(dict.fromkeys(seed_text))
char_to_id = {c: vocab.index(c) for c in seed_text}

def alloc_svm(shape, init_random=False, dtype=cltypes.half):
    np_dtype = np.float16 if dtype == cltypes.half else np.int32
    host_array = cl.csvm_empty(cl_context, shape, np_dtype, queue=cl_queue)
    if init_random:
        host_array[:] = np.random.rand(*shape).astype(np_dtype)
    buf = cl.SVM(host_array)
    buf.shape = shape
    buf.dtype = dtype
    buf.host_array = host_array
    return buf

sequence_length = 3
hidden_dim = 2
vocab_size = len(vocab)

weights = alloc_svm((hidden_dim, vocab_size, 2), init_random=True)
token_ids = alloc_svm((sequence_length + 1,), dtype=cltypes.int)
logits = alloc_svm((vocab_size,))

token_ids.host_array[0] = char_to_id[seed_text[0]]

class KernelArg:
    def __init__(self, buf):
        self.buf = buf
        self.cl_type = "half" if buf.dtype == cltypes.half else "int"
        for k, v in globals().items():
            if v is buf:
                self.var_name = k
                break
        def declaration(self, global_mem=False):
            space = "global" if global_mem else "constant"
            return f"{space} {self.cl_type}* {self.var_name}"
        self.declaration = declaration
        self.buf.declaration = declaration.__get__(self)

    def __getattr__(self, name):
        return getattr(self.buf, name)

KernelArg(weights)
KernelArg(token_ids)
KernelArg(logits)

alpha = 1.1
pad_size = 2 ** ceil(log2(vocab_size))

kernel_locals = f"""
int local_x = get_local_id(0);
int local_y = get_local_id(1);
int local_id = local_y * {hidden_dim} + local_x;

local half preactivation[{hidden_dim}][{vocab_size}];
local half linear_output[{pad_size}];
local half reduction_buffer[{vocab_size // 2 + 1}];
local half activations[{pad_size}];
local half tau, tau_min, tau_max;
"""

local_reduce = lambda source, padval, op: f"""
if (local_y < {pad_size - vocab_size})
    {source}[{vocab_size} + local_y] = {padval};
barrier(CLK_LOCAL_MEM_FENCE);
if (local_x && local_y % 2 == 0)
{{
    reduction_buffer[local_y / 2] = {op}({source}[local_y], {source}[local_y + 1]);
    {''.join(f'''
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_y % {2 ** (n + 1)} == 0)
        reduction_buffer[local_y / {2 ** (n + 1)}] =
            {op}(reduction_buffer[local_y / {2 ** n}],
                 reduction_buffer[local_y / {2 ** n} + 1]);
    ''' for n in range(1, ceil(log2(vocab_size))))}
}}
"""

forward_step = lambda step=None: f"""
preactivation[local_x][local_y] =
    weights[local_x][{"token_ids[" + str(step) + "]" if step is not None else "input"}][0] *
    weights[local_x][local_y][1];

barrier(CLK_LOCAL_MEM_FENCE);
if (local_x)
{{
    half sum = 0.0h;
    for (int i = 0; i < {hidden_dim}; i++)
        sum += preactivation[i][local_y];
    linear_output[local_y] = sum;
}}

{local_reduce("linear_output", "-HUGE_VAL", "max")}
barrier(CLK_LOCAL_MEM_FENCE);
if (!local_id)
{{
    tau_min = reduction_buffer[0] - 1;
    tau_max = reduction_buffer[0] - pow({vocab_size}.h, {1 - alpha}h);
    tau = (tau_min + tau_max) / 2;
}}
barrier(CLK_LOCAL_MEM_FENCE);

if (local_x)
    activations[local_y] =
        pow(max(0.h, linear_output[local_y] - tau), {1 / (alpha - 1)}h);

{local_reduce("activations", "0.h", "add")}
barrier(CLK_LOCAL_MEM_FENCE);

if (local_x)
    linear_output[local_y] =
        pow(max(0.h, linear_output[local_y] - tau), {1 / (alpha - 1)}h) /
        reduction_buffer[0];
"""

kernel_source = f"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define add(a,b) ((a)+(b))

kernel void sample_sequence({weights.declaration()}, {token_ids.declaration(True)})
{{
    constant half (*weights)[{vocab_size}][2] =
        (constant half (*)[{vocab_size}][2])weights;
    {kernel_locals}
    barrier(CLK_LOCAL_MEM_FENCE);
    {''.join(
        forward_step(i) + f'''
        barrier(CLK_LOCAL_MEM_FENCE);
        if (!local_id)
        {{
            half sample_threshold = {np.random.rand():.6f}h;
            for (int j = 0; j < {vocab_size}; j++)
            {{
                sample_threshold -= linear_output[j];
                if (sample_threshold <= 0.h)
                {{
                    token_ids[{i + 1}] = j;
                    break;
                }}
            }}
        }}
        ''' for i in range(sequence_length)
    )}
}}

kernel void compute_logits({weights.declaration()}, {logits.declaration(True)}, int input, int target)
{{
    constant half (*weights)[{vocab_size}][2] =
        (constant half (*)[{vocab_size}][2])weights;
    {kernel_locals}
    {forward_step()}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_x)
        logits[local_y] = linear_output[local_y];
}}
"""

program = cl.Program(cl_context, kernel_source).build()

program.sample_sequence(
    cl_queue,
    (hidden_dim, vocab_size),
    None,
    weights,
    token_ids
)

program.compute_logits(
    cl_queue,
    (hidden_dim, vocab_size),
    None,
    weights,
    logits,
    np.int32(char_to_id[seed_text[0]]),
    np.int32(char_to_id[seed_text[1]])
)

cl_queue.finish()

print(weights.host_array)
print(vocab)
print(token_ids.host_array)
print("".join(vocab[token_ids.host_array[i]] for i in range(sequence_length + 1)))
print(logits.host_array)
