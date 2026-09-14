import time

import torch

torch.manual_seed(0)
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")


def bench(tag, n=5):
    c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(tag, f"{dt*1000:.1f} ms", f"{2*4096**3/dt/1e12:.1f} TFLOPS")


bench("fp32 default")
torch.backends.cuda.matmul.allow_tf32 = True
bench("tf32")
torch.set_float32_matmul_precision("high")
bench("high")
a16, b16 = a.half(), b.half()


def bench16(tag, n=10):
    c = a16 @ b16
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        c = a16 @ b16
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(tag, f"{dt*1000:.1f} ms", f"{2*4096**3/dt/1e12:.1f} TFLOPS")


bench16("fp16")
print("torch", torch.__version__, "cuda", torch.version.cuda)
