import torch
import time


def generate_tensors(num_tensors, batch_size, seq_len, hidden_size):
    tensors_a = []
    tensors_b = []

    for i in range(num_tensors):
        tensor_a = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
        tensor_b = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
        if i < 5:
            tensor_a = tensor_a.to("cuda")
            tensor_b = tensor_b.to("cuda")
        tensors_a.append(tensor_a)
        tensors_b.append(tensor_b)
    return tensors_a, tensors_b


def generate_tensors2(num_tensors, batch_size, seq_len, hidden_size):
    tensors_a = []
    tensors_b = []

    for i in range(num_tensors):
        tensor_a = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
        tensor_b = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
        # if i < num_tensors / 2 - 6:
        #     tensor_a = tensor_a.to("cuda")
        #     tensor_b = tensor_b.to("cuda")
        tensors_a.append(tensor_a)
        tensors_b.append(tensor_b)
    return tensors_a, tensors_b


def complex_computation(a, b):
    scores = torch.matmul(a, b.transpose(-1, -2))

    scores = torch.nn.functional.softmax(scores, dim=-1, dtype=a.dtype)

    value = torch.randn_like(a, dtype=a.dtype)

    context = torch.matmul(scores, value)

    linear_weight = torch.randn(a.size(-1), a.size(-1), device=a.device, dtype=a.dtype)
    output = torch.relu(torch.matmul(context, linear_weight))

    return output.sum()


def method1(tensors_a, tensors_b):
    num_tensors = len(tensors_a)

    # 初始化缓冲区张量
    buffer_a = (
        tensors_a[0].to("cuda", non_blocking=True) if tensors_a[0].device != torch.device("cuda") else tensors_a[0]
    )
    buffer_b = (
        tensors_b[0].to("cuda", non_blocking=True) if tensors_b[0].device != torch.device("cuda") else tensors_b[0]
    )

    torch.cuda.synchronize()
    start_time = time.time()

    for i in range(num_tensors):
        if i + 1 < num_tensors:
            transfer_stream = torch.cuda.Stream()
            with torch.cuda.stream(transfer_stream):
                next_a = tensors_a[i + 1]
                next_b = tensors_b[i + 1]
                if next_a.device != torch.device("cuda"):
                    next_a = next_a.to("cuda")
                if next_b.device != torch.device("cuda"):
                    next_b = next_b.to("cuda")

        c = complex_computation(buffer_a, buffer_b)

        torch.cuda.current_stream().synchronize()

        if i + 1 < num_tensors:
            # transfer_stream.synchronize()
            buffer_a = next_a
            buffer_b = next_b

    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def method2(tensors_a, tensors_b, interval=6):
    num_tensors = len(tensors_a)
    buffer_a = []
    buffer_b = []

    for i in range(min(interval, num_tensors)):
        tensor_a = tensors_a[i]
        tensor_b = tensors_b[i]
        if tensor_a.device != torch.device("cuda"):
            tensor_a = tensor_a.to("cuda")
        if tensor_b.device != torch.device("cuda"):
            tensor_b = tensor_b.to("cuda")
        buffer_a.append(tensor_a)
        buffer_b.append(tensor_b)

    torch.cuda.synchronize()
    start_time = time.time()

    compute_idx = 0
    load_idx = interval

    while compute_idx < num_tensors:
        if compute_idx % interval == 0 and load_idx < num_tensors:
            transfer_stream = torch.cuda.Stream()
            with torch.cuda.stream(transfer_stream):
                for _ in range(interval):
                    if load_idx < num_tensors:
                        next_a = tensors_a[load_idx]
                        next_b = tensors_b[load_idx]
                        if next_a.device != torch.device("cuda"):
                            next_a = next_a.to("cuda")
                        if next_b.device != torch.device("cuda"):
                            next_b = next_b.to("cuda")
                        buffer_a.append(next_a)
                        buffer_b.append(next_b)
                        load_idx += 1

        current_a = buffer_a.pop(0)
        current_b = buffer_b.pop(0)
        c = complex_computation(current_a, current_b)

        torch.cuda.current_stream().synchronize()

        compute_idx += 1

    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def main():
    num_runs = 20
    num_tensors = 80
    batch_size = 2
    seq_len = 512
    hidden_size = 1024

    tensors_a, tensors_b = generate_tensors(num_tensors, batch_size, seq_len, hidden_size)
    tensors_a2, tensors_b2 = generate_tensors2(num_tensors, batch_size, seq_len, hidden_size)

    times_method1 = []
    times_method2 = []

    for i in range(num_runs):
        t = method1(tensors_a, tensors_b)
        times_method1.append(t)

    avg_time_method1 = sum(times_method1) / num_runs
    print(f"\n{num_runs} method1: {avg_time_method1:.6f} s")

    for i in range(num_runs):
        t = method2(tensors_a2, tensors_b2)
        times_method2.append(t)

    avg_time_method2 = sum(times_method2) / num_runs
    print(f"\n{num_runs} method2: {avg_time_method2:.6f} s")

    time_diff = avg_time_method2 - avg_time_method1
    print(f"\nmethod2 - method1: {time_diff:.6f} s")
    time_diff = (avg_time_method1 - avg_time_method2) / avg_time_method1
    print(f"\n(method2 - method1)/method1: {time_diff*100:.1f}%")


if __name__ == "__main__":
    main()
