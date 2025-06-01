import numpy as np
import torch
import tensorflow as tf
import time
import pandas as pd
from scipy.stats import t

def run_benchmark():
    tf.config.set_visible_devices([], 'GPU')
    torch_device = torch.device("cpu")

    sizes = [2**i for i in range(1, 15)]
    num_runs = 20

    results = []

    for n in sizes:
        print(f"Benchmarking n={n}...")

        # Speicher für Zeiten
        numpy_times = []
        torch_times = []
        tf_times = []

        for _ in range(num_runs):
            # NumPy
            A = np.random.rand(n, n)
            B = np.random.rand(n, n)
            start = time.perf_counter()
            np.dot(A, B)
            numpy_times.append(time.perf_counter() - start)

            # PyTorch
            A = torch.rand((n, n), device=torch_device)
            B = torch.rand((n, n), device=torch_device)
            start = time.perf_counter()
            torch.matmul(A, B)
            torch_times.append(time.perf_counter() - start)

            # TensorFlow
            A = tf.random.uniform((n, n))
            B = tf.random.uniform((n, n))
            start = time.perf_counter()
            tf.linalg.matmul(A, B)
            tf_times.append(time.perf_counter() - start)

        # Konfidenzintervall 95 %
        def conf_interval(data):
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            ci = t.ppf(0.975, len(data)-1) * std / np.sqrt(len(data))
            return mean, std, ci

        numpy_mean, numpy_std, numpy_ci = conf_interval(numpy_times)
        torch_mean, torch_std, torch_ci = conf_interval(torch_times)
        tf_mean, tf_std, tf_ci = conf_interval(tf_times)

        flops = 2 * (n ** 3)
        results.append({
            "n": n,
            "numpy_mean": numpy_mean,
            "numpy_std": numpy_std,
            "numpy_ci": numpy_ci,
            "numpy_gflops": flops / (numpy_mean * 1e9),
            "torch_mean": torch_mean,
            "torch_std": torch_std,
            "torch_ci": torch_ci,
            "torch_gflops": flops / (torch_mean * 1e9),
            "tf_mean": tf_mean,
            "tf_std": tf_std,
            "tf_ci": tf_ci,
            "tf_gflops": flops / (tf_mean * 1e9)
        })

    # Speichern als CSV
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("Benchmark abgeschlossen: benchmark_results.csv erstellt.")