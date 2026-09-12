"""Measure real CPU throughput for each detection method.

Reports candidates scored per second using the same settings as the main
experiments (default PyTorch threading, same batch sizes).
"""
import time, warnings, json, platform
warnings.filterwarnings("ignore")

from src.data_loader import load_task
from src import scorers

N_FAST = 300   # lexical / embedding methods
N_NLI = 200    # NLI is the bottleneck; smaller sample keeps this quick

_, test = load_task("qa")
fast = test.iloc[:N_FAST].reset_index(drop=True)
slow = test.iloc[:N_NLI].reset_index(drop=True)

results = {}


def timeit(label, fn, n):
    t0 = time.time()
    fn()
    dt = time.time() - t0
    results[label] = {"n": n, "seconds": round(dt, 2),
                      "per_second": round(n / dt, 1)}
    print(f"{label:<20} n={n:<5} {dt:8.2f}s   {n/dt:8.1f}/s", flush=True)


timeit("ROUGE-L",         lambda: scorers.rouge_l_scores(fast), N_FAST)
timeit("Sem. Similarity", lambda: scorers.similarity_scores(fast), N_FAST)
timeit("BERTScore",       lambda: scorers.bertscore_scores(fast), N_FAST)
timeit("NLI",             lambda: scorers.nli_scores(slow), N_NLI)

# Ensemble cost = similarity + NLI on the same inputs
t0 = time.time()
scorers.similarity_scores(slow)
scorers.nli_scores(slow)
dt = time.time() - t0
results["Ensemble"] = {"n": N_NLI, "seconds": round(dt, 2),
                       "per_second": round(N_NLI / dt, 1)}
print(f"{'Ensemble':<20} n={N_NLI:<5} {dt:8.2f}s   {N_NLI/dt:8.1f}/s", flush=True)

results["_env"] = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python": platform.python_version(),
}

with open("results/metrics/throughput.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved -> results/metrics/throughput.json")
