import logging
import os
import time
from math import ceil
from typing import Optional

import numpy as np
import torch
import transformers

LOGGER_FORMAT = "[%(levelname)s][%(name)s]: %(message)s"
logging.basicConfig(format=LOGGER_FORMAT)
log = logging.getLogger(__file__)

MODEL_NAME = "bert-base-uncased"
QUANTIZATION = True
BATCH_SIZE = 1
SEQ_LENGTH = 128


def get_data(model_):
    vocab_size = model_.config.vocab_size
    data = torch.randint(vocab_size, size=[BATCH_SIZE, SEQ_LENGTH])
    return data


def run_benchmark(
    model: transformers.BertModel,
    data: torch.Tensor,
    benchmark_time_seconds: Optional[float] = None,
    warmup_time_seconds: Optional[float] = None,
    benchmark_iterations: Optional[int] = None,
    warmup_iterations: Optional[int] = None,
):
    """
    We can use seconds rather than number of runs because larger models (and larger batch sizes) typically exhibit
    smaller performance variance between runs, so we need fewer iterations for larger models to get a reasonable
    average throughput. Running the benchmark for a specified time rather than number of iterations accommodates that.

    If preferable, it possible to specify the number of iterations instead.
    """

    def timed_run(model: transformers.BertModel, data: torch.Tensor):
        with torch.inference_mode():
            start = time.perf_counter()
            model(data)
            end = time.perf_counter()

        return end - start

    def run_for_seconds(
        model: transformers.BertModel, data: torch.Tensor, run_time_seconds: float
    ):
        elapsed_time_seconds = 0.0
        num_runs = 0
        while elapsed_time_seconds < run_time_seconds:
            log.debug(
                f"Running iteration {num_runs}, elapsed time {elapsed_time_seconds}"
            )
            elapsed_time_seconds += timed_run(model, data)
            num_runs += 1
        return elapsed_time_seconds, num_runs

    def run_for_iterations(
        model: transformers.BertModel, data: torch.Tensor, iterations: int
    ):
        elapsed_time_seconds = 0.0
        for _ in range(iterations):
            log.debug(f"Running iteration {_}, elapsed time {elapsed_time_seconds}")
            elapsed_time_seconds += timed_run(model, data)

        return elapsed_time_seconds, iterations

    if benchmark_time_seconds is not None and benchmark_iterations is not None:
        raise ValueError(
            f"Run time and number of iterations cannot be specified at the same time."
        )

    if benchmark_time_seconds is None and benchmark_iterations is None:
        raise ValueError(f"Either run time or number of iterations must be specified.")

    batch_size = data.shape[0]

    if benchmark_time_seconds is not None:
        # Warmup runs
        warmup_time_seconds = (
            0.1 * benchmark_time_seconds
            if not warmup_time_seconds
            else warmup_time_seconds
        )

        log.info(f"Running warmup cycles for {warmup_time_seconds} seconds.")
        run_for_seconds(model, data, warmup_time_seconds)

        # Benchmark runs
        log.info(f"Running benchmark cycles for {benchmark_time_seconds} seconds.")
        total_time_seconds, num_runs = run_for_seconds(
            model, data, benchmark_time_seconds
        )

    else:
        # Warmup runs
        warmup_iterations = (
            ceil(0.1 * benchmark_iterations)
            if not warmup_iterations
            else warmup_iterations
        )

        log.info(f"Running warmup cycles for {warmup_iterations} iterations.")
        run_for_iterations(model, data, warmup_iterations)

        # Benchmark runs
        log.info(f"Running benchmark cycles for {benchmark_iterations} iterations.")
        total_time_seconds, num_runs = run_for_iterations(
            model, data, benchmark_iterations
        )

    average_time = total_time_seconds / num_runs
    average_throughput = batch_size / average_time

    return average_throughput, average_time


# from transformers.generation import GenerationConfig
print("..................bert_op............")
from bentomlx.intel.custom_ops import bert_op  # noqa

print("..............................")

config = transformers.BertConfig.from_pretrained(
    MODEL_NAME,
    use_quantization=QUANTIZATION,
    quantization_factors=np.tile(np.tile([-10.0, 10.0], 4), 12),
)
config.use_quantization = QUANTIZATION
# config.use_bfloat16 = args.bfloat16
if config.use_quantization:
    config.quantization_factors = np.tile(
        np.tile([-10.0, 10.0], 4),  # min/max values for one layer
        config.num_hidden_layers,
    )  # repeat for number of layers


model = transformers.BertModel.from_pretrained(MODEL_NAME, config=config)
model.eval()

data = get_data(model)
with torch.inference_mode():
    model = torch.jit.trace(model, data, strict=False)
    model = torch.jit.freeze(model)
    # Run inference twice to initialize the optimizations
    model(data)
    model(data)

model.config = config  # we restore the config after tracing so
# that get_data() can read the vocab size and
# max seq len from the model


if __name__ == "__main__":

    # torch.set_num_interop_threads(20)
    print(f"torch num_threads: {torch.get_num_threads()}")
    print(f"torch num_interop_threads: {torch.get_num_interop_threads()}")

    import pandas as pd

    run_time = 20
    warmup_time = 5
    iterations = None
    warmup_iterations = None
    ipex = False
    bert_op = True
    bfloat16 = False

    data = get_data(model)
    results = pd.DataFrame(
        columns=[
            "Model",
            "IPEX",
            "BERT Op",
            "Quantization",
            "BFloat16",
            "Batch Size",
            "Seq Len",
            "Throughput [samples/s]",
            "Latency [ms]",
        ]
    )
    # throughput, latency = run_benchmark(model, data, run_time, warmup_time, iterations, warmup_iterations)

    import torch.utils.benchmark as benchmark

    t00 = benchmark.Timer(
        stmt="""
           with torch.inference_mode():
               model(data)
           """,
        # setup='from __main__ import batched_dot_bmm',
        globals={"model": model, "data": data},
        num_threads=2,
    )

    t0 = benchmark.Timer(
        stmt="""
        with torch.inference_mode():
            model(data)
        """,
        # setup='from __main__ import batched_dot_bmm',
        globals={"model": model, "data": data},
        num_threads=4,
    )
    t1 = benchmark.Timer(
        stmt="""
        with torch.inference_mode():
            model(data)
        """,
        # setup='from __main__ import batched_dot_bmm',
        globals={"model": model, "data": data},
        num_threads=8,
    )
    t2 = benchmark.Timer(
        stmt="""
        with torch.inference_mode():
            model(data)
        """,
        # setup='from __main__ import batched_dot_bmm',
        globals={"model": model, "data": data},
        num_threads=16,
    )

    m00 = t00.timeit(100)
    m0 = t0.timeit(100)
    m1 = t1.timeit(100)
    m2 = t2.timeit(100)

    print(m00.mean * 1000.0, m00.median * 1000.0)
    print(m0.mean * 1000.0, m0.median * 1000.0)
    print(m1.mean * 1000.0, m1.median * 1000.0)
    print(m2.mean * 1000.0, m2.median * 1000.0)

    # row = pd.Series({
    #     'Model': MODEL_NAME,
    #     'IPEX': str(ipex),
    #     'BERT Op': str(bert_op),
    #     'Quantization': str(QUANTIZATION),
    #     'BFloat16': str(bfloat16),
    #     'Batch Size': BATCH_SIZE,
    #     'Seq Len': SEQ_LENGTH,
    #     'Throughput [samples/s]': f'{throughput:.3f}',
    #     'Latency [ms]': f'{latency * 1000 :.3f} ms'
    # })
    # results = pd.concat([results, row.to_frame().T], ignore_index=True)

    # print(results.to_markdown())
