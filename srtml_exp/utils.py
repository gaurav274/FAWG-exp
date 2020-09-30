import git
import os
import json
import jsonlines
from sklearn.isotonic import IsotonicRegression
from srtml.modellib.constants import THROUGHPUT_KEY, LATENCY_KEY
import srtml
import ray
import pandas as pd
from srtml_exp.sysinfo import get_sysinfo
import numpy as np
from srtml.planner.optim import (
    SimulatedAnnealing,
    VanillaInferline,
    ImprovedInferline,
    DummyPlanner,
)

ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
IMAGE_CLASSIFICATION_DIR = os.path.join(ROOT_DIR, "image_preprocessing")
IMAGE_CLASSIFICATION_DIR_TWO_VERTEX = os.path.join(
    IMAGE_CLASSIFICATION_DIR, "two_vertex"
)

IMAGE_CLASSIFICATION_FEATURE = "ImageClassification"
IMAGE_DATASET_INFORMATION = {"dataset": "Imagenet", "dataset_category": "IMAGE"}
PLANNER_CLS = {
    "SimulatedAnnealing": SimulatedAnnealing,
    "VanillaInferline": VanillaInferline,
    "ImprovedInferline": ImprovedInferline,
    "DummyPlanner": DummyPlanner,
}
import numpy as np


def gamma(mean, cv, size):
    if cv == 0.0:
        return np.ones(size) * mean
    else:
        return np.random.gamma(1.0 / cv, cv * mean, size=size)


def generate_fixed_arrival_process(mean_qps, cv, num_requests):
    """
    mean_qps : float
        Mean qps
    cv : float
    duration: float
        Duration of the trace in seconds
    """
    # deltas_path = os.path.join(arrival_process_dir,
    #                            "fixed_{mean_qps}_{cv}_{dur}_{ts:%y%m%d_%H%M%S}.deltas".format(
    #                                mean_qps=mean_qps, cv=cv, dur=duration, ts=datetime.now()))
    inter_request_delay_ms = 1.0 / float(mean_qps) * 1000.0
    num_deltas = num_requests - 1
    if cv == 0:
        deltas = np.ones(num_deltas) * inter_request_delay_ms
    else:
        deltas = gamma(inter_request_delay_ms, cv, size=num_deltas)
    deltas = np.clip(deltas, a_min=2.5, a_max=None)
    return deltas


class BytesEncoder(json.JSONEncoder):
    """Allow bytes to be part of the JSON document.
    BytesEncoder will walk the JSON tree and decode bytes with utf-8 codec.
    (Adopted from serve 0.8.2)
    Example:
    >>> json.dumps({b'a': b'c'}, cls=BytesEncoder)
    '{"a":"c"}'
    """

    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, bytes):
            return o.decode("utf-8")
        return super().default(o)


def get_latency(filename):
    latency = list()
    with jsonlines.open(filename) as reader:
        for obj in reader:
            latency.append((obj["end"] - obj["start"]))
    return latency


def convert_profiles_to_regression_models(profile_dict):
    learned_profile = dict()
    for ppu in profile_dict:
        learned_profile[ppu] = dict()
        for hardware in profile_dict[ppu]:
            learned_profile[ppu][hardware] = dict()
            learned_profile[ppu][hardware][
                THROUGHPUT_KEY
            ] = IsotonicRegression().fit(
                *list(zip(*profile_dict[ppu][hardware][THROUGHPUT_KEY]))
            )
            learned_profile[ppu][hardware][
                LATENCY_KEY
            ] = IsotonicRegression().fit(
                *list(zip(*profile_dict[ppu][hardware][LATENCY_KEY]))
            )
    return learned_profile


def get_dataframe_from_profile(pgraph_identifier, profile_dict):
    columns = [
        "pgraph",
        "ppu",
        "hardware",
        "batch size",
        "latency (ms)",
        "throughput (qps)",
    ]
    df_dict = dict()
    row_index = 0
    for ppu in profile_dict:
        for hardware in profile_dict[ppu]:
            for latency_item, throughput_item in zip(
                profile_dict[ppu][hardware][LATENCY_KEY],
                profile_dict[ppu][hardware][THROUGHPUT_KEY],
            ):
                assert latency_item[0] == throughput_item[0], "Wrong Profile"
                batch_size = latency_item[0]
                latency_ms = latency_item[1]
                throughput_qps = throughput_item[1]
                df_row = [
                    pgraph_identifier,
                    ppu,
                    hardware,
                    batch_size,
                    latency_ms,
                    throughput_qps,
                ]
                df_dict[row_index] = df_row
                row_index += 1

    idx = pd.MultiIndex.from_product(
        [[get_sysinfo()], columns],
        names=["system information", "model repository"],
    )
    return pd.DataFrame.from_dict(df_dict, orient="index", columns=idx)


def _get_ingest_observed_throughput(start_time_list):
    start_time_list.sort()
    avg_time_diff = 0
    cnt = 0
    for i in range(len(start_time_list) - 1):
        avg_time_diff += start_time_list[i + 1] - start_time_list[i]
        cnt += 1
    avg_time_diff = avg_time_diff / cnt
    return 1.0 / avg_time_diff


def get_latency_stats(collected_latency):

    latency_list_ms = [
        (d["end"] - d["start"]) * 1000 for d in collected_latency
    ]
    p95_ms, p99_ms = np.percentile(latency_list_ms, [95, 99])

    ingest_throughput = _get_ingest_observed_throughput(
        [d["start"] for d in collected_latency]
    )
    return ingest_throughput, latency_list_ms, p95_ms, p99_ms


def shutdown():
    ray.shutdown()
    srtml.serve.api.global_state = None
    srtml.modellib.api.global_state = None


def set_seed(graph_handle):
    ppu_handles = graph_handle.ppu_handles
    for ppu_name in ppu_handles:
        phandle = ppu_handles[ppu_name]
        all_replicas = phandle.get_replica_handles()
        for replica_handle in all_replicas:
            ray.get(replica_handle.seed.remote())