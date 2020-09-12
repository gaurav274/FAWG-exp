import pandas as pd
import click
import json
import time
import subprocess
from srtml import modellib as mldev
from srtml.modellib import (
    ppu,
    ppu_type,
    predicate,
    SERVE_MODE,
)
from srtml.modellib.core import PGraph
from srtml.modellib import DATASET_INFORMATION, SERVE_MODE
from srtml_exp import (
    IMAGE_DATASET_INFORMATION,
    IMAGE_CLASSIFICATION_FEATURE,
    IMAGE_CLASSIFICATION_DIR,
    IMAGE_CLASSIFICATION_DIR_TWO_VERTEX,
    generate_fixed_arrival_process,
    get_latency_stats,
    shutdown,
)
from srtml.planner.optim import SimulatedAnnealing

import srtml
import ray
import os
import base64
from pprint import pprint
from srtml_exp.server import HTTPProxyActor
from pprint import pprint
import numpy as np


@ppu
class Node:
    def __init__(self) -> None:
        np.random.seed(0)
    @ppu_type()
    def __call__(self, a: int) -> int:
        return np.random.random()

@predicate
@ppu_type()
def predicate_fn(data: int) -> bool:
    if data < 0.5:
        return True
    return False

def create_pgraph(model_name, num_conditionals):

    graphs = {}
    with PGraph() as graphs[0]:
        # initialization
        source = mldev.placeholder(data=int)
        first_pred = predicate_fn()

        #connection
        source >> first_pred
        node = Node(_name=f"noop-{model_name}",_dummy_kwargs={"a":1})
        first_branch = mldev.cond(
            first_pred, true=node, default=1
        )
        source >> first_branch
    for i in range(1, num_conditionals):
        with PGraph() as graphs[i]:
            # initialization
            pred = predicate_fn()

            #connection
            graphs[i - 1] >> pred
            node = Node(_name=f"noop-{model_name}",_dummy_kwargs={"a":1})
            branch = mldev.cond(
                pred, true=node, default=1
            )
            graphs[i - 1] >> branch

    return graphs[num_conditionals - 1]



@click.command()
@click.option(
    "--xls-file", type=str, default="noop_conditionals.xlsx"
)
@click.option("--start-cmd", type=str, default=None)
@click.option("--end-cmd", type=str, default=None)
def main(xls_file, start_cmd, end_cmd):

    df = pd.read_excel(xls_file, sheet_name="Model Information")
    columns = [
        "mu (qps)",
        "cv",
        "# requests",
        "Latency Constraint (ms)",
        "Ingest mu Observed (qps)",
        "Throughput (qps)",
        "p95 (ms)",
        "p99 (ms)",
    ]
    raw_columns = [*columns[:3], "Latency (ms)"]
    new_df_clean = pd.DataFrame(columns=columns)
    new_df_raw = pd.DataFrame(columns=raw_columns)

    for index, row in df.iterrows():
        row_df = {columns[i]: df.loc[index, columns[i]] for i in range(3)}
        raw_row_df = dict(row_df)

        graph = create_pgraph(df.loc[index, "Model Name"], df.loc[index, "Number of Conditionals"])
        graph.configure(SERVE_MODE)

        if start_cmd:
            os.system(start_cmd)
        srtml.init(start_server=False)

        # pgraph_metadata = json.loads(df.loc[index, "configuration"])
        # graph.materialize(pgraph_metadata=pgraph_metadata)

        arrival_curve = generate_fixed_arrival_process(
            mean_qps=df.loc[index, columns[0]],
            cv=df.loc[index, columns[1]],
            num_requests=df.loc[index, columns[2]],
        ).tolist()

        graph.provision(SERVE_MODE)
        # img_path = os.path.join(IMAGE_CLASSIFICATION_DIR, "elephant.jpg")
        # data = base64.b64encode(open(img_path, "rb").read())

        # Warm-up and throughput calculation
        WARMUP = 200
        NUM_REQUESTS = 1000
        ppu = graph.handle
        futures = [ppu.remote(data=1) for _ in range(WARMUP)]
        ray.get(futures)
        start_time = time.time()
        futures = [ppu.remote(data=1) for _ in range(NUM_REQUESTS)]
        ray.wait(futures, num_returns=len(futures))
        end_time = time.time()
        time_taken = end_time - start_time
        throughput_qps = NUM_REQUESTS / time_taken

        row_df[columns[5]] = throughput_qps

        # latency calculation
        http_actor = HTTPProxyActor.remote(host="127.0.0.1", port=8000)
        ray.get(http_actor.register_route.remote("/noop", ppu))
        ray.get(http_actor.init_latency.remote())

        client_path = os.path.join(
            "noop_client.go",
        )

        row_df[columns[3]] = df.loc[index, columns[3]]

        ls_output = subprocess.Popen(
            [
                "go",
                "run",
                client_path,
                str(df.loc[index, columns[3]]),
                str(1),
                *[str(val) for val in arrival_curve],
            ]
        )
        ls_output.communicate()

        latency_list = ray.get(http_actor.get_latency.remote())
        ingest_mu, latency_ms, p95_ms, p99_ms = get_latency_stats(
            collected_latency=latency_list
        )
        row_df[columns[4]] = ingest_mu
        row_df[columns[6]] = p95_ms
        row_df[columns[7]] = p99_ms

        raw_row_df["Latency (ms)"] = latency_ms

        shutdown()
        if end_cmd:
            os.system(end_cmd)

        new_df_clean = new_df_clean.append(row_df, ignore_index=True)
        new_df_raw = new_df_raw.append(raw_row_df, ignore_index=True)

    with pd.ExcelWriter(xls_file, mode="a") as writer:
        new_df_clean.to_excel(writer, sheet_name="Results")
        new_df_raw.to_excel(
            writer, sheet_name="Results(Raw)"
        )


if __name__ == "__main__":
    main()
# print(raw.to_string())