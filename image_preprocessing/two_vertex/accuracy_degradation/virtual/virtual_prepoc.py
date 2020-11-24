import pandas as pd
import click
import json
import time
import subprocess
from srtml import VGraph, vpu, vpu_type
from srtml.modellib import DATASET_INFORMATION, SERVE_MODE, LOCAL_MODE
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


@vpu
@vpu_type(output_type=list)
class Classifier:
    text: list


dinfo = DATASET_INFORMATION(**IMAGE_DATASET_INFORMATION)


@click.command()
@click.option(
    "--xls-file", type=str, default="virtual_image_classification.xlsx"
)
@click.option("--start-cmd", type=str, default=None)
@click.option("--end-cmd", type=str, default=None)
def main(xls_file, start_cmd, end_cmd):

    df = pd.read_excel(xls_file, sheet_name="Planner Configuration")
    columns = [
        "mu (qps)",
        "cv",
        "# requests",
        "Latency Constraint (ms)",
        "Planner",
        "Estimated Latency (ms)",
        "Cost",
        "Accuracy",
        "Ingest mu Observed (qps)",
        "Throughput (qps)",
        "p95 (ms)",
        "p99 (ms)",
    ]
    raw_columns = [*columns[:8], "Latency (ms)"]
    new_df_clean = pd.DataFrame(columns=columns)
    new_df_raw = pd.DataFrame(columns=raw_columns)

    for index, row in df.iterrows():
        row_df = {columns[i]: df.loc[index, columns[i]] for i in range(8)}
        raw_row_df = dict(row_df)

        if start_cmd:
            os.system(start_cmd)
        ray_serve_kwargs={
        "ray_init_kwargs": {
            "object_store_memory": int(5e10),
            "num_cpus": 24,
            "_internal_config": json.dumps(
                {
                    "max_direct_call_object_size": 10 * 1024 * 1024,  # 10Mb
                    "max_grpc_message_size": 100 * 1024 * 1024,  # 100Mb
                }
            ),
            # "resources": resources,
        },
        "start_server": False,
        }

        srtml.init(ray_serve_kwargs = ray_serve_kwargs)

        with VGraph(name="bert") as graph:
            classifier = Classifier(
                feature=IMAGE_CLASSIFICATION_FEATURE,
                dataset_information=dinfo,
                name="VClassifier",
            )

        pgraph_metadata = json.loads(df.loc[index, "configuration"])
        # pgraph_metadata['bert/VClassifier']['ppu_state']['Sentimental-bert24-2/bert24_p2_stage0']['resources']['Tesla P40'] = 0.2
        # pgraph_metadata['bert/VClassifier']['ppu_state']['Sentimental-bert24-2/bert24_p2_stage0']['num_gpus'] = 0.2
        graph.materialize(pgraph_metadata=pgraph_metadata)

        arrival_curve = generate_fixed_arrival_process(
            mean_qps=df.loc[index, columns[0]],
            cv=df.loc[index, columns[1]],
            num_requests=df.loc[index, columns[2]],
        ).tolist()
        graph.provision(SERVE_MODE)
        # img_path = os.path.join(IMAGE_CLASSIFICATION_DIR, "elephant.jpg")
        # data = base64.b64encode(open(img_path, "rb").read())
        data = 'I am hot and sexy'
        # Warm-up and throughput calculation
        WARMUP = 2
        NUM_REQUESTS = 10
        vpu = graph.handle
        futures = [vpu.remote(text=data) for _ in range(WARMUP)]
        ray.get(futures)
        start_time = time.time()
        futures = [vpu.remote(text=data) for _ in range(NUM_REQUESTS)]
        ray.wait(futures, num_returns=len(futures))
        end_time = time.time()
        time_taken = end_time - start_time
        throughput_qps = NUM_REQUESTS / time_taken

        row_df[columns[9]] = throughput_qps

        print('#######################################')
        # latency calculation
        http_actor = HTTPProxyActor.remote(host="127.0.0.1", port=8001)
        ray.get(http_actor.register_route.remote("/bert", vpu))
        ray.get(http_actor.init_latency.remote())

        client_path = os.path.join(
            IMAGE_CLASSIFICATION_DIR_TWO_VERTEX,
            "accuracy_degradation",
            "virtual",
            "image_prepoc_client.go",
        )

        ls_output = subprocess.Popen(
            [
                "go",
                "run",
                client_path,
                str(df.loc[index, columns[3]]),
                'lol',
                *[str(val) for val in arrival_curve],
            ]
        )
        ls_output.communicate()

        latency_list = ray.get(http_actor.get_latency.remote())
        ingest_mu, latency_ms, p95_ms, p99_ms = get_latency_stats(
            collected_latency=latency_list
        )
        row_df[columns[8]] = ingest_mu
        row_df[columns[10]] = p95_ms
        row_df[columns[11]] = p99_ms

        raw_row_df["Latency (ms)"] = latency_ms

        elapsed_latency_list = [latency_dict["end"] - latency_dict["start"] for latency_dict in latency_list]
        elapsed_latency_list.sort()
        f = open("/nethome/gkakkar7/srtml-experiments/experimental_results/qps_latency_tradeoff/125q_200l_p2/latencies.txt", "w")
        for entry in elapsed_latency_list:
            f.write(str(entry))
            f.write("\n")
        f.close()

        shutdown()
        if end_cmd:
            os.system(end_cmd)

        new_df_clean = new_df_clean.append(row_df, ignore_index=True)
        new_df_raw = new_df_raw.append(raw_row_df, ignore_index=True)

    with pd.ExcelWriter(xls_file, mode="a") as writer:
        new_df_clean.to_excel(writer, sheet_name="Planner Config Run Results")
        new_df_raw.to_excel(
            writer, sheet_name="Planner Config Run Results(Raw)"
        )


if __name__ == "__main__":
    main()
# print(raw.to_string())
