import pandas as pd
import click
import json
import time
import subprocess
from srtml import VGraph, vpu, vpu_type
from srtml.modellib import DATASET_INFORMATION, SERVE_MODE
from srtml_exp import (
    IMAGE_DATASET_INFORMATION,
    IMAGE_CLASSIFICATION_FEATURE,
    IMAGE_CLASSIFICATION_DIR,
    generate_fixed_arrival_process,
    get_latency_stats,
    PLANNER_CLS,
    shutdown,
)


import srtml
import ray
import os
import base64
from pprint import pprint
from srtml_exp.server import HTTPProxyActor
from pprint import pprint
import click


@vpu
@vpu_type(output_type=list)
class Classifier:
    image: list


dinfo = DATASET_INFORMATION(**IMAGE_DATASET_INFORMATION)


@click.command()
@click.option(
    "--xls-file", type=str, default="virtual_image_classification.xlsx"
)
@click.option("--start-cmd", type=str, default=None)
@click.option("--end-cmd", type=str, default=None)
def main(xls_file, start_cmd, end_cmd):
    if start_cmd:
        os.system(start_cmd)
    srtml.init()

    df = pd.read_excel(xls_file, sheet_name="Arrival Information")
    columns = [
        "mu (qps)",
        "cv",
        "# requests",
        "Latency Constraint (ms)",
        "Planner",
        "configuration",
        "Estimated Latency (ms)",
        "Cost",
        "Accuracy",
    ]
    new_df = pd.DataFrame(columns=columns)

    for index, row in df.iterrows():
        row_df = {columns[i]: df.loc[index, columns[i]] for i in range(5)}

        with VGraph(name="classifier") as graph:
            classifier = Classifier(
                feature=IMAGE_CLASSIFICATION_FEATURE,
                dataset_information=dinfo,
                name="VClassifier",
            )

        arrival_curve = generate_fixed_arrival_process(
            mean_qps=df.loc[index, columns[0]],
            cv=df.loc[index, columns[1]],
            num_requests=df.loc[index, columns[2]],
        ).tolist()

        config = graph.configure(
            throughput_qps_constraint=df.loc[index, columns[0]],
            latency_ms_constraint=df.loc[index, columns[3]],
            planner_kwargs={"inter_arrival_process": arrival_curve},
            planner_cls=PLANNER_CLS.get(df.loc[index, columns[4]], None),
            print_state=True,
            materialize=False,
        )

        estimated_values = graph.state.get_estimate_values()._asdict()

        row_df["configuration"] = json.dumps(config)
        row_df["Estimated Latency (ms)"] = estimated_values["latency"]
        row_df["Cost"] = estimated_values["cost"]
        row_df["Accuracy"] = estimated_values["accuracy"]

        new_df = new_df.append(row_df, ignore_index=True)

    shutdown()
    if end_cmd:
        os.system(end_cmd)

    with pd.ExcelWriter(xls_file, mode="a") as writer:
        new_df.to_excel(writer, sheet_name="Planner Configuration")


if __name__ == "__main__":
    main()
