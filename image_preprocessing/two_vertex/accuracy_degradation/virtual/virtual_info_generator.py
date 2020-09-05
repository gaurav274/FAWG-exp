import pandas as pd
import click

from srtml_exp import IMAGE_DATASET_INFORMATION, IMAGE_CLASSIFICATION_FEATURE
import json


@click.command()
@click.option("--config-json", type=str, default="default.json")
@click.option(
    "--save-path", type=str, default="virtual_image_classification.xlsx"
)
def main(config_json, save_path):

    with open(config_json, "r") as fp:
        arrival_configs = json.load(fp)

    df = pd.DataFrame(
        columns=[
            "mu (qps)",
            "cv",
            "# requests",
            "Latency Constraint (ms)",
            "Planner",
        ]
    )
    for arrival_config in arrival_configs:

        df = df.append(
            arrival_config,
            ignore_index=True,
        )
    with pd.ExcelWriter(save_path, mode="w") as writer:
        df.to_excel(writer, sheet_name="Arrival Information")


if __name__ == "__main__":
    main()
