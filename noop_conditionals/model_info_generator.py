import pandas as pd
import click

from srtml_exp import IMAGE_DATASET_INFORMATION, IMAGE_CLASSIFICATION_FEATURE
import json


@click.command()
@click.option("--config-json", type=str, default="default.json")
@click.option("--save-path", type=str, default="noop_conditionals.xlsx")
def main(config_json, save_path):
    with open(config_json, "r") as fp:
        model_configs = json.load(fp)

    df = pd.DataFrame(
        columns=[
            "Model Name",
            "Number of Conditionals",
            "mu (qps)",
            "Latency Constraint (ms)",
            "cv",
            "# requests",
        ]
    )
    for model_config in model_configs:
        df = df.append(
            {
                "Model Name": model_config["Model Name"],
                "Number of Conditionals": model_config["Number of Conditionals"],
                "mu (qps)": model_config["mu (qps)"],
                "Latency Constraint (ms)": model_config["Latency Constraint (ms)"],
                "cv": model_config["cv"],
                "# requests": model_config["# requests"]
            },
            ignore_index=True,
        )
    with pd.ExcelWriter(save_path, mode="w") as writer:
        df.to_excel(writer, sheet_name="Model Information")


if __name__ == "__main__":
    main()
