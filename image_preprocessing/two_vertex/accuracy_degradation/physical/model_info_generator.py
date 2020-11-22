import pandas as pd
import click

from srtml_exp import IMAGE_DATASET_INFORMATION, IMAGE_CLASSIFICATION_FEATURE
import json


@click.command()
@click.option("--config-json", type=str, default="default.json")
@click.option("--save-path", type=str, default="bert24.xlsx")
def main(config_json, save_path):
    with open(config_json, "r") as fp:
        model_configs = json.load(fp)
    profile_config = {
        "warmup": 200,
        "num_requests": 500,
        "percentile": 99,
    }

    df = pd.DataFrame(
        columns=[
            "Model Name",
            "Accuracy",
            "Dataset Information",
            "feature",
            "profile configuration",
        ]
    )
    for model_config in model_configs:
        df = df.append(
            {
                "Model Name": model_config["Model Name"],
                "Accuracy": model_config["Accuracy"],
                "Dataset Information": json.dumps(IMAGE_DATASET_INFORMATION),
                "feature": IMAGE_CLASSIFICATION_FEATURE,
                "profile configuration": json.dumps(profile_config),
            },
            ignore_index=True,
        )

    with pd.ExcelWriter(save_path, mode="w") as writer:
        df.to_excel(writer, sheet_name="Model Information")


if __name__ == "__main__":
    main()
