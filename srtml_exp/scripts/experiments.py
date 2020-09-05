import click
import os
from srtml_exp.utils import IMAGE_CLASSIFICATION_DIR_TWO_VERTEX


@click.command()
@click.option("--type", type=str, default="two_vertex")
@click.option("--exp-type", type=str, default="accuracy_degradation")
@click.option("--xls-file", type=str, default=None)
@click.option("--config-file", type=str, default=None)
@click.option("--start-cmd", type=str, default=None)
@click.option("--end-cmd", type=str, default=None)
def prepoc_profile(type, exp_type, xls_file, config_file, start_cmd, end_cmd):
    if type == "two_vertex" and exp_type == "accuracy_degradation":

        python_file_dir = os.path.join(
            IMAGE_CLASSIFICATION_DIR_TWO_VERTEX,
            "accuracy_degradation",
            "physical",
        )

        if config_file is None:
            config_file = os.path.join(
                IMAGE_CLASSIFICATION_DIR_TWO_VERTEX,
                "accuracy_degradation",
                "physical",
                "default.json",
            )

        if xls_file is None:
            save_xls = "image_classification.xlsx"
            xls_file = os.path.join(python_file_dir, save_xls)

        os.system(
            f"python {os.path.join(python_file_dir, 'model_info_generator.py')}"
            f" --config-json {config_file} --save-path {xls_file}"
        )

        if start_cmd is not None and end_cmd is not None:
            os.system(
                f"python {os.path.join(python_file_dir, 'profiles_models.py')}"
                f" --xls-file {xls_file} --start-cmd {start_cmd}"
                f" --end-cmd {end_cmd}"
            )

        if start_cmd is None and end_cmd is None:
            os.system(
                f"python {os.path.join(python_file_dir, 'profiles_models.py')}"
                f" --xls-file {xls_file}"
            )
        else:
            raise ValueError("Wrong start and end commands specified")


@click.command()
@click.option("--type", type=str, default="two_vertex")
@click.option("--exp-type", type=str, default="accuracy_degradation")
@click.option("--xls-file", type=str, default=None)
@click.option("--start-cmd", type=str, default=None)
@click.option("--end-cmd", type=str, default=None)
@click.option("--cleanmr", type=bool, default=True)
def prepoc_populate(type, exp_type, xls_file, start_cmd, end_cmd, cleanmr):
    if type == "two_vertex" and exp_type == "accuracy_degradation":
        python_file_dir = os.path.join(
            IMAGE_CLASSIFICATION_DIR_TWO_VERTEX,
            "accuracy_degradation",
            "physical",
        )

        if xls_file is None:
            save_xls = "image_classification.xlsx"
            xls_file = os.path.join(python_file_dir, save_xls)

        if start_cmd is not None and end_cmd is not None:
            os.system(
                f"python {os.path.join(python_file_dir, 'populate_s3.py')}"
                f" --xls-file {xls_file} --start-cmd {start_cmd}"
                f" --end-cmd {end_cmd}"
            )

        if start_cmd is None and end_cmd is None:
            os.system(
                f"python {os.path.join(python_file_dir, 'populate_s3.py')}"
                f" --xls-file {xls_file} --cleanmr {cleanmr}"
            )

        else:
            raise ValueError("Wrong start and end commands specified")


@click.command()
@click.option("--type", type=str, default="two_vertex")
@click.option("--exp-type", type=str, default="accuracy_degradation")
@click.option("--xls-file", type=str, default=None)
@click.option("--config-file", type=str, default=None)
@click.option("--start-cmd", type=str, default=None)
@click.option("--end-cmd", type=str, default=None)
def prepoc_configure(type, exp_type, xls_file, config_file, start_cmd, end_cmd):
    if type == "two_vertex" and exp_type == "accuracy_degradation":

        python_file_dir = os.path.join(
            IMAGE_CLASSIFICATION_DIR_TWO_VERTEX,
            "accuracy_degradation",
            "virtual",
        )

        if config_file is None:
            config_file = os.path.join(
                IMAGE_CLASSIFICATION_DIR_TWO_VERTEX,
                "accuracy_degradation",
                "virtual",
                "default.json",
            )

        if xls_file is None:
            save_xls = "virtual_image_classification.xlsx"
            xls_file = os.path.join(python_file_dir, save_xls)

        os.system(
            f"python "
            f"{os.path.join(python_file_dir,'virtual_info_generator.py')}"
            f" --config-json {config_file} --save-path {xls_file}"
        )

        if start_cmd is not None and end_cmd is not None:
            os.system(
                f"python "
                f"{os.path.join(python_file_dir,'virtual_prepoc_configure.py')}"
                f" --xls-file {xls_file} --start-cmd {start_cmd}"
                f" --end-cmd {end_cmd}"
            )

        if start_cmd is None and end_cmd is None:
            os.system(
                f"python "
                f"{os.path.join(python_file_dir,'virtual_prepoc_configure.py')}"
                f" --xls-file {xls_file}"
            )
        else:
            raise ValueError("Wrong start and end commands specified")


@click.command()
@click.option("--type", type=str, default="two_vertex")
@click.option("--exp-type", type=str, default="accuracy_degradation")
@click.option("--xls-file", type=str, default=None)
@click.option("--start-cmd", type=str, default=None)
@click.option("--end-cmd", type=str, default=None)
def prepoc_provision(type, exp_type, xls_file, start_cmd, end_cmd):
    if type == "two_vertex" and exp_type == "accuracy_degradation":

        python_file_dir = os.path.join(
            IMAGE_CLASSIFICATION_DIR_TWO_VERTEX,
            "accuracy_degradation",
            "virtual",
        )

        if xls_file is None:
            save_xls = "virtual_image_classification.xlsx"
            xls_file = os.path.join(python_file_dir, save_xls)

        if start_cmd is not None and end_cmd is not None:
            os.system(
                f"python {os.path.join(python_file_dir, 'virtual_prepoc.py')}"
                f" --xls-file {xls_file} --start-cmd {start_cmd}"
                f" --end-cmd {end_cmd}"
            )

        if start_cmd is None and end_cmd is None:
            os.system(
                f"python {os.path.join(python_file_dir, 'virtual_prepoc.py')}"
                f" --xls-file {xls_file}"
            )
        else:
            raise ValueError("Wrong start and end commands specified")
