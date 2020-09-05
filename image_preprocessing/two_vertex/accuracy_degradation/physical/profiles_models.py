from srtml_exp import (
    IMAGE_CLASSIFICATION_DIR,
    Experiment,
    Plotter,
    get_dataframe_from_profile,
)


from typing import Any
from torch.autograd import Variable
from PIL import Image
import base64
import torch
import torchvision.transforms as transforms
from torchvision import models
import io
import os
import srtml
from srtml.modellib import ppu, ppu_type, PGraph, SERVE_MODE
import ray
from srtml.profiler.profile_actor import profile_pgraph
from pprint import pprint
from srtml_exp import shutdown, get_sysinfo

import pandas as pd
import click
import json


@ppu
class Transform:
    """
    Standard pytorch pre-processing functionality
    - gets a raw image
    - converts it to tensor
    """

    def __init__(self, transform: Any) -> None:

        self.transform = transform

    @ppu_type(
        output_shape=(3, 224, 224),
        hardware_reqs="Hardware.CPU.CPU1",
        accept_batch=True,
    )
    def __call__(self, data: list) -> list:
        data_list = list()
        for img in data:
            data = Image.open(io.BytesIO(base64.b64decode(img)))
            if data.mode != "RGB":
                data = data.convert("RGB")
            data = self.transform(data)
            data_list.append(data)
        return data_list


@ppu
class PredictModelPytorch:
    """
    Standard pytorch prediction functionality
    - gets a preprocessed tensor
    - predicts it's class
    """

    def __init__(self, model_name: str, is_cuda: bool = False) -> None:
        self.model = models.__dict__[model_name](pretrained=True)
        self.is_cuda = is_cuda
        if is_cuda:
            self.model = self.model.cuda()

    @ppu_type(
        input_shape=(3, 224, 224),
        hardware_reqs="Hardware.GPU.Nvidia.Tesla_P40",
        accept_batch=True,
    )
    def __call__(self, data: list) -> list:
        data = torch.stack(data)
        data = Variable(data)
        if self.is_cuda:
            data = data.cuda()
        outputs = self.model(data)
        _, predicted = outputs.max(1)
        return predicted.cpu().numpy().tolist()


def create_pgraph(model_name):

    with PGraph(name=f"Classifier-{model_name}") as graph:

        min_img_size = 224
        transform = transforms.Compose(
            [
                transforms.Resize(min_img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        img = base64.b64encode(
            open(
                os.path.join(IMAGE_CLASSIFICATION_DIR, "elephant.jpg"), "rb"
            ).read()
        )

        prepoc_dummy_kwarg = {"data": [img]}

        model_dummy_kwarg = {"data": [torch.zeros((3, 224, 224))]}

        prepoc = Transform(
            _name=f"prepoc-{model_name}",
            _dummy_kwargs=prepoc_dummy_kwarg,
            transform=transform,
        )

        model = PredictModelPytorch(
            _name=f"model-{model_name}",
            _dummy_kwargs=model_dummy_kwarg,
            model_name=model_name,
            is_cuda=True,
        )

        # connection
        prepoc >> model

    return graph


# srtml.init()
# # graph.configure(SERVE_MODE)
# # graph.provision(SERVE_MODE)

# # new_img = base64.b64encode(
# #     open(os.path.join(IMAGE_CLASSIFICATION_DIR, "elephant.jpg"), "rb").read()
# # )
# # print(ray.get(graph.remote(data=new_img)))
# profile_dict = profile_pgraph(
#     graph, warmup=200, num_requests=500, percentile=99
# )
# pprint(profile_dict)


@click.command()
@click.option("--xls-file", type=str, default="image_classification.xlsx")
@click.option("--start-cmd", type=str, default=None)
@click.option("--end-cmd", type=str, default=None)
def main(xls_file, start_cmd, end_cmd):
    if start_cmd:
        assert end_cmd is not None, "Wrong input"
    if end_cmd:
        assert start_cmd is not None, "Wrong input"

    df = pd.read_excel(xls_file, sheet_name="Model Information")
    clean_profile_df = pd.DataFrame()
    raw_profile_df = pd.DataFrame(
        columns=[
            "pgraph",
            "raw profile",
            "Dataset Information",
            "feature",
            "Model Name",
            "Accuracy",
            "sysinfo",
        ]
    )
    for index, row in df.iterrows():

        if start_cmd:
            os.system(start_cmd)
        srtml.init()

        # profile_df = pd.DataFrame(
        #     columns=[
        #         "pgraph"
        #     ]
        # )
        pgraph = create_pgraph(df.loc[index, "Model Name"])

        profile_dict = profile_pgraph(
            pgraph, **json.loads(df.loc[index, "profile configuration"])
        )

        pprint(profile_dict)
        clean_profile_df = pd.concat(
            [
                clean_profile_df,
                get_dataframe_from_profile(pgraph.ppu_identifier, profile_dict),
            ]
        )
        raw_profile_df = raw_profile_df.append(
            {
                "pgraph": pgraph.ppu_identifier,
                "raw profile": json.dumps(profile_dict),
                "Dataset Information": df.loc[index, "Dataset Information"],
                "feature": df.loc[index, "feature"],
                "Model Name": df.loc[index, "Model Name"],
                "Accuracy": df.loc[index, "Accuracy"],
                "sysinfo": get_sysinfo(),
            },
            ignore_index=True,
        )

        shutdown()
        if end_cmd:
            os.system(end_cmd)

    with pd.ExcelWriter(xls_file, mode="a") as writer:
        clean_profile_df.to_excel(writer, sheet_name="Model Profile")
        raw_profile_df.to_excel(writer, sheet_name="Model Raw Profile")

    # for index, row in df.iterrows():
    #     pass


if __name__ == "__main__":
    main()
