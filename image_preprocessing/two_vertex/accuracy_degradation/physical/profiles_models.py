from srtml_exp import (
    IMAGE_CLASSIFICATION_DIR,
    Experiment,
    Plotter,
    get_dataframe_from_profile,
)

from srtml.modellib import LOCAL_MODE

from typing import Any
from torch.autograd import Variable
from PIL import Image
import base64
import torch
import torchvision.transforms as transforms
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

###################################################################################################

from transformers import BertTokenizer, BertConfig
import torch
from transformers.modeling_bert import BertLayer, BertEmbeddings, BertPooler, BertPreTrainingHeads
from torch.nn import LayerNorm as BertLayerNorm

class Stage0(torch.nn.Module):
    def __init__(self, config):
        super(Stage0, self).__init__()
        self.embedding_layer = BertEmbeddings(config)
        self.layers = []
        print(config.num_hidden_layers)
        for i in range(config.num_hidden_layers // 8):
            self.layers.append(BertLayer(config))
        self.layers = torch.nn.ModuleList(self.layers)
        self.config = config
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input0, input1):
        out0 = input0
        out1 = input1
        out = self.embedding_layer(out0, out1)
        for layer in self.layers:
            out,  = layer(out)
        return out

class Stage1(torch.nn.Module):
    def __init__(self, config):
        super(Stage1, self).__init__()
        self.layers = []
        for i in range(config.num_hidden_layers // 8):
            self.layers.append(BertLayer(config))
        self.layers = torch.nn.ModuleList(self.layers)
        self.config = config;
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input0):
        out = input0
        for layer in self.layers:
            out,  = layer(out)
        return out


class Stage2(torch.nn.Module):
    def __init__(self, config):
        super(Stage2, self).__init__()
        self.layers = []
        for i in range(config.num_hidden_layers // 8):
            self.layers.append(BertLayer(config))
        self.layers = torch.nn.ModuleList(self.layers)
        self.config = config;
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input0):
        out = input0
        for layer in self.layers:
            out,  = layer(out)
        return out

class Stage3(torch.nn.Module):
    def __init__(self, config):
        super(Stage3, self).__init__()
        self.layers = []
        for i in range(config.num_hidden_layers // 8):
            self.layers.append(BertLayer(config))
        self.layers = torch.nn.ModuleList(self.layers)
        self.pooling_layer = BertPooler(config)
        self.pre_training_heads_layer = BertPreTrainingHeads(config)
        self.config = config;
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input0):
        out = input0
        for layer in self.layers:
            out,  = layer(out)
        out2 = self.pooling_layer(out)
        out3 = self.pre_training_heads_layer(out, out2)
        return out3

##################################################################################################

config = BertConfig.from_pretrained('bert-large-uncased')

# PPUs START
#####################################################################################################
@ppu
class Tokenizer:
    """
    Standard pytorch pre-processing functionality
    - gets a raw image
    - converts it to tensor
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @ppu_type(
        hardware_reqs="Hardware.CPU.CPU1",
        accept_batch=True,
    )
    def __call__(self, data: list) -> list:
        data_list = list()
        for txt in data:
            encoded = self.tokenizer(text=txt, add_special_tokens=True,  # Add [CLS] and [SEP]
                                 max_length = 64,  # maximum length of a sentence
                                 padding='max_length',  # Add [PAD]s
                                 return_attention_mask = True,  # Generate the attention mask
                                 return_tensors = 'pt')
            data_list.append(encoded)
        return data_list


@ppu
class BertPartition:
    def __init__(self, model: Any, is_cuda: bool = False) -> None:
        self.model = model
        self.is_cuda = is_cuda
        if is_cuda:
            self.model = self.model.cuda()

    @ppu_type(
        hardware_reqs="Hardware.GPU.Nvidia.Tesla_P40",
        accept_batch=True,
    )
    def __call__(self, data: list) -> list:
        input0 = torch.stack([item['input_ids'][0] for item in data])
        input1 = torch.stack([item['attention_mask'][0] for item in data])

        if self.is_cuda:
            input0 = input0.cuda()
            input1 = input1.cuda()
        outputs = self.model(input0, input1)
        res = list(outputs.cpu().unbind())
        # res = [[a, b] for a, b in zip(res[0], res[1])]
        return res

@ppu
class BertIntermediate:
    def __init__(self, model: Any, is_cuda: bool = False) -> None:
        self.model = model
        self.is_cuda = is_cuda
        if is_cuda:
            self.model = self.model.cuda()

    @ppu_type(
        hardware_reqs="Hardware.GPU.Nvidia.Tesla_P40",
        accept_batch=True,
    )
    def __call__(self, data: list) -> list:
        input0 = torch.stack(data)
        if self.is_cuda:
            input0 = input0.cuda()
        outputs = self.model(input0)
        res = list(outputs.cpu().unbind())
        return res
        
@ppu
class BertFinalPartition:
    def __init__(self, model: Any, is_cuda: bool = False) -> None:
        self.model = model
        self.is_cuda = is_cuda
        if is_cuda:
            self.model = self.model.cuda()

    @ppu_type(
        hardware_reqs="Hardware.GPU.Nvidia.Tesla_P40",
        accept_batch=True,
    )
    def __call__(self, data: list) -> list:
        input0 = torch.stack(data)
        if self.is_cuda:
            input0 = input0.cuda()
        outputs = self.model(input0)
        res = [i.cpu().unbind()[1] for i in outputs[1]]
        return [1] * len(data)

# PPUs END
#####################################################################################################

def create_pgraph(model_name = 'gg'):

    with PGraph(name=f"Sentimental-{model_name}") as graph:

        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        txt = 'Debugging'
        encoded = tokenizer(text=txt, add_special_tokens=True,  # Add [CLS] and [SEP]
                                 max_length = 64,  # maximum length of a sentence
                                 padding='max_length',  # Add [PAD]s
                                 return_attention_mask = True,  # Generate the attention mask
                                 return_tensors = 'pt')

        prepoc_dummy_kwarg = {"data": [txt]}
        model_dummy_kwarg = {"data": [encoded]}
        model_dummy_kwarg_1 = {"data": [torch.rand(64, 1024)]}
        model_dummy_kwarg_2 = {"data": [torch.rand(64, 1024)]}
        model_dummy_kwarg_3 = {"data": [torch.rand(64, 1024)]}

        prepoc = Tokenizer(
            _name=f"tokenizer",
            _dummy_kwargs=prepoc_dummy_kwarg,
            tokenizer=tokenizer,
        )

        model = BertPartition(
            _name=f"bert24_p2_stage0",
            _dummy_kwargs=model_dummy_kwarg,
            model = Stage0(config),
            is_cuda=True,
        )

        model_2 = BertIntermediate(
            _name=f"bert24-p2-stage1",
            _dummy_kwargs=model_dummy_kwarg_1,
            model=Stage1(config),
            is_cuda=True,
        )

        model_3 = BertIntermediate(
            _name=f"bert24-p2-stage2",
            _dummy_kwargs=model_dummy_kwarg_1,
            model=Stage2(config),
            is_cuda=True,
        )

        model_4 = BertFinalPartition(
            _name=f"bert24-p2-stage3",
            _dummy_kwargs=model_dummy_kwarg_1,
            model=Stage3(config),
            is_cuda=True,
        )

        # connection
        prepoc >> model >> model_2 >> model_3 >> model_4

    return graph


ray_serve_kwargs={
        "ray_init_kwargs": {
            "object_store_memory": int(10e10),
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


@click.command()
@click.option("--xls-file", type=str, default="bert24.xlsx")
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

        srtml.init(ray_serve_kwargs = ray_serve_kwargs)
        pgraph = create_pgraph(df.loc[index, "Model Name"])
        pgraph.configure(SERVE_MODE)
        pgraph.provision(SERVE_MODE)
        print('done')
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
