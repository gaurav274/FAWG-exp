from typing import Any
import srtml
from srtml.modellib import ppu, ppu_type, PGraph, SERVE_MODE
import ray


import torch
import json
from transformers.modeling_bert import BertLayer
from transformers.modeling_bert import BertEmbeddings
from torch.nn import LayerNorm as BertLayerNorm
from transformers.modeling_bert import BertPooler
from transformers.modeling_bert import BertPreTrainingHeads

###################################################################################################

from transformers import BertTokenizer, BertConfig
import torch
from transformers.modeling_bert import BertLayer, BertEmbeddings, BertPooler, BertPreTrainingHeads
from torch.nn import LayerNorm as BertLayerNorm

class Bert(torch.nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.embedding_layer = BertEmbeddings(config)
        self.layers = []
        num_hidden_layers = config.num_hidden_layers
        # num_hidden_layers -= 10
        for i in range(num_hidden_layers):
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

    def forward(self, input0, input1):
        out0 = input0
        out1 = input1
        out = self.embedding_layer(out0, out1)
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
class BertOriginal:
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
        res = [i.cpu().unbind()[1] for i in outputs[1]]
        return [1] * len(res)


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

        model_dummy_kwarg = {"data": [encoded]}


        model = BertOriginal(
            _name=f"bert",
            _dummy_kwargs=model_dummy_kwarg,
            model = Bert(config),
            is_cuda=True,
        )

        model
    return graph


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
graph = create_pgraph()
graph.configure(SERVE_MODE)
graph.provision(SERVE_MODE)
