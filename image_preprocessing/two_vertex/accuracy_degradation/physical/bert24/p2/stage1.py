import torch
from transformers.modeling_bert import BertLayer
from transformers.modeling_bert import BertPreTrainingHeads
from transformers.modeling_bert import BertPooler
from torch.nn import LayerNorm as BertLayerNorm

class Stage1(torch.nn.Module):
    def __init__(self, config):
        super(Stage1, self).__init__()
        self.layers = []
        for i in range(config.num_hidden_layers // 2):
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

    def forward(self, input1, input0):
        out0 = input0
        out1 = input1
        out = out0
        for layer in self.layers:
            out,  = layer(out, out1)
        out2 = self.pooling_layer(out)
        out3 = self.pre_training_heads_layer(out, out2)
        return out3
