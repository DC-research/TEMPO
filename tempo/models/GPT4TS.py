import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from tempo.embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from tempo.utils.rev_in import RevIn
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        # self.mlp = configs.mlp
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        config = LoraConfig(
            # task_type=TaskType.CAUSAL_LM, # causal language model
            r=16,
            lora_alpha=16,
            # target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",               # bias, set to only lora layers to train
            # modules_to_save=["classifier"],
        )
        
        # for i, (name, param) in enumerate(self.gpt2_season.named_parameters()):
        #     if 'ln' in name or 'wpe' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # for i, (name, param) in enumerate(self.gpt2_noise.named_parameters()):
        #     if 'ln' in name or 'wpe' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        
        # self.gpt2 = get_peft_model(self.gpt2, config)
        # print_trainable_parameters(self.gpt2_trend)

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0

        self.num_nodes = 1
        self.rev_in = RevIn(num_features=self.num_nodes).to(device)


    def forward(self, x, itr):
        B, L, M = x.shape # 4, 512, 1

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x) # 4, 1, 420
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #4,1, 64, 16
        x = rearrange(x, 'b m n p -> (b m) n p') # 4, 64, 16

        outputs = self.in_layer(x) # 4, 64, 768
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state # 4, 64, 768

        outputs = self.out_layer(outputs.reshape(B*M, -1)) # 4, 96
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B) # 4, 96, 1

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
