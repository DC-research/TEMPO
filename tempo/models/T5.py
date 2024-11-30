import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config # import T5 package
from einops import rearrange



class T54TS(nn.Module):
    
    def __init__(self, configs, device):
        super(T54TS, self).__init__()
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
               
                self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base') # load T5
                # self.t5 = t5Model.from_pretrained('t5', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.t5 = T5ForConditionalGeneration(T5Config()) # load T5
        
            self.t5.encoder.block = self.t5.encoder.block[:configs.gpt_layers]
            print("t5= {}".format(self.t5))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.t5.named_parameters()):
                # if 'ln' in name or 'wpe' in name:
                #     param.requires_grad = True
                # else:
                #     param.requires_grad = False
                if 'layer_norm' in name or 'relative_attention_bias' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.t5, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


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
            encoder_outputs = self.t5.encoder(inputs_embeds=outputs)
            last_hidden_state = encoder_outputs.last_hidden_state
            outputs = last_hidden_state # 8 3 768

        outputs = self.out_layer(outputs.reshape(B*M, -1)) # 4, 96
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B) # 4, 96, 1

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs