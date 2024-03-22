# -*- coding = utf-8 -*-

import time
import torch.nn as nn

class BasicModel(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs, device):
        super(BasicModel, self).__init__()
        # if configs.model == 'Merge':
        #     print(f'{"Input: ":<12}' + configs.model + ' start...')
        #     print('----------------------------------------------------')
        # else:
        #     print(f'{configs.data_label + ": ":<12}' + configs.model + ' start...')
        # time.sleep(0.1)
        self.args = configs
        self.device = device
        #self.data_analysis = DataAnalysis(args=configs, device=self.device)

    def forward(self, *args, **kwargs):
        pass
