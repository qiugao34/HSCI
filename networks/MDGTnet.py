import torch.nn as nn
from networks.baseblock import *
from networks.intra_DUEB import AEM


class IFEM(nn.Module):
    def __init__(self, in_ch, out_ch: list, padding=0):
        super(IFEM, self).__init__()
        self.block1 = conv_block4(in_ch, out_ch[0], padding=padding)
        self.block2 = conv_block4(out_ch[0], out_ch[1], padding=0)
        self.block3 = conv_block4(out_ch[1], out_ch[2], padding=0)
        self.block4 = conv_block4(out_ch[2], out_ch[3], padding=0)

    def forward(self, x, x_side: list):
        y_main_1, y_side_1 = self.block1(x, x_side[0])
        y_main_2, y_side_2 = self.block2(y_main_1, x_side[1])
        y_main_3, y_side_3 = self.block3(y_main_2, x_side[2])
        y_main_4, y_side_4 = self.block4(y_main_3, x_side[3])

        return y_main_1, y_main_2, y_main_3, y_main_4, y_side_1, y_side_2, y_side_3, y_side_4


class MDGTnet(nn.Module):
    def __init__(self, in_ch, out_ch: list, padding, slice_size: int, spec_range: list, 
                 class_num: int, drop_path_rate=0.05, layer_scale_init_value=1e-6, depths=[3, 3, 9, 3]):
        super(MDGTnet, self).__init__()

        self.AEM = AEM(in_ch, padding, slice_size, spec_range)
        # self.intra_1 = conv_block2(in_ch, out_ch[0], padding=padding)
        # self.intra_2 = conv_block2(out_ch[0], out_ch[1])
        # self.intra_3 = conv_block2(out_ch[1], out_ch[2])
        # self.intra_4 = conv_block2(out_ch[2], out_ch[3])

        self.IFEH = nn.Sequential(conv_block1(in_ch, in_ch * out_ch[0]),
                                  conv_block1(in_ch * out_ch[0], in_ch * out_ch[0]))
        self.IFEM = IFEM(in_ch * out_ch[0], [out_ch[0] * in_ch, out_ch[1] * in_ch, out_ch[2] * in_ch, out_ch[3] * in_ch], padding=padding)

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(slice_size*slice_size*out_ch[3] * in_ch, out_ch[6]),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(out_ch[6], out_ch[7]),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(out_ch[7], class_num))
        
        self.stages = nn.ModuleList() # 4 feature resolution stages
        self.down_samples = nn.ModuleList()
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        dims = [1]
        dims.extend(out_ch[:4])
        for i in range(4):
            downsample_layer = nn.Sequential(
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(dims[i + 1])
                )
            if i < 2:
                stage = nn.Sequential(
                    *[Block(dim=dims[i + 1], drop_path=dp_rates[cur + j], 
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                    *[Transformer(dim=dims[i + 1], heads=4, drop_path=dp_rates[cur + j],
                                  init_values=layer_scale_init_value) for j in range(depths[i])]
                )
            self.stages.append(stage)
            self.down_samples.append(downsample_layer)
            cur += depths[i]

    def forward(self, x_intra, x_inter):
        y_intra = self.AEM(x_intra)
        # x_side = [y_intra_1, y_intra_2, y_intra_3, y_intra_4]
        # y_intra_1 = self.intra_1(y_intra)
        # y_intra_2 = self.intra_2(y_intra_1)
        # y_intra_3 = self.intra_3(y_intra_2)
        # y_intra_4 = self.intra_4(y_intra_3)
        B, C, H, W = y_intra.shape
        y_intra = y_intra.reshape(B, -1, C, H, W)
        x_side = []
        for i in range(4):
            y_intra = self.down_samples[i](y_intra)
            y_intra = self.stages[i](y_intra)
            B, C_, S, H_, W_ = y_intra.shape
            x_side.append(y_intra.reshape(B, -1, H_, W_))
        y_inter = self.IFEH(x_inter)
        __, __, __, y_inter_4, y_side_1, y_side_2, y_side_3, y_side_4 = self.IFEM(y_inter, x_side)
        y = self.classifier(y_inter_4).squeeze()

        return y, y_side_1, y_side_2, y_side_3, y_side_4

