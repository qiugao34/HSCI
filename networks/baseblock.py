import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_, DropPath
from einops.layers.torch import Rearrange

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class conv_block1(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(conv_block1, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=padding),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.net(x)

        return y


class conv_block2(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(conv_block2, self).__init__()
        self.net_block1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=padding),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU(inplace=True))
        self.ca1 = ChannelAttention(out_ch)
        self.net_block2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU(inplace=True))
        self.net_block3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(out_ch),
                                        nn.ReLU(inplace=True))
        self.ca2 = ChannelAttention(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=padding)

    def forward(self, x):
        y = self.net_block1(x)
        w_1 = self.ca1(y)
        y = y.mul(w_1)
        y = self.net_block2(y)
        w_2 = self.ca2(y)
        y = y.mul(w_2)
        y += self.conv1(x)

        return y


class conv_block3(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(conv_block3, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=padding),
                                 nn.BatchNorm2d(out_ch),
                                 nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=padding),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.net(x)

        return y


class conv11_block(nn.Module):
    def __init__(self, in_ch):
        super(conv11_block, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, 2*in_ch, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(2*in_ch),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(2*in_ch, in_ch, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(in_ch),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.net(x)

        return y


class conv_block4(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(conv_block4, self).__init__()
        self.net_main = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=padding),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_ch, 2 * out_ch, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(2*out_ch),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(2 * out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(out_ch))
        self.net_side = nn.Sequential(nn.Conv2d(out_ch, int(out_ch/15), kernel_size=3, stride=1, padding=0),
                                      nn.BatchNorm2d(int(out_ch/15)))
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=padding),
                                  nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_main, x_side):
        y_main = self.net_main(x_main)
        y = y_main + self.conv(x_main) - x_side
        y_side = self.net_side(y_main - x_side)
        y_side = torch.softmax(y_side, dim=1)

        return y, y_side

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim)
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                             requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 4, 1) # (N, C, S, H, W) -> (N, S, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, S, H, W, C) -> (N, C, S, H, W)

        x = input + self.drop_path(x)
        return x

class unfold_3d(nn.Module):
    def __init__(self, kernel_size, stride, padd=[1, 1, 1], padd_mode='replicate'):
        super(unfold_3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padd = padd
        self.padd_mode = padd_mode

    def forward(self, x):
        x = F.pad(x, (self.padd[0], self.padd[0], self.padd[1], self.padd[1], self.padd[2], self.padd[2]),
                  mode=self.padd_mode)
        # x = F.ReflectionPad3d(x, [0, 0, padd[0], padd[1], padd[2]], mode=padd_mode)
        x = x.unfold(2, self.kernel_size[0], self.stride[0]) \
            .unfold(3, self.kernel_size[1], self.stride[1]) \
            .unfold(4, self.kernel_size[2], self.stride[2])
        # x = rearrange(x, 'b c h w d k1 k2 k3 -> b h w d (k1 k2 k3) c')
        x = rearrange(x, 'b c h w d k1 k2 k3 -> b (h w d) (k1 k2 k3) c')
        return x

class MLP_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim * 3),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim * 3, dim, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim)
        )

    def forward(self, x):
        return self.net(x)

class Rconv_3D(nn.Module):
    def __init__(self, dim, kernel_size=[3, 3, 3], stride=[1, 1, 1], heads=4):
        super(Rconv_3D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padd = [kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2]
        self.num_heads = heads

        self.proj = nn.Conv3d(dim, dim, kernel_size=1)

        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Sequential(
            nn.Conv3d(dim, dim * 3, kernel_size=1),
            nn.Conv3d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        )

        self.norm = nn.Sequential(
            Rearrange('b c h w d -> b (h w d) c'),
            # nn.LayerNorm(dim)
        )

        self.unfold_k = nn.Sequential(
            unfold_3d(kernel_size=self.kernel_size, stride=self.stride, padd=self.padd),
            # nn.LayerNorm(dim)
        )
        self.unfold_v = nn.Sequential(
            unfold_3d(kernel_size=self.kernel_size, stride=self.stride, padd=self.padd),
            # nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        B, C, H, W, S = x.shape

        qkv = self.qkv(x).reshape(B, 3, C, H, W, S)
        q, k, v = qkv.unbind(1)
        q = self.norm(q)
        k = self.unfold_k(k)
        v = self.unfold_v(v)

        B, L, K, C = k.shape
        # q = q.reshape(B, self.num_heads, L, 1, -1)
        q = q.contiguous().view(B, self.num_heads, L, 1, -1)  # (B,head,(h*w*d),1,c/head)
        k = k.view(B, self.num_heads, L, K, -1)  # (B,head,(hwd),27,c/head)
        v = v.view(B, self.num_heads, L, K, -1)

        attn = q @ k.transpose(-2, -1)  # (B,head,(hwd),1,27)
        attn = (attn * self.scale).softmax(dim=-1)

        x = (attn @ v).transpose(1, 2)  # (B,head,(hwd),1,c/head)
        # x = torch.einsum('bhlxk,bhlkc->bhlxc', attn, v)
        x = x.reshape(B, L, C).transpose(-2, -1).reshape(B, C, H, W, S)  # B, n, C
        x = self.proj(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, heads, init_values=1e-4, drop_path=0.2):
        super(Transformer, self).__init__()
        # self.layers = nn.ModuleList([])  #它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器

        self.norm1 = nn.BatchNorm3d(dim)
        self.norm2 = nn.BatchNorm3d(dim)

        self.mlp = nn.Sequential(
            MLP_Block(dim=dim),
            # Rearrange('B C S H W-> B S H W C')
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = nn.Sequential(
            Rconv_3D(
                dim, heads=heads),
            # Rearrange('B C S H W-> B S H W C')
        )

    def forward(self, x):
        # x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x