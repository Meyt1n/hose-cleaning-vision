# Derived from OpenMMLab MMPose commit759b39c;Apache2 license retained under official_mmpose_commit759b39c/LICENSE.
import math,re
from typing import Optional,Sequence,Tuple,List,Union
import torch
from torch import Tensor,nn
import torch.nn.functional as F
ConfigType=dict;OptConfigType=Optional[dict];OptMultiConfig=Optional[dict];OptMultiConfig=Optional[dict]
TORCH_VERSION=torch.__version__
def digit_version(v):return tuple(int(x) for x in re.findall(r'\d+',str(v).split('+')[0])[:3])
class BaseModule(nn.Module):
    def __init__(self,init_cfg=None):super().__init__()
class DropPath(nn.Identity):
    def __init__(self,p=0):
        if p:raise ValueError('ONLY_OFFICIAL_HAND5_ZERO_DROPPATH_SUPPORTED')
        super().__init__()
class ConvModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias='auto',conv_cfg=None,norm_cfg=None,act_cfg=dict(type='ReLU'),**kwargs):
        super().__init__()
        if conv_cfg is not None:raise ValueError('CUSTOM_CONV_NOT_SUPPORTED')
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias=(norm_cfg is None) if bias=='auto' else bias)
        self.with_norm=norm_cfg is not None
        if self.with_norm:
            cls=nn.SyncBatchNorm if norm_cfg['type']=='SyncBN' else nn.BatchNorm2d
            self.bn=cls(out_channels,eps=norm_cfg.get('eps',1e-5),momentum=norm_cfg.get('momentum',.1))
        typ=(act_cfg or {}).get('type')
        self.activate=nn.Identity() if typ is None else nn.SiLU(inplace=True) if typ in ['SiLU','Swish'] else nn.ReLU(inplace=True) if typ=='ReLU' else None
        if self.activate is None:raise ValueError('ACTIVATION_NOT_SUPPORTED')
    def forward(self,x):
        x=self.conv(x)
        if self.with_norm:x=self.bn(x)
        return self.activate(x)
class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,norm_cfg=None,act_cfg=dict(type='ReLU'),conv_cfg=None,**kwargs):
        super().__init__()
        self.depthwise_conv=ConvModule(in_channels,in_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=in_channels,norm_cfg=norm_cfg,act_cfg=act_cfg,conv_cfg=conv_cfg)
        self.pointwise_conv=ConvModule(in_channels,out_channels,1,norm_cfg=norm_cfg,act_cfg=act_cfg,conv_cfg=conv_cfg)
    def forward(self,x):return self.pointwise_conv(self.depthwise_conv(x))

class ScaleNorm(nn.Module):
    """Scale Norm.

    Args:
        dim (int): The dimension of the scale vector.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.

    Reference:
        `Transformers without Tears: Improving the Normalization
        of Self-Attention <https://arxiv.org/abs/1910.05895>`_
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The tensor after applying scale norm.
        """

        if torch.onnx.is_in_onnx_export() and \
                digit_version(TORCH_VERSION) >= digit_version('1.12'):

            norm = torch.linalg.norm(x, dim=-1, keepdim=True)

        else:
            norm = torch.norm(x, dim=-1, keepdim=True)
        norm = norm * self.scale
        return x / norm.clamp(min=self.eps) * self.g

class ChannelAttention(BaseModule):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self, channels: int, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        if digit_version(torch.__version__) < (1, 7, 0):
            self.act = nn.Hardsigmoid()
        else:
            self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out

class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): The kernel size of the convolution.
            Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out.
            Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out

class CSPNeXtBlock(BaseModule):
    """The basic bottleneck block used in CSPNeXt.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): Expand ratio of the hidden channel. Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out. Only works
            when in_channels == out_channels. Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        kernel_size (int): The kernel size of the second convolution layer.
            Defaults to 5.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 kernel_size: int = 5,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out

class CSPLayer(BaseModule):
    """Cross Stage Partial Layer.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1.
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        use_cspnext_block (bool): Whether to use CSPNeXt block.
            Defaults to False.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks. Defaults to False.
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish')
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 channel_attention: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            block(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)

class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.cat(
                [x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x

class CSPNeXt(BaseModule):
    """CSPNeXt backbone used in RTMDet.

    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
            Defaults to P5.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        spp_kernel_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Defaults to (5, 9, 13).
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_ovewrite: dict = None,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        conv_cfg: Optional[ConfigType] = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU'),
        norm_eval: bool = False,
        init_cfg: Optional[ConfigType] = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.stem = nn.Sequential(
            ConvModule(
                3,
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernel_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

def rope(x, dim):
    """Applies Rotary Position Embedding to input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int | list[int]): The spatial dimension(s) to apply
            rotary position embedding.

    Returns:
        torch.Tensor: The tensor after applying rotary position
            embedding.

    Reference:
        `RoFormer: Enhanced Transformer with Rotary
        Position Embedding <https://arxiv.org/abs/2104.09864>`_
    """
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]

    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i

    position = torch.reshape(
        torch.arange(total_len, dtype=torch.int, device=x.device),
        spatial_shape)

    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = torch.unsqueeze(position, dim=-1)

    half_size = shape[-1] // 2
    freq_seq = -torch.arange(
        half_size, dtype=torch.int, device=x.device) / float(half_size)
    inv_freq = 10000**-freq_seq

    sinusoid = position[..., None] * inv_freq[None, None, :]

    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class Scale(nn.Module):
    """Scale vector by element multiplications.

    Args:
        dim (int): The dimension of the scale vector.
        init_value (float, optional): The initial value of the scale vector.
            Defaults to 1.0.
        trainable (bool, optional): Whether the scale vector is trainable.
            Defaults to True.
    """

    def __init__(self, dim, init_value=1., trainable=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        """Forward function."""

        return x * self.scale

class RTMCCBlock(nn.Module):
    """Gated Attention Unit (GAU) in RTMBlock.

    Args:
        num_token (int): The number of tokens.
        in_token_dims (int): The input token dimension.
        out_token_dims (int): The output token dimension.
        expansion_factor (int, optional): The expansion factor of the
            intermediate token dimension. Defaults to 2.
        s (int, optional): The self-attention feature dimension.
            Defaults to 128.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
        drop_path (float, optional): The drop path rate. Defaults to 0.0.
        attn_type (str, optional): Type of attention which should be one of
            the following options:

            - 'self-attn': Self-attention.
            - 'cross-attn': Cross-attention.

            Defaults to 'self-attn'.
        act_fn (str, optional): The activation function which should be one
            of the following options:

            - 'ReLU': ReLU activation.
            - 'SiLU': SiLU activation.

            Defaults to 'SiLU'.
        bias (bool, optional): Whether to use bias in linear layers.
            Defaults to False.
        use_rel_bias (bool, optional): Whether to use relative bias.
            Defaults to True.
        pos_enc (bool, optional): Whether to use rotary position
            embedding. Defaults to False.

    Reference:
        `Transformer Quality in Linear Time
        <https://arxiv.org/abs/2202.10447>`_
    """

    def __init__(self,
                 num_token,
                 in_token_dims,
                 out_token_dims,
                 expansion_factor=2,
                 s=128,
                 eps=1e-5,
                 dropout_rate=0.,
                 drop_path=0.,
                 attn_type='self-attn',
                 act_fn='SiLU',
                 bias=False,
                 use_rel_bias=True,
                 pos_enc=False):

        super(RTMCCBlock, self).__init__()
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.e = int(in_token_dims * expansion_factor)
        if use_rel_bias:
            if attn_type == 'self-attn':
                self.w = nn.Parameter(
                    torch.rand([2 * num_token - 1], dtype=torch.float))
            else:
                self.a = nn.Parameter(torch.rand([1, s], dtype=torch.float))
                self.b = nn.Parameter(torch.rand([1, s], dtype=torch.float))
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        if attn_type == 'self-attn':
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)
            nn.init.xavier_uniform_(self.k_fc.weight)
            nn.init.xavier_uniform_(self.v_fc.weight)

        self.ln = ScaleNorm(in_token_dims, eps=eps)

        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn == 'SiLU' or act_fn == nn.SiLU:
            assert digit_version(TORCH_VERSION) >= digit_version('1.7.0'), \
                'SiLU activation requires PyTorch version >= 1.7'

            self.act_fn = nn.SiLU(True)
        elif act_fn == 'ReLU' or act_fn == nn.ReLU:
            self.act_fn = nn.ReLU(True)
        else:
            raise NotImplementedError

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False

        self.sqrt_s = math.sqrt(s)

        self.dropout_rate = dropout_rate

        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len, k_len=None):
        """Add relative position bias."""

        if self.attn_type == 'self-attn':
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(k_len, 1), dim=0)
            t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def _forward(self, inputs):
        """GAU Forward function."""

        if self.attn_type == 'self-attn':
            x = inputs
        else:
            x, k, v = inputs

        x = self.ln(x)

        # [B, K, in_token_dims] -> [B, K, e + e + s]
        uv = self.uv(x)
        uv = self.act_fn(uv)

        if self.attn_type == 'self-attn':
            # [B, K, e + e + s] -> [B, K, e], [B, K, e], [B, K, s]
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
            # [B, K, 1, s] * [1, 1, 2, s] + [2, s] -> [B, K, 2, s]
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta

            if self.pos_enc:
                base = rope(base, dim=1)
            # [B, K, 2, s] -> [B, K, s], [B, K, s]
            q, k = torch.unbind(base, dim=2)

        else:
            # [B, K, e + s] -> [B, K, e], [B, K, s]
            u, q = torch.split(uv, [self.e, self.s], dim=2)

            k = self.k_fc(k)  # -> [B, K, s]
            v = self.v_fc(v)  # -> [B, K, e]

            if self.pos_enc:
                q = rope(q, 1)
                k = rope(k, 1)

        # [B, K, s].permute() -> [B, s, K]
        # [B, K, s] x [B, s, K] -> [B, K, K]
        qk = torch.bmm(q, k.permute(0, 2, 1))

        if self.use_rel_bias:
            if self.attn_type == 'self-attn':
                bias = self.rel_pos_bias(q.size(1))
            else:
                bias = self.rel_pos_bias(q.size(1), k.size(1))
            qk += bias[:, :q.size(1), :k.size(1)]
        # [B, K, K]
        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if self.dropout_rate > 0.:
            kernel = self.dropout(kernel)
        # [B, K, K] x [B, K, e] -> [B, K, e]
        x = u * torch.bmm(kernel, v)
        # [B, K, e] -> [B, K, out_token_dims]
        x = self.o(x)

        return x

    def forward(self, x):
        """Forward function."""

        if self.shortcut:
            if self.attn_type == 'cross-attn':
                res_shortcut = x[0]
            else:
                res_shortcut = x
            main_branch = self.drop_path(self._forward(x))
            return self.res_scale(res_shortcut) + main_branch
        else:
            return self.drop_path(self._forward(x))


class Hand5Head(nn.Module):
    def __init__(self):
        super().__init__();self.final_layer=nn.Conv2d(768,21,7,padding=3)
        self.mlp=nn.Sequential(ScaleNorm(64),nn.Linear(64,256,bias=False))
        self.gau=RTMCCBlock(21,256,256,s=128,expansion_factor=2,dropout_rate=0.,drop_path=0.,attn_type='self-attn',act_fn='SiLU',use_rel_bias=False,pos_enc=False)
        self.cls_x=nn.Linear(256,512,bias=False);self.cls_y=nn.Linear(256,512,bias=False)
    def forward(self,features):
        x=self.final_layer(features[-1]).flatten(2);x=self.gau(self.mlp(x));return self.cls_x(x),self.cls_y(x)
class Hand5Pose(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=CSPNeXt(arch='P5',expand_ratio=.5,deepen_factor=.67,widen_factor=.75,out_indices=(4,),channel_attention=True,norm_cfg=dict(type='SyncBN'),act_cfg=dict(type='SiLU'),init_cfg=None)
        self.head=Hand5Head()
    def forward(self,x):return self.head(self.backbone(x))
def decode_simcc(x,y):
    sx,ix=x.max(-1);sy,iy=y.max(-1)
    return torch.stack([ix,iy],-1).float()/2,torch.minimum(sx,sy)
