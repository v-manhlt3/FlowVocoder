import torch

import utils

from torch import nn
from torch.nn import functional as F, init
from torch.nn.utils import weight_norm

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act
    return acts

@torch.jit.script
def fused_res_skip(tensor, res_skip, n_channels):
    n_channels_int = n_channels[0]
    res = res_skip[:, :n_channels_int]
    skip = res_skip[:, n_channels_int:]
    return (tensor + res), skip

class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_h=1, dilation_w=1,
                 causal=True):
        super(Conv2D, self).__init__()
        self.causal = causal
        self.dilation_h, self.dilation_w = dilation_h, dilation_w

        if self.causal:
            self.padding_h = dilation_h * (kernel_size - 1)  # causal along height
        else:
            self.padding_h = dilation_h * (kernel_size - 1) // 2
        self.padding_w = dilation_w * (kernel_size - 1) // 2  # noncausal along width
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              dilation=(dilation_h, dilation_w), padding=(self.padding_h, self.padding_w))
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding_h != 0:
            out = out[:, :, :-self.padding_h, :]
        return out

    def reverse_fast(self, tensor):
        self.conv.padding = (0, self.padding_w)
        out = self.conv(tensor)
        return out


def concat_elu(x):
    axis = len(x.size()) - 1
    return torch.nn.functional.elu(torch.cat((x, -x), dim=1))

class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = concat_elu(x)
        out = self.conv(out)
        return out

class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size,
                 cin_channels=None, local_conditioning=True, dilation_h=None, dilation_w=None,
                 causal=True):
        super(ResBlock2D, self).__init__()
        self.out_channels = out_channels
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.skip = True
        assert in_channels == out_channels == skip_channels

        self.filter_gate_conv = Conv2D(in_channels, 2*out_channels, kernel_size, dilation_h, dilation_w, causal=causal)

        self.filter_gate_conv_c = nn.Conv2d(cin_channels, 2*out_channels, kernel_size=1)
        self.filter_gate_conv_c = nn.utils.weight_norm(self.filter_gate_conv_c)
        nn.init.kaiming_normal_(self.filter_gate_conv_c.weight)

        self.res_skip_conv = nn.Conv2d(out_channels, 2*in_channels, kernel_size=1)
        self.res_skip_conv = nn.utils.weight_norm(self.res_skip_conv)
        nn.init.kaiming_normal_(self.res_skip_conv.weight)


    def forward(self, tensor, c=None):
        n_channels_tensor = torch.IntTensor([self.out_channels])

        h_filter_gate = self.filter_gate_conv(tensor)
        c_filter_gate = self.filter_gate_conv_c(c)
        out = fused_add_tanh_sigmoid_multiply(h_filter_gate, c_filter_gate, n_channels_tensor)

        res_skip = self.res_skip_conv(out)

        return fused_res_skip(tensor, res_skip, n_channels_tensor)


    def reverse(self, tensor, c=None):
        # used for reverse. c is a cached tensor
        h_filter_gate = self.filter_gate_conv(tensor)
        n_channels_tensor = torch.IntTensor([self.out_channels])
        out = fused_add_tanh_sigmoid_multiply(h_filter_gate, c, n_channels_tensor)

        res_skip = self.res_skip_conv(out)

        return fused_res_skip(tensor, res_skip, n_channels_tensor)

    def reverse_fast(self, tensor, c=None):
        h_filter_gate = self.filter_gate_conv.reverse_fast(tensor)
        n_channels_tensor = torch.IntTensor([self.out_channels])
        out = fused_add_tanh_sigmoid_multiply(h_filter_gate, c, n_channels_tensor)

        res_skip = self.res_skip_conv(out)

        return fused_res_skip(tensor[:, :, -1:, :], res_skip, n_channels_tensor)


class Wavenet2D(nn.Module):
    # a variant of WaveNet-like arch that operates on 2D feature for WF
    def __init__(self, in_channels=1, out_channels=2, num_layers=6,
                 residual_channels=256, gate_channels=256, skip_channels=256,
                 kernel_size=3, cin_channels=80, dilation_h=None, dilation_w=None,
                 causal=True):
        super(Wavenet2D, self).__init__()
        assert dilation_h is not None and dilation_w is not None

        self.residual_channels = residual_channels
        self.skip = True if skip_channels is not None else False

        self.front_conv = nn.Sequential(
            Conv2D(in_channels, residual_channels, 1, 1, 1, causal=causal),
        )

        self.res_blocks = nn.ModuleList()

        for n in range(num_layers):
            self.res_blocks.append(ResBlock2D(residual_channels, gate_channels, skip_channels, kernel_size,
                                              cin_channels=cin_channels, local_conditioning=True,
                                              dilation_h=dilation_h[n], dilation_w=dilation_w[n],
                                              causal=causal))


    def forward(self, x, c=None):
        h = self.front_conv(x)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s

        return skip

    def reverse(self, x, c=None):
        # used for reverse op. c is cached tesnor
        h = self.front_conv(x)    # [B, 64, 1, 13264]
        skip = 0
        for i, f in enumerate(self.res_blocks):
            c_i = c[i]
            h, s = f.reverse(h, c_i) # modification: conv_queue + previous layer's output concat , c_i + conv_queue update: conv_queue last element & previous layer's output concat
            skip += s
        return skip

    def reverse_fast(self, x, c=None):
        # input: [B, 64, 1, T]
        # used for reverse op. c is cached tesnor
        h = self.front_conv(x)  # [B, 64, 1, 13264]
        skip = 0
        for i, f in enumerate(self.res_blocks):
            c_i = c[i]
            h_new = torch.cat((self.conv_queues[i], h), dim=2)  # [B, 64, 3, T]
            h, s = f.reverse_fast(h_new, c_i)
            self.conv_queues[i] = h_new[:, :, 1:, :]  # cache the tensor to queue
            skip += s

        return skip

    def conv_queue_init(self, x):
        self.conv_queues = []
        B, _, _, W = x.size()
        for i in range(len(self.res_blocks)):
            conv_queue = torch.zeros((B, self.residual_channels, 2, W), device=x.device)
            if x.type() == 'torch.cuda.HalfTensor':
                conv_queue = conv_queue.half()
            self.conv_queues.append(conv_queue)


class ConvAttnBlock(nn.Module):
    def __init__(self, num_channels, drop_prob, use_attn, aux_channels):
        super(ConvAttnBlock, self).__init__()
        self.conv = GatedConv(num_channels, drop_prob, aux_channels)
        self.norm_1 = nn.LayerNorm(num_channels)


    def forward(self, x, aux=None):
        x = self.conv(x, aux) + x
        x = x.permute(0, 2, 3, 1)  # (b, h, w, c)
        x = self.norm_1(x)

        x = x.permute(0, 3, 1, 2)  # (b, c, h, w)

        return x

class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution.
    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Side length of each convolutional kernel.
        padding (int): Padding to add on edges of input.
        bias (bool): Use bias in the convolution operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        return x


class GatedConv(nn.Module):
    """Gated Convolution Block
    Originally used by PixelCNN++ (https://arxiv.org/pdf/1701.05517).
    Args:
        num_channels (int): Number of channels in hidden activations.
        drop_prob (float): Dropout probability.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, num_channels, drop_prob=0., aux_channels=None):
        super(GatedConv, self).__init__()
        self.nlin = concat_elu
        # print("gate conv")
        self.conv = WNConv2d(2 * num_channels, num_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(drop_prob)
        # print("gate gate conv")
        self.gate = WNConv2d(2 * num_channels, 2*num_channels, kernel_size=1, padding=0)
        if aux_channels is not None:
            self.aux_conv = WNConv2d(2 * aux_channels, num_channels, kernel_size=1, padding=0)
        else:
            self.aux_conv = None

    def forward(self, x, aux=None):
        x = self.nlin(x)
        x = self.conv(x)
        if aux is not None:
            aux = self.nlin(aux)
            x = x + self.aux_conv(aux)
        x = self.nlin(x)
        x = self.drop(x)
        x = self.gate(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)

        return x

class NN(nn.Module):
    """Neural network used to parametrize the transformations of an MLCoupling.
    An `NN` is a stack of blocks, where each block consists of the following
    two layers connected in a residual fashion:
      1. Conv: input -> nonlinearit -> conv3x3 -> nonlinearity -> gate
      2. Attn: input -> conv1x1 -> multihead self-attention -> gate,
    where gate refers to a 1×1 convolution that doubles the number of channels,
    followed by a gated linear unit (Dauphin et al., 2016).
    The convolutional layer is identical to the one used by PixelCNN++
    (Salimans et al., 2017), and the multi-head self attention mechanism we
    use is identical to the one in the Transformer (Vaswani et al., 2017).
    Args:
        in_channels (int): Number of channels in the input.
        num_channels (int): Number of channels in each block of the network.
        num_blocks (int): Number of blocks in the network.
        num_components (int): Number of components in the mixture.
        drop_prob (float): Dropout probability.
        use_attn (bool): Use attention in each block.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, in_channels, num_channels, num_blocks, num_components, drop_prob, use_attn=True, aux_channels=None):
        super(NN, self).__init__()
        self.k = num_components  # k = number of mixture components
        self.in_conv = WNConv2d(in_channels, num_channels, kernel_size=3, padding=1)
        # print("in channels: ", in_channels)
        # print("num channels: ", num_channels)
        self.mid_convs = nn.ModuleList([ConvAttnBlock(num_channels, drop_prob, use_attn, aux_channels)
                                        for _ in range(num_blocks)])
        self.out_conv = WNConv2d(num_channels, in_channels * (2 + 3 * self.k),
                                 kernel_size=3, padding=1)
        self.rescale = weight_norm(Rescale(in_channels))

    def forward(self, x, aux=None):
        b, c, h, w = x.size()
        x = self.in_conv(x)
        for conv in self.mid_convs:
            x = conv(x, aux)
        x = self.out_conv(x)

        # Split into components and post-process
        x = x.view(b, -1, c, h, w)
        s, t, pi, mu, scales = x.split((1, 1, self.k, self.k, self.k), dim=1)
        s = self.rescale(torch.tanh(s.squeeze(1)))
        t = t.squeeze(1)
        scales = scales.clamp(min=-7)  # From the code in original Flow++ paper

        return s, t, pi, mu, scales


# class MADE(nn.Module):
#     """Implementation of MADE.
#     It can use either feedforward blocks or residual blocks (default is residual).
#     Optionally, it can use batch norm or dropout within blocks (default is no).
#     """

#     def __init__(self,
#                  features,
#                  hidden_features,
#                  context_features=None,
#                  num_blocks=2,
#                  output_multiplier=1,
#                  use_residual_blocks=True,
#                  random_mask=False,
#                  activation=F.relu,
#                  dropout_probability=0.,
#                  use_batch_norm=False):
#         if use_residual_blocks and random_mask:
#             raise ValueError('Residual blocks can\'t be used with random masks.')
#         super().__init__()

#         # Initial layer.
#         self.initial_layer = MaskedLinear(
#             in_degrees=_get_input_degrees(features),
#             out_features=hidden_features,
#             autoregressive_features=features,
#             random_mask=random_mask,
#             is_output=False
#         )

#         if context_features is not None:
#             self.context_layer = nn.Linear(context_features, hidden_features)

#         # Residual blocks.
#         blocks = []
#         if use_residual_blocks:
#             block_constructor = MaskedResidualBlock
#         else:
#             block_constructor = MaskedFeedforwardBlock
#         prev_out_degrees = self.initial_layer.degrees
#         for _ in range(num_blocks):
#             blocks.append(
#                 block_constructor(
#                     in_degrees=prev_out_degrees,
#                     autoregressive_features=features,
#                     context_features=context_features,
#                     random_mask=random_mask,
#                     activation=activation,
#                     dropout_probability=dropout_probability,
#                     use_batch_norm=use_batch_norm,
#                 )
#             )
#             prev_out_degrees = blocks[-1].degrees
#         self.blocks = nn.ModuleList(blocks)

#         # Final layer.
#         self.final_layer = MaskedLinear(
#             in_degrees=prev_out_degrees,
#             out_features=features * output_multiplier,
#             autoregressive_features=features,
#             random_mask=random_mask,
#             is_output=True
#         )

#     def forward(self, inputs, context=None):
#         outputs = self.initial_layer(inputs)
#         if context is not None:
#             outputs += self.context_layer(context)
#         for block in self.blocks:
#             outputs = block(outputs, context)
#         outputs = self.final_layer(outputs)
#         return outputs

# from torchsummary import summary

# made = MADE(128, 256)
# print(summary(made, 128, 64, 16000/64))