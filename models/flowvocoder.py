from torch import nn
import sys
sys.path.append('/root/TTS-dir/WaveFlow/')
from modules import Wavenet2D, Conv2D, ZeroConv2d, ZeroConv2d_1, NN, Wavenet2DHyperMultGate, WNConv2d
import time

from utils import log_dist as logistic
from torch.nn.utils import weight_norm
from torch.distributions.normal import Normal 
from functions import *
from utils.log_dist import Sigmoid 





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

class WaveNet2DHyperDensityEstimator(nn.Module):
    def __init__(self, in_channel, cin_channel, hyper_channels,filter_size=256, num_layer=6, num_height=None,
                 layers_per_dilation_h_cycle=3, k=32):
        super().__init__()
        assert num_height is not None
        self.in_channel = in_channel
        self.num_height = num_height
        self.k = k
        self.layers_per_dilation_h_cycle = layers_per_dilation_h_cycle
        # dilation for width & height generation loop
        self.dilation_h = []
        self.dilation_w = []
        self.kernel_size = 3
        for i in range(num_layer):
            self.dilation_h.append(2 ** (i % self.layers_per_dilation_h_cycle))
            self.dilation_w.append(2 ** i)

        self.num_layer = num_layer
        self.filter_size = filter_size
        self.net = Wavenet2DHyperMultGate(in_channels=in_channel, out_channels=filter_size,
                             num_layers=num_layer, residual_channels=filter_size,
                             gate_channels=filter_size, skip_channels=filter_size,
                             hyper_channels=hyper_channels,
                             kernel_size=3, cin_channels=cin_channel, dilation_h=self.dilation_h,
                             dilation_w=self.dilation_w)
        # self.proj = WNConv2d(filter_size, in_channel, kernel_size=1, padding=0)


    def forward(self, x, c=None, context=None, multgate=None, debug=False):
        out = self.net(x, c, context, multgate)
        # out = self.proj(out)
        return out

    def reverse(self, x, c=None, context=None, multgate=None, debug=False):
        out = self.net.reverse(x, c, context, multgate)
        # out = self.proj(out)
        return out

    def reverse_fast(self, x, c=None, context=None, multgate=None, debug=False):
        out = self.net.reverse_fast(x, c, context, multgate)
        # out = self.proj(out)
        return out

    # def reverse_faster(self, x, c=None, multgate=None, debug=False):
    #     out = self.net.reverse_faster(x, c, multgate)
    #     out = self.proj(out)
    #     return out


class Flow(nn.Module):
    def __init__(self, in_channel, cin_channel, n_flow, filter_size, num_layer, num_height, layers_per_dilation_h_cycle, bipartize=False):
        super().__init__()

        self.k = 32
        self.layer_density_estimator = NN(in_channel, 32, num_blocks=5, num_components=32, drop_prob=0, filter_size=filter_size)
        # self.layer_density_estimator = ZeroConv2d(filter_size*2, (2+3*self.k)*in_channel)
        self.multgate = nn.Parameter(torch.ones(num_layer, filter_size))
        self.n_flow = n_flow  # useful for selecting permutation
        self.bipartize = bipartize
        self.scale_flow = Sigmoid()
        self.num_height = num_height
        self.rescale = weight_norm(Rescale(in_channel))


    def forward(self, estimator, x, c=None,i=None, embedding=None,debug=False):
        logdet = 0

        x = reverse_order(x)
        c = reverse_order(c)

        x_shift = shift_1d(x)

        b, ch, h, w = x_shift.size()

        feat = estimator(x_shift, c, embedding, self.multgate)
        a, b, pi, mu, scales = self.layer_density_estimator(feat)

        x_out = x
        x_out = logistic.mixture_log_cdf(x_out, pi, mu, scales).exp()
        x_out, scale_ldj = self.scale_flow.forward(x_out)
        out = x_out*torch.exp(a) + b

        logistic_ldj = logistic.mixture_log_pdf(x, pi, mu, scales)
        logdet = torch.flatten(logistic_ldj + a + scale_ldj).sum(-1)

        if debug:
            return out, c, logdet, log_s, t
        else:
            return out, c, logdet, None, None

    def reverse(self,estimator, z, c, i, embedding):
        x = torch.zeros_like(z[:, :, 0:1, :])

        for i_h in range(0, self.num_height):
            b, ch,h, w = x.size()
            # print("flow: {}, x shape: {}------ c shape: {}".format(i_h, x.shape, c[:, :, :, :i_h+1, :].shape))
            feat = estimator.reverse(x, c[:, :, :, :i_h+1, :], embedding[:, :, :, :i_h + 1, :]
                    , self.multgate)[:, :, -1, :].unsqueeze(2)

            """ compute inver data-parameterized family"""
            a, b, pi, mu, scales = self.layer_density_estimator(feat, reverse=True)
            end = time.time()

            x_new = (z[:, :, i_h, :].unsqueeze(2) - b)*torch.exp(-a)
            x_new, scale_ldj = self.scale_flow.inverse(x_new)
            x_new = x_new.clamp(1e-5, 1.0 - 1e-5)
            x_new = logistic.mixture_inv_cdf(x_new, pi, mu, scales)
            x = torch.cat((x, x_new), 2)

        x = x[:,:,1:,:]
        x = reverse_order(x)
        c = reverse_order(c, dim=3)

        return x, c

    def reverse_fast(self,estimator, z, c, embedding):
        x = torch.zeros_like(z[:, :, 0:1, :])
        estimator.net.conv_queue_init(x)

        for i_h in range(0, self.num_height):
            b, ch,h, w = x.size()
            # print("flow: {}, x shape: {}------ c shape: {}".format(i_h, x.shape, c[:, :, :, :i_h+1, :].shape))
            feat = estimator.reverse_fast(x if i_h == 0 else x_new, c[:, :, :, i_h:i_h+1, :], embedding[:, :, :, :, :]
                    , self.multgate)[:, :, -1, :].unsqueeze(2)

            """ compute inver data-parameterized family"""
            a, b, pi, mu, scales = self.layer_density_estimator(feat, reverse=True)
            end = time.time()

            x_new = (z[:, :, i_h, :].unsqueeze(2) - b)*torch.exp(-a)
            x_new, scale_ldj = self.scale_flow.inverse(x_new)
            x_new = x_new.clamp(1e-5, 1.0 - 1e-5)
            x_new = logistic.mixture_inv_cdf(x_new, pi, mu, scales)
            x = torch.cat((x, x_new), 2)

        x = x[:,:,1:,:]
        x = reverse_order(x)
        c = reverse_order(c, dim=3)

        return x, c


class WaveFlow(nn.Module):
    def __init__(self, in_channel, cin_channel, res_channel, n_height, n_flow, n_layer, 
                layers_per_dilation_h_cycle, size_flow_embed,
                bipartize=False):
        super().__init__()
        self.in_channel = in_channel
        self.cin_channel = cin_channel
        self.res_channel = res_channel
        self.n_height = n_height
        self.n_flow = n_flow
        self.n_layer = n_layer

        self.layers_per_dilation_h_cycle = layers_per_dilation_h_cycle
        self.bipartize = bipartize
        if self.bipartize:
            print("INFO: bipartization version for permutation is on for reverse_order. Half the number of flows will use bipartition & reverse over height.")

        ################ create embedding for flow layers #######################################
        self.size_flow_embed = size_flow_embed
        self.flow_embedding = nn.Embedding(self.n_flow, self.size_flow_embed)

        # self.use_weightnorm_embed = use_weightnorm_embed
        # if self.use_weightnorm_embed:
        #     print("INFO: using weightnorm at embedding layer")
        #     self.flow_embedding = nn.utils.weight_norm(self.flow_embedding)
        nn.init.orthogonal_(self.flow_embedding.weight)
        #########################################################################################

        self.estimator = WaveNet2DHyperDensityEstimator(self.in_channel, self.cin_channel, self.size_flow_embed,
                                                        self.res_channel, self.n_layer, self.n_height,
                                                        self.layers_per_dilation_h_cycle)

        self.flows = nn.ModuleList()
        for i in range(self.n_flow):
            self.flows.append(Flow(self.in_channel, self.cin_channel, n_flow=self.n_flow, filter_size=self.res_channel,
                                   num_layer=self.n_layer, num_height=self.n_height,
                                   layers_per_dilation_h_cycle=self.layers_per_dilation_h_cycle,
                                   bipartize=self.bipartize))

        self.upsample_conv = nn.ModuleList()
        for s in [16, 16]:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

        self.upsample_conv_kernel_size = (2*s)**2
        self.upsample_conv_stride = s**2

    def cache_flow_embed(self, remove_after_cache=False):
        # cache flow embedding after training, since it is not dependent on inputs
        # used for reverse operation
        # generate flow indicator token, flipped
        flow_token = torch.arange((self.n_flow)).flip(dims=(0,)).cuda()
        flow_embed = self.flow_embedding(flow_token)
        flow_embed = flow_embed.unsqueeze(-1).unsqueeze(-1)

        h_cache = []
        for i, resblock in enumerate(self.estimator.net.res_blocks):
            filter_gate_conv_h = resblock.filter_gate_conv_h(flow_embed)
            h_cache.append(filter_gate_conv_h)
            if remove_after_cache:
                del resblock.filter_gate_conv_h
        h_cache = torch.stack(h_cache)
        h_cache = h_cache.permute(1, 0, 2, 3, 4)

        if remove_after_cache:
            print("INFO: filter_gate_conv_h removed after global embedding caching. only reverse_fast can be used!")
        return torch.nn.Parameter(h_cache)

    def forward(self, x, c, debug=False):
        x = x.unsqueeze(1)
        B, h, T = x.size()
        #  Upsample spectrogram to size of audio
        c = self.upsample(c)
        assert(c.size(2) >= x.size(2))
        if c.size(2) > x.size(2):
            c = c[:, :, :x.size(2)]

        x, c = squeeze_to_2d(x, c, h=self.n_height)
        out = x

        flow_token = torch.arange((self.n_flow)).to(c.device)
        flow_token = torch.repeat_interleave(flow_token.unsqueeze(0), c.size()[0], dim=0)

        logdet = 0
        list_logdet = []

        if debug:
            list_log_s, list_t  = [], []
        # print("number of flow block: ", self.n_flow)

        for i, flow in enumerate(self.flows):
            i_flow = i
            flow_embed = self.flow_embedding(flow_token[:, i])
            flow_embed = flow_embed.unsqueeze(-1).unsqueeze(-1) # match shape
            out, c, logdet_new, log_s, t = flow(self.estimator, out, c, i_flow,flow_embed, debug)
            # print("logdet_new: ", logdet_new.shape)
            list_logdet.append(logdet_new.sum().divide(B*h*T))
            if debug:
                list_log_s.append(log_s)
                list_t.append(t)
            logdet = logdet + logdet_new

        if debug:
            return out, logdet, list_log_s, list_t
        else:
            list_logdet = torch.tensor(list_logdet, requires_grad=True).to(x.device)
            return out, logdet, list_logdet

    def reverse(self, c, temp=1.0, debug_z=None):
        # plain implementation of reverse ops
        # print("Size of c: ", c.shape)

        c = self.upsample(c)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample_conv_kernel_size - self.upsample_conv_stride
        c = c[:, :, :-time_cutoff]

        B, _, T_c = c.size()

        _, c = squeeze_to_2d(None, c, h=self.n_height)

        if debug_z is None:
            # sample gaussian noise that matches c
            q_0 = Normal(c.new_zeros((B, 1, c.size()[2], c.size()[3])), c.new_ones((B, 1, c.size()[2], c.size()[3])))
            z = q_0.sample() * temp
        else:
            z = debug_z
        # print("number of flow model: ", 8)
        c_cache = []
        for i, resblock in enumerate(self.estimator.net.res_blocks):
            filter_gate_conv_c = resblock.filter_gate_conv_c(c)
            c_cache.append(filter_gate_conv_c)
        c_cache = torch.stack(c_cache)

        for i, flow in enumerate(self.flows[::-1]):
            flow_embed = self.h_cache[i].unsqueeze(1)
            i_flow = self.n_flow - (i+1)
            # print("reverse step: ", i)
            z, c = flow.reverse(self.estimator, z, c_cache, i_flow, flow_embed)

        x = unsqueeze_to_1d(z, self.n_height)

        return x

    def reverse_fast(self, c, temp=1.0, debug_z=None):
        c = self.upsample(c)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample_conv_kernel_size - self.upsample_conv_stride
        c = c[:, :, :-time_cutoff]

        B, _, T_c = c.size()

        _, c = squeeze_to_2d(None, c, h=self.n_height)

        if debug_z is None:
            # sample gaussian noise that matches c
            q_0 = Normal(c.new_zeros((B, 1, c.size()[2], c.size()[3])), c.new_ones((B, 1, c.size()[2], c.size()[3])))
            z = q_0.sample() * temp
        else:
            z = debug_z

        # pre-compute conditioning tensors and cache them
        c_cache =self.estimator.net.fused_filter_gate_conv_c(c)
        c_cache = c_cache.reshape(c_cache.shape[0], self.n_layer, self.res_channel*2, c_cache.shape[2], c_cache.shape[3])
        c_cache = c_cache.permute(1, 0, 2, 3, 4) # [num_layers, batch_size, res_channels, height, width]
        c_cache_reversed = reverse_order(c_cache, dim=3)

        for i, flow in enumerate(self.flows[::-1]):
            flow_embed = self.h_cache[i].unsqueeze(1)  # unsqueeze batch dim
            if z.type() == 'torch.cuda.HalfTensor':
                flow_embed = flow_embed.half()
            c_cache_i = c_cache if i % 2 == 0 else c_cache_reversed

            z, _ = flow.reverse_fast(self.estimator, z, c_cache_i, flow_embed)
            # c_cache_i = c_cache_i + flow_embed
            # z, _ = flow.reverse_faster(self.estimator, z, c_cache_i)

        x = unsqueeze_to_1d(z, self.n_height)
        return x

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c

    def remove_weight_norm(self):
        # remove weight norm from all weights
        for layer in self.upsample_conv.children():
            try:
                torch.nn.utils.remove_weight_norm(layer)
            except ValueError:
                pass

        net = self.estimator.net
        try:
            torch.nn.utils.remove_weight_norm(net.front_conv[0].conv)
        except ValueError:
            for i in range(len(net.front_conv[0].conv)):
                torch.nn.utils.remove_weight_norm(net.front_conv[0].conv[i])
        for resblock in net.res_blocks.children():
            try:
                torch.nn.utils.remove_weight_norm(resblock.filter_gate_conv.conv)
            except ValueError:
                for i in range(len(resblock.filter_gate_conv.conv)):
                    torch.nn.utils.remove_weight_norm(resblock.filter_gate_conv.conv[i])
            torch.nn.utils.remove_weight_norm(resblock.filter_gate_conv_c)
            if hasattr(resblock, "filter_gate_conv_h"):
                torch.nn.utils.remove_weight_norm(resblock.filter_gate_conv_h)
            torch.nn.utils.remove_weight_norm(resblock.res_skip_conv)
        try:
            torch.nn.utils.remove_weight_norm(self.flow_embedding)
        except ValueError:
            pass

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("weight_norm removed: {} params".format(total_params))


    def fuse_conditioning_layers(self):
        # fuse mel-spec conditioning layers into one big conv weight
        net = self.estimator.net
        cin_channels = net.res_blocks[0].cin_channels
        out_channels = net.res_blocks[0].out_channels
        fused_filter_gate_conv_c = nn.Conv2d(cin_channels, 2*out_channels*self.n_layer, kernel_size=1)
        fused_filter_gate_conv_c_weight = []
        fused_filter_gate_conv_c_bias = []
        for resblock in net.res_blocks.children():
            fused_filter_gate_conv_c_weight.append(resblock.filter_gate_conv_c.weight)
            fused_filter_gate_conv_c_bias.append(resblock.filter_gate_conv_c.bias)
            del resblock.filter_gate_conv_c

        fused_filter_gate_conv_c.weight = torch.nn.Parameter(torch.cat(fused_filter_gate_conv_c_weight).clone())
        fused_filter_gate_conv_c.bias = torch.nn.Parameter(torch.cat(fused_filter_gate_conv_c_bias).clone())
        self.estimator.net.fused_filter_gate_conv_c = fused_filter_gate_conv_c
        del self.flow_embedding

        print("INFO: conditioning layers fused for performance: only reverse_fast function can be used for inference!")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("model after optimization: {} params".format(total_params))


# unit test
if __name__ == "__main__":
    x = torch.randn((2, 15872)).cuda()
    c = torch.randn((2, 80, 62)).cuda()
    net = WaveFlow(1, 80, 64, 8, 4, 8, 1, 64,bipartize=False).cuda()
    out = net(x, c)

    net.h_cache = net.cache_flow_embed()
    with torch.no_grad():
        out = net.reverse(c)
        net.fuse_conditioning_layers()
        # out_fast = net.reverse_fast(c)


