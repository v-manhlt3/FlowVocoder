import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import spectral_norm
from modules import Conv1d, Linear

DELAY_THRESHOLD = 10

def reparameterize_trick(z, device=torch.device("cuda")):
    epsilon = torch.distributions.uniform.Uniform(0, 1).sample()
    epsilon = epsilon.to(device)

    # result = torch.log(epsilon) - torch.log(1 - epsilon) + torch.log(z) - torch.log(1 - z)
    result = torch.log(epsilon) - torch.log(1 - epsilon) + torch.log(z)
    # print("epsilon: ", epsilon)
    # print("before sigmoid: ", result)
    return torch.sigmoid(result) 


class Encoder(nn.Module):
    def __init__(self,
                encoder_dim,
                spk_embedding,
                z_dim):

        super(Encoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.spk_embedding_dim = spk_embedding
        self.z_dim = z_dim

        self.aligner = Aligner(encoder_dim, z_dim, spk_embedding)

    def forward(self, encoder_input, z, spk_embedding):

        encoder_output, duration = self.aligner(encoder_input, z, spk_embedding)

        return encoder_output, duration

class BatchNorm1dLayer(nn.Module):
    def __init__(self,
                num_features,
                s_channels=128,
                z_channels=128):
        super().__init__()

        self.num_features = num_features
        self.s_channels = s_channels
        self.z_channels = z_channels
        self.batch_norm = nn.BatchNorm1d(num_features, affine=False)

        self.scale_layer = spectral_norm(nn.Linear(z_channels, num_features))
        self.scale_layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.scale_layer.bias.data.zero_()        # Initialise bias at 0

        self.shift_layer = spectral_norm(nn.Linear(s_channels, num_features))
        self.shift_layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.shift_layer.bias.data.zero_()        # Initialise bias at 0

    def forward(self, inputs, z, s):
        outputs = self.batch_norm(inputs)
        scale = self.scale_layer(z)
        scale = scale.view(-1, self.num_features, 1)

        shift = self.shift_layer(s)
        shift = shift.view(-1, self.num_features, 1)

        outputs = outputs*scale + shift
        return outputs

class DilatedConvBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                z_channels,
                s_channels,
                dilation):
        super(DilatedConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.s_channels = s_channels

        self.conv1d = Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.batch_layer = BatchNorm1dLayer(out_channels, s_channels, z_channels)

    def forward(self, inputs, z, s):
        outputs = self.conv1d(inputs)
        outputs = self.batch_layer(outputs, z, s)
        return F.relu(outputs)

####################################################################################################
class EnergyLayer(nn.Module):
    """
        https://arxiv.org/pdf/1704.00784.pdf
        Online and Linear-Time Attention by Enforcing Monotonic Alignments. Section 2.4
    """
    def __init__(self,
                input_size=256):
        super().__init__()
        self.linear_s = nn.Linear(input_size, input_size, bias=False)
        self.linear_h = nn.Linear(input_size, input_size, bias=False)
        self.bias_b = torch.zeros(input_size, requires_grad=True)
        self.bias_r = torch.zeros(1, requires_grad=False)
        self.energy_gain = torch.tensor(1.0 / input_size)
        self.linear_v = nn.Linear(input_size, input_size, bias=False)
        self.activation = nn.Tanh()

        torch.nn.init.uniform_(self.linear_s.weight)
        torch.nn.init.uniform_(self.linear_h.weight)
        torch.nn.init.normal_(self.linear_v.weight, std = 1.0/input_size)

    def forward(self, s, h):
        linear_s = self.linear_s(s)
        linear_h = self.linear_h(h)
        output_tanh = self.activation(linear_s + linear_h + self.bias_b)

        normed_v = torch.norm(self.linear_v.weight)
        # a(s_i−1, h_j) = (g* / ||v||)*V^T*tanh(W*s_i−1 +V*h_j + b) + r
        result = (self.energy_gain*torch.sum(self.linear_v(output_tanh))) / normed_v + self.bias_r
        return result
######################################################################################################



class Aligner(nn.Module):
    def __init__(self,
                in_channels, 
                z_channels,
                s_channels,
                num_dilation_layer=10):
        super(Aligner, self).__init__ ()

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.s_channels = s_channels

        self.pre_process = Conv1d(in_channels, 256, kernel_size=3)
        self.dilated_conv_layers = nn.ModuleList()
        self.energy_layer = EnergyLayer(256)

        for i in range(num_dilation_layer):
            dilation = 2**i 
            self.dilated_conv_layers.append(DilatedConvBlock(256, 256, z_channels, s_channels, dilation))

        # self.duration_prediction = nn.Sequential(
        #     # Conv1d(in_channels, 256, kernel_size=3),
        #     Linear(256, 256),
        #     nn.ReLU(inplace=False),
        #     Linear(256, 1),
        #     nn.ReLU(inplace=False),
        # )
        """ state layer """
        self.state_layer = nn.Sequential(
            # Conv1d(in_channels, 256, kernel_size=3),
            nn.LSTM(256, 256, 1, batch_first=True)
        )
    
    def forward(self, inputs, z, s):
        print("inputs shape = ", inputs.shape)
        outputs = self.pre_process(inputs)

        for layer in self.dilated_conv_layers:
            outputs = layer(outputs, z, s)
        
        encoder_outputs = outputs.transpose(1, 2)
        print("Hidden shape = ", outputs.shape)
        # duration = self.duration_prediction(outputs.transpose(1, 2))
        duration,_ = self.state_layer(outputs.transpose(1, 2))

        # return encoder_outputs, duration.squeeze(-1)
        return encoder_outputs, duration

    

    def compute_hard_alignment(self, h, s):
        # h = torch.transpose(h, -1, -2)
        h = h.squeeze(0)
        # s = torch.transpose(s, -1, -2)
        s = h.squeeze(0)
        print("hidden shape: ", h.shape)
        alignment = torch.ones(h.size(0))

        for i in range(h.size(0)):
            delay = 0
            finised = False
            while finised and delay < DELAY_THRESHOLD:
                if i == 0:
                    s_i_1 = torch.zeros(256)
                else:
                    s_i_1 = s[i-1]
                energy = self.energy_layer(s_i_1, h[i])
                p_i_j = torch.sigmoid(energy)
                z_i_j = self.reparameterize_trick(p_i_j, device=torch.device("cpu"))
                if z_i_j >= 0.5:
                    finised = False
                    if i == h.size(0):
                        break
                else:
                    alignment[i] += 1
                    delay += 1 
        return alignment

    def compute_align_loss(self, alignment, h, s):
        s = alignment
        loss_length = F.l1_loss(torch.sum(s), torch.tensor(100))
        print("loss length: ", loss_length)

        likelihood_loss = torch.sum(torch.log(h)) + torch.sum(torch.log(s))
        loss = loss_length + likelihood_loss
        return loss

    def compute_mel_loss(self, mel, mel_hat):
        return F.l1_loss(mel, mel_hat)

    def compute_length_loss(self, gt_len, pd_len):
        return F.l1_loss(gt_len, pd_len)
    
    # def compute_OT_loss(self, source, target):


if __name__ == "__main__":

    
    model = Encoder(256, 64, 64)
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(10):
        encoder_inputs = torch.randn(1, 256, 10)
        z = torch.randn(1, 64)
        speaker = torch.randn(1, 64)
        output, duration = model(encoder_inputs, z, speaker)
        alignment = model.aligner.compute_hard_alignment(output, duration)

        output = output.squeeze(0)
        output = torch.sum(output, dim=-1)
        outpput = torch.sigmoid(output)
        # print("Proper output: ", outpput)
        
        loss = model.aligner.compute_align_loss(alignment, output, duration)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(output.shape, duration.shape)
        # print(alignment)
        print("lossss------------:", loss)

    # z = torch.randn(10)
    # print(z)
    # for i in range(z.size(0)):

    #     sample = reparameterize_trick(torch.sigmoid(z[i]), device=torch.device('cpu'))
    #     print(sample)

    # layer = EnergyLayer(256)
    # s = torch.randn(1, 256)
    # h = torch.randn(1, 256)
    # result = layer(s, h)
    # print(result)