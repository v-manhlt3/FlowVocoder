import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import spectral_norm
from commons import Conv1d, Linear
import math
import commons

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
                n_vocab,
                spk_embedding,
                z_dim,
                encoder_dim,
                encoder_out_dim=256,
                out_dim=80):

        super(Encoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.spk_embedding_dim = spk_embedding
        self.z_dim = z_dim

        self.aligner = Aligner(encoder_dim, z_dim, spk_embedding, encoder_out_dim)
        self.emb = nn.Embedding(n_vocab, encoder_dim)
        nn.init.normal_(self.emb.weight, 0.0, encoder_dim**-0.5)

        self.proj_x_m = Conv1d(encoder_out_dim, out_dim, 1)
        # self.proj_x_logs = Conv1d(encoder_out_dim, out_dim, 1)

    # def forward(self, inputs, inputs_length, z, spk_embedding,g=False):
    def forward(self, inputs, inputs_length, g=False):
        z = torch.normal(0,1, (inputs.size(0), 64)).to(inputs.device)
        spk_embedding = torch.zeros(inputs.size(0), 64).to(inputs.device)
        spk_embedding[:, 1] = 1

        # encoder_inputs = self.emb(inputs) * math.sqrt(self.encoder_dim) # [b, t, h]
        encoder_inputs = self.emb(inputs) # [b, t, h, embedding]
        encoder_inputs = encoder_inputs.squeeze(1)
        encoder_inputs = torch.transpose(encoder_inputs, 1, -1) # [b, h, t]
        # print("embedding inputs: ", encoder_inputs.shape)
        x_mask = torch.unsqueeze(commons.sequence_mask(inputs_length, encoder_inputs.size(2)), 1).to(inputs.dtype)
        encoder_outputs, duration = self.aligner(encoder_inputs, z, spk_embedding)

        # print("--------encoder output: ",encoder_outputs.shape)
        x_m = self.proj_x_m(encoder_outputs)

        # x_log_s = self.proj_x_logs(encoder_outputs)
        x_log_s = torch.zeros_like(x_m).to(inputs.device)

        return x_m, x_log_s, duration, x_mask

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
#####################################################################################################

class Aligner(nn.Module):
    def __init__(self,
                in_channels, 
                z_channels,
                s_channels,
                out_dim=256,
                num_dilation_layer=10):
        super(Aligner, self).__init__ ()

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.s_channels = s_channels

        self.pre_process = Conv1d(in_channels, out_dim, kernel_size=3)
        self.dilated_conv_layers = nn.ModuleList()
        self.energy_layer = EnergyLayer(256)

        for i in range(num_dilation_layer):
            dilation = 2**i 
            self.dilated_conv_layers.append(DilatedConvBlock(out_dim, out_dim, z_channels, s_channels, dilation))

        self.duration_prediction = nn.Sequential(
            # Conv1d(in_channels, 256, kernel_size=3),
            Linear(256, 256),
            nn.ReLU(inplace=False),
            Linear(256, 1),
            nn.ReLU(inplace=False),
        )
        """ state layer """
        # self.state_layer = nn.LSTM(256, 256, 1, batch_first=True)
        # self.predict_layer = nn.Sequential(
        #     nn.Linear(256, 1),
        #     nn.ReLU()
        # )
    
    def forward(self, inputs, z, s):
        # print("inputs shape = ", inputs.shape)
        outputs = self.pre_process(inputs)

        for layer in self.dilated_conv_layers:
            outputs = layer(outputs, z, s)
        
        # encoder_outputs = outputs.transpose(1, 2)
        # print("Hidden shape = ", outputs.shape)
        duration = self.duration_prediction(outputs.transpose(1, 2))

        return outputs, duration

    def compute_hard_alignment_training(self, h, s, gt_length=100):
        h = h.squeeze(0)
        s = h.squeeze(0)

        alignment = torch.zeros(h.size(0), requires_grad=False)
        off_set = 0
        finished = False
        warp_window = 20
        alignment_length = torch.zeros(1, requires_grad=True)

        for i in range(h.size(0)):
            delay = 0
            j = off_set

            for j in range(j, gt_length - j - warp_window):
                if i == 0:
                    s_i_1 = torch.zeros(256)
                else:
                    s_i_1 = s[i-1]
                energy = self.energy_layer(s_i_1, h[i])
                p_i_j = torch.sigmoid(energy)
                z_i_j = reparameterize_trick(p_i_j, device=torch.device("cpu"))
                if z_i_j >= 0.5:
                    off_set = off_set + delay
                    alignment[i] = delay
                    z_i_j = 0
                    alignment_length = alignment_length + z_i_j
                    break
                else:
                    delay += 1 
                    z_i_j = 1
                    alignment_length = alignment_length + z_i_j 
            
        return alignment, alignment_length

    def compute_hard_alignment_inference(self, h, s):
        h = h.squeeze(0)
        s = h.squeeze(0)
        alignment = torch.ones(h.size(0), requires_grad=False)
        pseudo_output = []

        for i in range(h.size(0)):
            delay = 0
            finised = True
            while finised and delay < DELAY_THRESHOLD:
                if i == 0:
                    s_i_1 = torch.zeros(256)
                else:
                    s_i_1 = s[i-1]
                energy = self.energy_layer(s_i_1, h[i])
                p_i_j = torch.sigmoid(energy)
                z_i_j = reparameterize_trick(p_i_j, device=torch.device("cpu"))

                if z_i_j >= 0.5:
                    finised = False
                    if i == h.size(0):
                        break
                else:
                    alignment[i] += 1
                    delay += 1 

        return alignment

    def compute_align_loss(self, alignment_length, h, gt_length=100.0):

        loss_length = F.l1_loss(alignment_length, torch.tensor(gt_length))

        likelihood_loss = torch.sum(torch.log(h)) + torch.sum(torch.log(alignment))
        return loss_length

    def compute_mel_loss(self, mel, mel_hat):
        return F.l1_loss(mel, mel_hat)

    def compute_length_loss(self, gt_len, pd_len):
        return F.l1_loss(gt_len, pd_len)
    

#############################################################################################
if __name__ == "__main__":

    model = Encoder(100, 64, 64, 100)
    z = torch.randn(10, 64)
    speaker = torch.randn(10, 64)
    encoder_inputs = torch.randint(1, 100, (10, 1, 13))
    print("input data shape: ", encoder_inputs.shape)
    x_m, x_logs, duration, x_mask = model(encoder_inputs, torch.tensor([10]), z, speaker)
    print("x_m shape: ", x_m.shape)
    print("x_log_s shape: ", x_logs.shape)
    print("duration shape: ", duration.shape)
    print("x_mask shape: ", x_mask.shape)
    print(x_mask)
    # print(duration[0])
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # iter_report = 50
    # esitmate_loss = 0.0
    # for i in range(50000):
        
    #     encoder_inputs = torch.randn(1, 256, 10)
    #     z = torch.randn(1, 64)
    #     speaker = torch.randn(1, 64)
    #     output, state, duration = model(encoder_inputs, z, speaker)
    #     alignment, alignment_length = model.aligner.compute_hard_alignment_training(output, state)

    #     output = output.squeeze(0)
    #     output = torch.sum(output, dim=-1)
    #     outpput = torch.sigmoid(output)
        
    #     loss = model.aligner.compute_align_loss(alignment_length, output)
    #     esitmate_loss += loss
    #     # print("lossss------------:", loss)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if i % iter_report == 0:
    #         print("estimate length mismatch: ", esitmate_loss/iter_report)
    #         esitmate_loss = 0.0
        # print(output.shape, duration.shape)
        # print(alignment)
        

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