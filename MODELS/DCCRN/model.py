import torch
import torch.nn as nn
import torchaudio_contrib as audio_nn

def istft(stft_matrix, 
          hop_length=None, 
          win_length=None, 
          window='hann',
          center=True, 
          normalized=False, 
          onesided=True, 
          length=None):
    
    n_fft = 2 * (stft_matrix.shape[-3] - 1)
    
    return torch.istft(stft_matrix, 
                       n_fft=n_fft, 
                       hop_length=hop_length, 
                       win_length=win_length, 
                       center=center, 
                       normalized=normalized, 
                       onesided=onesided, 
                       length=length, 
                       return_complex=False)

class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output

class ComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device, num_layers=1, bias=True, dropout=0, bidirectional=False):
        super().__init__()
        self.num_layer = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.lstm_re = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                               dropout=dropout, bidirectional=bidirectional)
        self.lstm_im = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                               dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        real_real, (_, _) = self.lstm_re(x[..., 0])
        imag_imag, (_, _) = self.lstm_im(x[..., 1])
        real = real_real - imag_imag
        imag_real, (_, _) = self.lstm_re(x[..., 1])
        real_imag, (_, _) = self.lstm_im(x[..., 0])
        imaginary = imag_real + real_imag
        output = torch.stack((real, imaginary), dim=-1)
        return output

class ComplexDense(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear_read = nn.Linear(in_channel, out_channel)
        self.linear_imag = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        real = x[..., 0]
        imag = x[..., 1]
        real = self.linear_read(real)
        imag = self.linear_imag(imag)
        out = torch.stack((real, imag), dim=-1)
        return out

class ComplexBatchNormal(nn.Module):
    def __init__(self, C, H, W, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.gamma_rr = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.gamma_ri = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.gamma_ii = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.epsilon = 1e-5
        self.running_mean_real = None
        self.running_mean_imag = None
        self.Vrr = None
        self.Vri = None
        self.Vii = None

    def forward(self, x, train=True):
        B, C, H, W, D = x.size()
        real = x[..., 0]
        imaginary = x[..., 1]
        if train:
            mu_real = torch.mean(real, dim=0)
            mu_imag = torch.mean(imaginary, dim=0)

            broadcast_mu_real = mu_real.repeat(B, 1, 1, 1)
            broadcast_mu_imag = mu_imag.repeat(B, 1, 1, 1)

            real_centred = real - broadcast_mu_real
            imag_centred = imaginary - broadcast_mu_imag

            Vrr = torch.mean(real_centred * real_centred, 0) + self.epsilon
            Vii = torch.mean(imag_centred * imag_centred, 0) + self.epsilon
            Vri = torch.mean(real_centred * imag_centred, 0)
            if self.Vrr is None:
                self.running_mean_real = mu_real
                self.running_mean_imag = mu_imag
                self.Vrr = Vrr
                self.Vri = Vri
                self.Vii = Vii
            else:
                # momentum
                self.running_mean_real = self.momentum * self.running_mean_real + (1 - self.momentum) * mu_real
                self.running_mean_imag = self.momentum * self.running_mean_imag + (1 - self.momentum) * mu_imag
                self.Vrr = self.momentum * self.Vrr + (1 - self.momentum) * Vrr
                self.Vri = self.momentum * self.Vri + (1 - self.momentum) * Vri
                self.Vii = self.momentum * self.Vii + (1 - self.momentum) * Vii
            return self.cbn(real_centred, imag_centred, Vrr, Vii, Vri, B)
        else:
            broadcast_mu_real = self.running_mean_real.repeat(B, 1, 1, 1)
            broadcast_mu_imag = self.running_mean_imag.repeat(B, 1, 1, 1)
            real_centred = real - broadcast_mu_real
            imag_centred = imaginary - broadcast_mu_imag
            return self.cbn(real_centred, imag_centred, self.Vrr, self.Vii, self.Vri, B)

    def cbn(self, real_centred, imag_centred, Vrr, Vii, Vri, B):
        tau = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2)
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)
        inverse_st = 1.0 / (s * t)

        Wrr = ((Vii + s) * inverse_st).repeat(B, 1, 1, 1)
        Wii = ((Vrr + s) * inverse_st).repeat(B, 1, 1, 1)
        Wri = (-Vri * inverse_st).repeat(B, 1, 1, 1)

        n_real = Wrr * real_centred + Wri * imag_centred
        n_imag = Wii * imag_centred + Wri * real_centred

        broadcast_gamma_rr = self.gamma_rr.repeat(B, 1, 1, 1)
        broadcast_gamma_ri = self.gamma_ri.repeat(B, 1, 1, 1)
        broadcast_gamma_ii = self.gamma_ii.repeat(B, 1, 1, 1)
        broadcast_beta = self.beta.repeat(B, 1, 1, 1)

        bn_real = broadcast_gamma_rr * n_real + broadcast_gamma_ri * n_imag + broadcast_beta
        bn_imag = broadcast_gamma_ri * n_real + broadcast_gamma_ii * n_imag + broadcast_beta
        return torch.stack((bn_real, bn_imag), dim=-1)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True):
        super().__init__()

        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation)

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output

class STFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.win_length = win_length

    def forward(self, signal):
        with torch.no_grad():
            x = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            mag, phase = audio_nn.magphase(x, power=1.)

        mix = torch.stack((mag, phase), dim=-1)
        return mix.unsqueeze(1)

class ISTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length

    def forward(self, x):
        B, C, F, T, D = x.shape
        x = x.view(B, F, T, D)
        x_istft = istft(x, hop_length=self.hop_length, length=600)
        return x_istft.view(B, C, -1)

class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None):
        super().__init__()
        if padding is None:
            padding = [int((i - 1) / 2) for i in kernel_size]  # same
            # padding
        self.conv = ComplexConv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()

    def forward(self, x, train):
        x = self.conv(x)

        x = self.bn(x, train)
        x = self.prelu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None):
        super().__init__()
        self.transconv = ComplexConvTranspose2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()

    def forward(self, x, train=True):
        x = self.transconv(x)
        x = self.bn(x, train)
        x = self.prelu(x)
        return x

class DCCRN(nn.Module):
    def __init__(self, net_params, device, batch_size=36):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.encoders = []
        self.lstms = []
        self.dense = ComplexDense(net_params["dense"][0], net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index]
            )
            self.add_module("encoder{%d}" % index, model)
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_dims[index + 1],
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)
            self.add_module("lstm{%d}" % index, model)
        # init decoder
        de_channels = net_params["decoder_channels"]
        de_ker_size = net_params["decoder_kernel_sizes"]
        de_strides = net_params["decoder_strides"]
        de_padding = net_params["decoder_paddings"]
        for index in range(len(de_channels) - 1):
            model = Decoder(
                in_channel=de_channels[index] + en_channels[len(self.encoders) - index],
                out_channel=de_channels[index + 1],
                kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                chw=decoder_chw[index]
            )
            self.add_module("decoder{%d}" % index, model)
            self.decoders.append(model)

        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)
        self.decoders = nn.ModuleList(self.decoders)
        self.linear = ComplexConv2d(in_channel=2, out_channel=1, kernel_size=1, stride=1)

    def forward(self, x, train=True):
        skiper = []
        for index, encoder in enumerate(self.encoders):
            skiper.append(x)
            x = encoder(x, train)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D)
        lstm_ = lstm_.permute(2, 0, 1, 3)
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_)
        lstm_ = lstm_.permute(1, 0, 2, 3)
        lstm_out = lstm_.reshape(B * T, -1, D)
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4)
        for index, decoder in enumerate(self.decoders):
            p = decoder(p, train)
            p = torch.cat([p, skiper[len(skiper) - index - 1]], dim=1)
        mask = torch.tanh(self.linear(p))
        return mask

class DCCRN_(nn.Module):
    def __init__(self, n_fft, hop_len, net_params, batch_size, device, win_length):
        super().__init__()
        self.stft = STFT(n_fft, hop_len, win_length=win_length)
        self.DCCRN = DCCRN(net_params, device=device, batch_size=batch_size)
        self.istft = ISTFT(n_fft, hop_len, win_length=win_length)

    def forward(self, signal, train=True):
        stft = self.stft(signal)
        mask_predict = self.DCCRN(stft, train=train)
        predict = stft * mask_predict
        clean = self.istft(predict)
        return clean

def get_net_params():
    params = {}
    encoder_dim_start = 32
    params["encoder_channels"] = [
        1,
        encoder_dim_start,
        encoder_dim_start * 2,
        encoder_dim_start * 4,
        encoder_dim_start * 8,
        encoder_dim_start * 8,
        encoder_dim_start * 8
    ]
    params["encoder_kernel_sizes"] = [
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2)
    ]
    params["encoder_strides"] = [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1)
    ]
    params["encoder_paddings"] = [
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0)
    ]
    
    # ----------lstm---------
    params["lstm_dim"] = [
        1280, 128
    ]
    params["dense"] = [
        128, 1280
    ]
    params["lstm_layer_num"] = 2
    
    # --------decoder--------
    params["decoder_channels"] = [
        0,
        encoder_dim_start * 8,
        encoder_dim_start * 8,
        encoder_dim_start * 4,
        encoder_dim_start * 2,
        encoder_dim_start,
        1
    ]
    params["decoder_kernel_sizes"] = [
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2)
    ]
    params["decoder_strides"] = [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1)
    ]
    params["decoder_paddings"] = [
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0),
        (2, 0)
    ]
    params["encoder_chw"] = [
        (32, 129, 6),
        (64, 65, 5),
        (128, 33, 4),
        (256, 17, 3),
        (256, 9, 2),
        (256, 5, 1)
    ]
    params["decoder_chw"] = [
        (256, 9, 2),
        (256, 17, 3),
        (128, 33, 4),
        (64, 65, 5),
        (32, 129, 6),
        (1, 257, 7)
    ]
    return params

# loss function
def si_snr(source, estimate_source, eps=1e-5):
    source = source.squeeze(1)
    estimate_source = estimate_source.squeeze(1)
    B, T = source.size()
    source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B , 1
    dot = torch.matmul(estimate_source, source.t())  # B , B
    s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
    e_noise = estimate_source - source
    snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
    lo = 0 - torch.mean(snr)
    return lo

class SiSnr(object):
    def __call__(self, source, estimate_source):
        return si_snr(source, estimate_source)