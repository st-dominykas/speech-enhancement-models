import torch
import torch.nn as nn

class Simple_STFT_Layer(nn.Module):
    def __init__(self, frame_len, frame_hop):
        super(Simple_STFT_Layer, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop

    def forward(self, x):
        if len(x.shape) != 2:
            print("x must be in [B, T]")
        y = torch.stft(x, n_fft=self.frame_len, hop_length=self.frame_hop,
                       win_length=self.frame_len, return_complex=True, center=False)
        r = y.real
        i = y.imag
        mag = torch.clamp(r ** 2 + i ** 2, self.eps) ** 0.5
        phase = torch.atan2(i + self.eps, r + self.eps)
        return mag, phase

class Pytorch_InstantLayerNormalization(nn.Module):
    """
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    """

    def __init__(self, channels):
        """
            Constructor
        """
        super(Pytorch_InstantLayerNormalization, self).__init__()
        self.epsilon = 1e-7
        self.gamma = nn.Parameter(torch.ones(1, 1, channels), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, 1, channels), requires_grad=True)
        self.register_parameter("gamma", self.gamma)
        self.register_parameter("beta", self.beta)

    def forward(self, inputs):
        # calculate mean of each frame
        mean = torch.mean(inputs, dim=-1, keepdim=True)

        # calculate variance of each frame
        variance = torch.mean(torch.square(inputs - mean), dim=-1, keepdim=True)
        # calculate standard deviation
        std = torch.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs

class SeperationBlock(nn.Module):
    def __init__(self, input_size=257, hidden_size=128, dropout=0.25, LSTM_size=2):
        super(SeperationBlock, self).__init__()
        self.LSTM_size = LSTM_size

        self.rnn1 = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.rnn2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        if LSTM_size ==3 or LSTM_size==4:
            self.rnn3 = nn.LSTM(input_size=hidden_size,
                                hidden_size=hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.0,
                                bidirectional=False)
        if LSTM_size==4:
            self.rnn4 = nn.LSTM(input_size=hidden_size,
                                hidden_size=hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.0,
                                bidirectional=False)

        self.drop = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, (h, c) = self.rnn1(x)
        x1 = self.drop(x1)
        x2, _ = self.rnn2(x1)
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)

        if self.LSTM_size==3 or self.LSTM_size==4:
            x3, _ = self.rnn2(x2)
            x3 = self.drop(x3)
            mask = self.dense(x3)
            mask = self.sigmoid(mask)
        if self.LSTM_size==4:
            x4, _ = self.rnn2(x3)
            x4 = self.drop(x4)
            mask = self.dense(x4)
            mask = self.sigmoid(mask)

        return mask

class Pytorch_DTLN(nn.Module):
    def __init__(self, frame_len=512, frame_hop=128, dropout=0.25, 
                       encoder_size = 256, hidden_size = 128, LSTM_size = 2, window='rect'):
        super(Pytorch_DTLN, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.stft = Simple_STFT_Layer(frame_len, frame_hop)

        self.sep1 = SeperationBlock(input_size=(frame_len // 2 + 1), hidden_size=hidden_size, dropout=dropout, LSTM_size=LSTM_size)

        #self.encoder_size = 256
        self.encoder_conv1 = nn.Conv1d(in_channels=frame_len, out_channels=encoder_size,
                                       kernel_size=1, stride=1, bias=False)

        
        self.encoder_norm1 = Pytorch_InstantLayerNormalization(channels=encoder_size)

        self.sep2 = SeperationBlock(input_size=encoder_size, hidden_size=hidden_size, dropout=dropout)

        self.decoder_conv1 = nn.Conv1d(in_channels=encoder_size, out_channels=frame_len,
                                       kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        """
        :param x:  [N, T]
        :return:
        """
        batch, n_frames = x.shape

        mag, phase = self.stft(x)
        mag = mag.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)

        # N, T, hidden_size
        mask = self.sep1(mag)
        estimated_mag = mask * mag

        s1_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft2(s1_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)

        mask_2 = self.sep2(encoded_f_norm)
        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0, 2, 1)

        decoded_frame = self.decoder_conv1(estimated)

        ## overlap and add
        out = torch.nn.functional.fold(
            decoded_frame,
            (n_frames, 1),
            kernel_size=(self.frame_len, 1),
            padding=(0, 0),
            stride=(self.frame_hop, 1),
        )
        out = out.reshape(batch, -1)

        return out

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