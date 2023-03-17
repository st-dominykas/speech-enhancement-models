import torch
import torch.nn as nn

class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        output = torch.stack([c_real, c_im], dim=-1)
        return output

class CConvTranspose2d(nn.Module):
    """
      Class of complex valued dilation convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                             out_channels=self.out_channels, 
                                             kernel_size=self.kernel_size, 
                                             output_padding=self.output_padding,
                                             padding=self.padding,
                                             stride=self.stride)
        
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                           out_channels=self.out_channels, 
                                           kernel_size=self.kernel_size, 
                                           output_padding=self.output_padding, 
                                           padding=self.padding,
                                           stride=self.stride)
        
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        output = torch.stack([ct_real, ct_im], dim=-1)
        return output

class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                     affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                   affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)  
        
        output = torch.stack([n_real, n_im], dim=-1)
        return output

class Encoder(nn.Module):
    """
    Class of upsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)
        
        return acted

class Decoder(nn.Module):
    """
    Class of downsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45,
                 output_padding=(0,0), padding=(0,0), last_layer=False):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        
        self.last_layer = last_layer
        
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                       kernel_size=self.filter_size, stride=self.stride_size, 
                                       output_padding=self.output_padding, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconvt(x)
        
        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag
            
        return output

class DCUnet20(nn.Module):
    """
    Deep Complex U-Net class of the model.
    """
    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(7,1), stride_size=(1,1), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(1,7), stride_size=(1,1), in_channels=45, out_channels=45)
        self.downsample2 = Encoder(filter_size=(7,5), stride_size=(2,2), in_channels=45, out_channels=90)
        self.downsample3 = Encoder(filter_size=(7,5), stride_size=(2,1), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample5 = Encoder(filter_size=(5,3), stride_size=(2,1), in_channels=90, out_channels=90)
        self.downsample6 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample7 = Encoder(filter_size=(5,3), stride_size=(2,1), in_channels=90, out_channels=90)
        self.downsample8 = Encoder(filter_size=(5,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample9 = Encoder(filter_size=(5,3), stride_size=(2,1), in_channels=90, out_channels=128)
        
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(5,3), stride_size=(2,1), in_channels=128, out_channels=90, output_padding=(1,0))
        self.upsample1 = Decoder(filter_size=(5,3), stride_size=(2,2), in_channels=180, out_channels=90, output_padding=(1,1))
        self.upsample2 = Decoder(filter_size=(5,3), stride_size=(2,1), in_channels=180, out_channels=90, output_padding=(1,0))
        self.upsample3 = Decoder(filter_size=(5,3), stride_size=(2,2), in_channels=180, out_channels=90, output_padding=(1,1))
        self.upsample4 = Decoder(filter_size=(5,3), stride_size=(2,1), in_channels=180, out_channels=90, output_padding=(1,0))
        self.upsample5 = Decoder(filter_size=(5,3), stride_size=(2,2), in_channels=180, out_channels=90, output_padding=(0,1))
        self.upsample6 = Decoder(filter_size=(7,5), stride_size=(2,1), in_channels=180, out_channels=90, output_padding=(1,0))
        self.upsample7 = Decoder(filter_size=(7,5), stride_size=(2,2), in_channels=180, out_channels=90, output_padding=(0,1))
        self.upsample8 = Decoder(filter_size=(1,7), stride_size=(1,1), in_channels=135, out_channels=90)
        self.upsample9 = Decoder(filter_size=(7,1), stride_size=(1,1), in_channels=135, out_channels=1, last_layer=True)
        
        
    def forward(self, x, is_istft=True):
        # downsampling/encoding
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0) 
        d2 = self.downsample2(d1)        
        d3 = self.downsample3(d2)        
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4)
        d6 = self.downsample6(d5)
        d7 = self.downsample7(d6)
        d8 = self.downsample8(d7)
        d9 = self.downsample9(d8)
        
        
        # upsampling/decoding 
        u0 = self.upsample0(d9)
        # skip-connection
        c0 = torch.cat((u0, d8), dim=1)
        
        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d7), dim=1)
        
        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d6), dim=1)
        
        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d5), dim=1)

        u4 = self.upsample4(c3)
        c4 = torch.cat((u4, d4), dim=1)

        u5 = self.upsample5(c4)
        c5 = torch.cat((u5, d3), dim=1)

        u6 = self.upsample6(c5)
        c6 = torch.cat((u6, d2), dim=1)

        u7 = self.upsample7(c6)
        c7 = torch.cat((u7, d1), dim=1)

        u8 = self.upsample8(c7)
        c8 = torch.cat((u8, d0), dim=1)

        u9 = self.upsample9(c8)
        
        # u9 - the mask
        output = u9 * x
        if is_istft:
            output = torch.squeeze(output, 1)
            output = torch.istft(output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
        
        return output