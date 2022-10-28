import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from SEnet import SELayer

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        # print(x.device, label.device, one_hot.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output


class CausalConv1d(nn.Module):
    """
    Input and output sizes will be the same.
    """
    def __init__(self, in_size, out_size, kernel_size, dilation=1, groups=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size, padding=self.pad,
                               dilation=dilation, groups=groups, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., :-self.pad]
        return x


class ResidualLayer(nn.Module):
    def __init__(self, residual_size, skip_size, dilation):
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_size, residual_size,
                                        kernel_size=3, dilation=dilation)
        self.bn_filter = nn.BatchNorm1d(residual_size)
        self.conv_gate = CausalConv1d(residual_size, residual_size,
                                      kernel_size=3, dilation=dilation)
        self.bn_gate = nn.BatchNorm1d(residual_size)
        self.resconv1_1 = nn.Conv1d(residual_size, residual_size, kernel_size=1)
        # self.res_bn = nn.BatchNorm1d(residual_size)

        # self.res_se = SELayer(residual_size, reduction=8)
        self.skipconv1_1 = nn.Conv1d(residual_size, skip_size, kernel_size=1)
        # self.skip_bn = nn.BatchNorm1d(skip_size)
        # self.skip_se = SELayer(skip_size, reduction=8)

    def forward(self, x):
        conv_filter = self.conv_filter(x)
        conv_filter = self.bn_filter(conv_filter)

        conv_gate = self.conv_gate(x)
        conv_gate = self.bn_gate(conv_gate)

        activation = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)

        fx = self.resconv1_1(activation)

        skip = self.skipconv1_1(fx)
        residual = fx + x
        # residual=[batch,residual_size,seq_len]  skip=[batch,skip_size,seq_len]
        return skip, residual


class DilatedStack(nn.Module):
    def __init__(self, residual_size, skip_size, dilation_depth):
        super(DilatedStack, self).__init__()
        residual_stack = [ResidualLayer(residual_size, skip_size, 2 ** layer)
                          for layer in range(dilation_depth)]
        self.residual_stack = nn.ModuleList(residual_stack)

    def forward(self, x):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
            # skip =[1,batch,skip_size,seq_len]
        return torch.cat(skips, dim=0), x  # [layers,batch,skip_size,seq_len]


class SpecWaveNet(nn.Module):
    def __init__(self, input_size=128, out_size=128, residual_size=512, skip_size=512,
                 dilation_cycles=2, dilation_depth=4, num_classes=41):

        super(SpecWaveNet, self).__init__()

        self.tgramnet = WavegramWaveNet()
        # self.conv_extrctor = nn.Conv1d(1, input_size, 1024, 512, 1024 // 2, bias=False)
        self.input_conv = CausalConv1d(input_size, residual_size, kernel_size=2)
        self.dilated_stacks = nn.ModuleList(
            [DilatedStack(residual_size, skip_size, dilation_depth)
             for cycle in range(dilation_cycles)]
        )
        self.post = nn.Sequential(#
            nn.BatchNorm1d(skip_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_size, skip_size, 313, groups=skip_size),
            nn.BatchNorm1d(skip_size),
            nn.Conv1d(skip_size, out_size, 1),
        )
        # self.fc = nn.Linear(out_size*2, out_size)
        self.arcface = ArcMarginProduct(out_size*2, num_classes, m=0.7, s=30)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, wav, label=None):
        # x_wav = self.conv_extrctor(wav)
        # x_wav = self.conv_encoder(x_wav)
        _, tgram_feature = self.tgramnet(x, wav, label)

        # x = torch.cat([x, x_wav], dim=1)
        x = self.input_conv(x)  # [batch,residual_size, seq_len]
        skip_connections = []

        for cycle in self.dilated_stacks:

            skips, x = cycle(x)
            skip_connections.append(skips)

        ## skip_connection=[total_layers,batch,skip_size,seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)

        # gather all output skip connections to generate output, discard last residual output
        out = skip_connections.sum(dim=0)  # [batch,skip_size,seq_len]

        out = self.post(out)

        feature = out.view(out.size(0), -1)
        feature = torch.cat([tgram_feature, feature], dim=1)
        if label is None:
            return feature
        out = self.arcface(feature, label)

        #[bacth,seq_len,out_size]
        return out, feature


class WavegramWaveNet(nn.Module):
    def __init__(self, input_size=128, out_size=128, residual_size=512, skip_size=512,
                 dilation_cycles=2, dilation_depth=4, num_classes=41):

        super(WavegramWaveNet, self).__init__()
        self.conv_extrctor = nn.Conv1d(1, input_size, 1024, 512, 1024 // 2, bias=False)
        self.input_conv = CausalConv1d(input_size, residual_size, kernel_size=2)
        self.dilated_stacks = nn.ModuleList(
            [DilatedStack(residual_size, skip_size, dilation_depth)
             for cycle in range(dilation_cycles)]
        )
        self.post = nn.Sequential(#
            nn.BatchNorm1d(skip_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_size, skip_size, 313, groups=skip_size),
            nn.BatchNorm1d(skip_size),
            nn.Conv1d(skip_size, out_size, 1),
        )
        self.arcface = ArcMarginProduct(out_size, num_classes, m=0.7, s=30)
        # self.fc = nn.Linear(out_size, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, wav, label=None):
        x_wav = self.conv_extrctor(wav)
        # x_wav = self.conv_encoder(x_wav)

        # x = torch.cat([x, x_wav], dim=1)
        x = self.input_conv(x_wav)  # [batch,residual_size, seq_len]
        skip_connections = []

        for cycle in self.dilated_stacks:

            skips, x = cycle(x)
            skip_connections.append(skips)

        ## skip_connection=[total_layers,batch,skip_size,seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)

        # gather all output skip connections to generate output, discard last residual output
        out = skip_connections.sum(dim=0)  # [batch,skip_size,seq_len]

        out = self.post(out)
        # out = self.pool(out)

        feature = out.view(out.size(0), -1)
        if label is None:
            return out, feature
        out = self.arcface(feature, label)

        #[bacth,seq_len,out_size]
        return out, feature
