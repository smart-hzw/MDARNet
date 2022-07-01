import torch
from torch import nn
from torch.nn import functional as F
from DWT import dwt_97,idwt_97

class SepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=bias)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        # 将maxpooling 与 global average pooling 结果拼接在一起
        return torch.cat((torch.max(x, 1) [0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = SepConv(in_channel=in_planes, out_channel=out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avgPoolH = nn.AdaptiveAvgPool2d((None, 1))
        self.avgPoolW = nn.AdaptiveAvgPool2d((1, None))
        self.maxPoolH = nn.AdaptiveMaxPool2d((None, 1))
        self.maxPoolW = nn.AdaptiveMaxPool2d((1, None))

        self.conv_1x1_h = nn.Conv2d(in_channels=2 * channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                                    bias=False)
        self.conv_1x1_w = nn.Conv2d(in_channels=2 * channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                                    bias=False)
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                                  bias=False)
        self.bn = nn.BatchNorm2d(channel, eps=1e-5, momentum=0.01, affine=True)
        self.Relu = nn.ReLU()

        self.F_h = nn.Sequential(  # 激发操作
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.F_w = nn.Sequential(  # 激发操作
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()
        res = x
        x_h = torch.cat([self.avgPoolH(x), self.maxPoolH(x)], 1).permute(0, 1, 3, 2)
        x_h = self.conv_1x1_h(x_h)
        x_w = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)
        x_w = self.conv_1x1_w(x_w)
        x_cat = torch.cat([x_h, x_w], dim=3)
        x = self.Relu(self.bn(self.conv_1x1(x_cat)))
        x_h, x_w = x.split([H, W], 3)

        x_h = self.F_h(x_h.permute(0, 1, 3, 2))
        x_w = self.F_w(x_w)
        s_h = self.sigmoid_h(x_h)
        s_w = self.sigmoid_h(x_w)

        out = res * s_h.expand_as(res) * s_w.expand_as(res)

        return out


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class SCDAB(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super(SCDAB, self).__init__()
        pooling_r = 4
        self.head = nn.Sequential(
            SepConv(in_channel=n_feat, out_channel=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(n_feat),
            nn.ReLU(),
            SepConv(in_channel=n_feat, out_channel=n_feat, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            SepConv(in_channel=n_feat, out_channel=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(n_feat)
        )
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=1),
            SepConv(in_channel=n_feat, out_channel=n_feat, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.ReLU = nn.ReLU()
        self.tail = SepConv(in_channel=n_feat, out_channel=n_feat, kernel_size=3)

    def forward(self, x):
        # res = self.body(x)
        res = x
        x = self.head(x)
        sa_branch = self.SA(x)
        ca_branch = self.CA(x)
        x1 = torch.cat([sa_branch, ca_branch], dim=1)
        x1 = self.conv1x1(x1)
        # 相比于之前的DAB不一样的地方
        x2 = torch.sigmoid(
            torch.add(x, F.interpolate(self.SC(x), x.size() [2:])))
        out = torch.mul(x1, x2)
        out = self.tail(out)
        out = out + res
        out = self.ReLU(out)
        return out


class RRG(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, num_dab):
        super(RRG, self).__init__()
        modules_body = [SCDAB(n_feat, reduction) for _ in range(num_dab)]
        modules_body.append(SepConv(in_channel=n_feat, out_channel=n_feat, kernel_size=kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RRG_v1(nn.Module):
    def __init__(self, n_feat, reduction, num_dab):
        super(RRG_v1, self).__init__()

        modules_body = [SCDAB(n_feat, reduction) for _ in range(num_dab)]
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(
            SepConv(in_channel=n_feat, out_channel=n_feat, kernel_size=3, padding=1, stride=1)
        )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        res = x
        # x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += res
        x = self.ReLU(x)
        return x


class LRDB(nn.Module):
    def __init__(self, n_feat, reduction, num_dab):
        super(LRDB, self).__init__()
        self.num_dab = num_dab
        modules_body = nn.ModuleList()
        for i in range(num_dab):
            modules_body.append(SCDAB(n_feat, reduction))
        conv1_1 = nn.ModuleList()
        conv3_3 = nn.ModuleList()
        for i in range(num_dab - 1):
            conv1_1.append(nn.Conv2d(in_channels=n_feat * 2, out_channels=n_feat, kernel_size=1, padding=0, stride=1))
            conv3_3.append(SepConv(in_channel=n_feat, out_channel=n_feat, kernel_size=3, padding=1, stride=1))
        self.conv1_1 = nn.Sequential(*conv1_1)
        self.conv3_3 = nn.Sequential(*conv3_3)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=n_feat * (num_dab + 1), out_channels=n_feat, kernel_size=1, padding=0, stride=1)
        )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        res = x
        body_dense = []
        body_dense.append(x)
        res_dense = x
        out = self.body [0](x)
        body_dense.append(out)
        x = torch.cat([out, res_dense], 1)
        for i in range(1, self.num_dab):
            res_dense = out
            x = self.conv1_1 [i - 1](x)
            x = self.conv3_3 [i - 1](x)
            out = self.body [i](x)
            body_dense.append(out)
            x = torch.cat([out, res_dense], 1)
        out = torch.cat(body_dense, 1)
        out = self.tail(out)
        out += res
        return out


class LRDN(nn.Module):
    def __init__(self, in_channel, out_channel, num_dab=4, num_lrdb=4, n_feats=64, reduction=16):
        super(LRDN, self).__init__()
        self.num_lrdb = num_lrdb
        self.conv1 = SepConv(in_channel=in_channel, out_channel=n_feats, kernel_size=3, padding=1, stride=1)
        self.conv2 = SepConv(in_channel=n_feats, out_channel=n_feats, kernel_size=3, padding=1, stride=1)
        module_body = nn.ModuleList()
        for i in range(num_lrdb):
            module_body.append(LRDB(n_feats, reduction, num_dab))
        self.body = nn.Sequential(*module_body)
        self.conv1_1 = nn.Conv2d(in_channels=n_feats * (num_lrdb + 1), out_channels=n_feats, kernel_size=1, padding=0,
                                 stride=1)
        self.conv3 = SepConv(in_channel=n_feats, out_channel=n_feats, kernel_size=3, padding=1, stride=1)
        self.conv4 = SepConv(in_channel=n_feats, out_channel=out_channel, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        res = x
        x = self.conv2(x)
        module_body = []
        module_body.append(x)
        for i in range(self.num_lrdb):
            x = self.body [i](x)
            module_body.append(x)
        x = torch.cat(module_body, 1)
        x = self.conv1_1(x)
        x = self.conv3(x)
        x += res
        x = self.conv4(x)
        return x


class NET(nn.Module):
    def __init__(self, in_channel, out_channel, num_dab=4, num_rrg=4, n_feats=64, reduction=16):
        super(NET, self).__init__()
        kernel_size = 3
        self.num_rrg = num_rrg
        self.conv1 = nn.Sequential(
            SepConv(in_channel=in_channel, out_channel=n_feats, kernel_size=kernel_size),
        )
        self.conv2 = nn.Sequential(
            SepConv(in_channel=n_feats, out_channel=n_feats, kernel_size=kernel_size)
        )
        modules_body = [RRG_v1(n_feat=n_feats, reduction=reduction, num_dab=num_dab)
                        for _ in range(self.num_rrg)]
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats * (self.num_rrg + 1), out_channels=n_feats, kernel_size=1, padding=0,
                      stride=1),
            SepConv(in_channel=n_feats, out_channel=n_feats, kernel_size=kernel_size)
        )
        self.conv4 = nn.Sequential(
            SepConv(in_channel=n_feats, out_channel=out_channel, kernel_size=kernel_size)
        )

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.conv1(x)
        res = x
        x = self.conv2(x)
        body = []
        body.append(x)
        for i in range(self.num_rrg):
            x = self.body [i](x)
            body.append(x)
        x = torch.cat(body, 1)
        x = self.conv3(x)
        x = self.conv4(x + res)

        return x


class Net_test1(nn.Module):
    def __init__(self):
        super(Net_test1, self).__init__()

        self.Net1 = NET(in_channel=12, out_channel=12, num_dab=4, num_rrg=4, n_feats=64, reduction=16)
        self.Net2 = NET(in_channel=12, out_channel=12, num_dab=4, num_rrg=4, n_feats=64, reduction=16)

    def forward(self, x):
        subband1 = dwt_97(x)  # Nx12xWxH
        subband2 = dwt_97(subband1 [:, 0:3, :, :])  # Nx12xWxH

        x = self.Net2(subband2)
        x = idwt_97(x)
        subband1 [:, 0:3, :, :] = x

        lo2 = x
        x = self.Net1(subband1)
        lo1 = x
        x = idwt_97(x)
        return lo1, lo2, x




from utils import print_network

if __name__ == '__main__':
    x = torch.randint(0, 250, [1, 3, 128, 128]).type(torch.float32).cuda()
    # net = MDARNet(128,128,1).cuda()
    # net = LRDN(64,16,3,3).cuda()

    # net = SCDAB(64)
    # net = NET(in_channel=3, out_channel=3).cuda()
    net = Net_test1().cuda()
    print_network(net)
    out = net(x)




