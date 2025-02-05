from torch import nn


class Conv2dReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True,
                 use_batchnorm=False):

        super().__init__()

        if use_batchnorm:
            bias = False

        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           bias=bias)

        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            net = nn.Sequential(conv2d, nn.BatchNorm2d(out_channels), relu)
        else:
            net = nn.Sequential(conv2d, relu)

        self.net = net

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return self.net(x)


class Conv2dReLU6(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True,
                 use_batchnorm=False):

        super().__init__()

        if use_batchnorm:
            bias = False

        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           bias=bias)

        relu = nn.ReLU6(inplace=True)

        if use_batchnorm:
            net = nn.Sequential(conv2d, nn.BatchNorm2d(out_channels), relu)
        else:
            net = nn.Sequential(conv2d, relu)

        self.net = net

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return self.net(x)


class InvertedBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 expand_ratio=1.,
                 bias=True,
                 use_batchnorm=False,
                 omit_last_activation=False):

        super().__init__()

        bias = bias and (not use_batchnorm)

        modules = []

        internal_channels = int(out_channels * expand_ratio)

        # 1x1
        module = nn.Conv2d(in_channels,
                           internal_channels,
                           1,
                           bias=bias)

        modules.append(module)

        if use_batchnorm:
            module = nn.BatchNorm2d(internal_channels)
            modules.append(module)

        modules.append(nn.ReLU6(inplace=True))

        # 3x3
        module = nn.Conv2d(internal_channels,
                           internal_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=internal_channels,
                           bias=bias)

        modules.append(module)

        if use_batchnorm:
            module = nn.BatchNorm2d(internal_channels)
            modules.append(module)

        modules.append(nn.ReLU6(inplace=True))

        # 1x1
        module = nn.Conv2d(internal_channels,
                           out_channels,
                           1,
                           bias=bias)

        modules.append(module)

        if use_batchnorm:
            module = nn.BatchNorm2d(out_channels)
            modules.append(module)

        if not omit_last_activation:
            modules.append(nn.ReLU6(inplace=True))

        self.net = nn.Sequential(*modules)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return self.net(x)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()

