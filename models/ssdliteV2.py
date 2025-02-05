import torch
import os
from .layer import *
from .mobilenetv2 import mobilenet_v2
import nn
from anchors import DefaultBox

class SSDLiteV2(nn.Module):
    presets = {
        'default': {
            's_min': 0.2,
            's_max': 0.9,
            's_extra_min': None,
        },
        'ssdlite320': {
            'inherit': 'ssdlite320-voc'
        },
        'ssdlite416': {
            'inherit': 'ssdlite416-voc'
        },
        'ssdlite512': {
            'inherit': 'ssdlite512-voc'
        },
        'ssdlite640': {
            'inherit': 'ssdlite640-voc'
        },
        'ssdlite320-voc': {
            'inherit': 'ssdlite-voc1',

            'width': 320,
            'height': 320,
            'num_grids': (20, 10, 5, 3, 1),
        },
        'ssdlite416-voc': {
            'inherit': 'ssdlite-voc2',

            'width': 416,
            'height': 416,
            'num_grids': (26, 13, 7, 4, 2, 1),
        },
        'ssdlite512-voc': {
            'inherit': 'ssdlite-voc2',

            'width': 512,
            'height': 512,
            'num_grids': (32, 16, 8, 4, 2, 1),
        },
        'ssdlite640-voc': {
            'inherit': 'ssdlite-voc3',

            'width': 640,
            'height': 640,
            'num_grids': (40, 20, 10, 5, 3, 1),
        },
        'ssdlite-voc1': {
            'num_class': 20,
            'extras': (
                # output_channels, kernel_size, stride, padding
                (512, 3, 2, 1), # 5x5
                (256, 3, 2, 1), # 3x3
                (128, 3, 1, 0)  # 1x1
            ),
            'ratios': (
                (2., 3.), (2., 3.), (2., 3.), (2., 3.), (2.)
            ),
        },
        'ssdlite-voc2': {
            'num_class': 20,
            'extras': (
                # output_channels, kernel_size, stride, padding
                (512, 3, 2, 1), # 8x8
                (256, 3, 2, 1), # 4x4
                (256, 3, 2, 1), # 2x2
                (128, 3, 2, 1)  # 1x1
            ),
            'ratios': (
                (2., 3.), (2., 3.), (2., 3.), (2., 3.), (2.), (2.)
            ),
        },
        'ssdlite-voc3': {
            'num_class': 20,
            'extras': (
                # output_channels, kernel_size, stride, padding
                (512, 3, 2, 1), # 8x8
                (256, 3, 2, 1), # 4x4
                (256, 3, 2, 1), # 2x2
                (128, 3, 2, 0)  # 1x1
            ),
            'ratios': (
                (2., 3.), (2., 3.), (2., 3.), (2., 3.), (2., 3.), (2.)
            ),
        },
    }

    def __init__(self, preset='ssdlite',args=None, params=None, pretrained=False,class_number=20):
        super().__init__()

        self.args = args
        self.name = self.args.model.lower()
        #set rfb_net
        self.rfb_ = BasicRFB_a(192,192,stride=1,scale=1.0)
        self.rfb_s = BasicRFB_a(576,576,stride=1,scale=1.0)
        self.rfb = BasicRFB(576,576,stride=1,scale=1.0)

        # to avoid PyCharm warning
        self.extras = None

        self.conf_regressions = None
        self.loc_regressions = None

        self.in_channels = None

        # merge network parameters
        self.params = {}
        self.apply_params(self.presets['default'])
        self.apply_params(self.presets[self.name])
        self.apply_params(params)

        p = self.params

        self.num_class = class_number + 1 #background
        # build anchor
        if self.args.feature == True:
            p['num_grids'] = (64, ) + p['num_grids']
            p['ratios'] = ((2.0, 3.0), ) + p['ratios']
            
        self.default_box = DefaultBox(p['num_grids'],
                                      p['ratios'],
                                      p['s_min'],
                                      p['s_max'],
                                      p['s_extra_min'])


        # setup backbone
        self.build_backbone(pretrained)

        # build extra layers
        self.build_extras()

        # build regression layers
        self.build_regressions()

        # initialize weight/bias
        self.initialize_parameters()

    def forward(self, x):
        pyramid = []

        if self.args.feature == True:
            x = self.b0(x)
            s = self.rfb_(x)
            pyramid.append(s)

            x = self.b1(x)
            s = self.rfb_s(x)
            pyramid.append(s)

            x = self.b2(x)
            s = self.rfb(x)
            pyramid.append(s)

        else:
            x = self.b0(x)
            s = self.rfb_s(x)
            pyramid.append(s)

            # scale1, ...
            x = self.b1(x)
            s = self.rfb(x)
            pyramid.append(s)

        for extra in self.extras:
            x = extra(x)
            pyramid.append(x)

        batch_size = int(x.shape[0])

        conf = []
        loc = []

        for i, x in enumerate(pyramid):
            tmp = self.conf_regressions[i](x).permute(0, 2, 3, 1)
            conf.append(tmp.reshape(batch_size, -1, self.num_class))

            tmp = self.loc_regressions[i](x).permute(0, 2, 3, 1)
            loc.append(tmp.reshape(batch_size, -1, 4))

        conf = torch.cat(conf, dim=1)
        loc = torch.cat(loc, dim=1)

        return conf, loc

    def get_anchor_box(self):
        return self.default_box

    def get_input_size(self):
        return self.params['width'], self.params['height']

    def build_extras(self):
        in_channels = self.calc_in_channel_width(self.b1)

        extras = []
        for layers in self.params['extras']:
            extra, in_channels = self.build_extra(in_channels, layers)

            extras.append(extra)

        self.extras = nn.ModuleList(extras)

    def build_extra(self, in_channels, layer):
        extra = []

        out_channels = layer[0]
        kernel_size = layer[1]
        stride = layer[2]
        padding = layer[3]

        conv = nn.InvertedBottleneck(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     expand_ratio=.5,
                                     use_batchnorm=True)

        extra.append(conv)

        extra = nn.Sequential(*extra)
        extra.out_channels = out_channels

        return extra, out_channels

    def build_regressions(self):
        conf_regressions = []
        loc_regressions = []

        extras = [self.b0, self.b1]
        if self.args.feature == True:
            extras.append(self.b2)
            
        extras.extend(self.extras)

        # from extras
        for i, extra in enumerate(extras):
            in_channels = self.calc_in_channel_width(extra)
           # if i == 1:
            #    in_channels = 640
            n = self.default_box.get_num_ratios(i)

            conf_regressions.append(Regression(in_channels,
                                               n * self.num_class,
                                               3, 1, 1,
                                               bias=True,
                                               use_batchnorm=True))
            loc_regressions.append(Regression(in_channels,
                                              n * 4,
                                              3, 1, 1,
                                              bias=True,
                                              use_batchnorm=True))

        self.conf_regressions = nn.ModuleList(conf_regressions)
        self.loc_regressions = nn.ModuleList(loc_regressions)

    def build_backbone(self, pretrained):
        model = mobilenet_v2(pretrained=pretrained)
        features = model.features

        if self.args.feature == True:
            b0 = nn.Sequential(features[0:7], features[7].conv[0])
            b1 = nn.Sequential(features[7].conv[1:], features[8:14], features[14].conv[0])
            b2 = nn.Sequential(features[14].conv[1:],
                                features[15:],
                                nn.Conv2d(1280, 576, kernel_size=1, stride=1, padding=0, groups=1))
            b0.out_channels = 192
            b1.out_channels = 576
            b2.out_channels = 576

            self.b0 = b0
            self.b1 = b1
            self.b2 = b2
        else:
            b0 = nn.Sequential(features[0:14], features[14].conv[0])
            b1 = nn.Sequential(features[14].conv[1:], 
                                features[15:],
                                nn.Conv2d(1280, 576, kernel_size=1, stride=1, padding=0, groups=1))
            b0.out_channels = 576
            b1.out_channels = 576

            self.b0 = b0
            self.b1 = b1

        in_channels = self.calc_in_channel_width(b0)

        self.in_channels = in_channels

    def initialize_parameters(self):
        self.init_parameters(self.extras)

        self.init_parameters(self.conf_regressions)
        self.init_parameters(self.loc_regressions)

    def apply_params(self, params):
        if params is None:
            return

        if 'inherit' in params.keys():
            self.apply_params(self.presets[params['inherit']])

        for k, v in params.items():
            if k == 'inherit':
                continue

            self.params[k] = v

    @staticmethod
    def init_parameters(layer):
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def calc_in_channel_width(prev):
        if not hasattr(prev, 'out_channels'):
            raise Exception("failed to guess input channel width")

        return prev.out_channels


def build_ssdlite(preset='ssdlite',args=None, params=None, pretrained=False, class_num=21):
    return SSDLiteV2(preset, args, params=params, pretrained=pretrained,class_number=class_num)

