import torch
import torch.nn as nn

from .image_encoders.mnas_net import MnasMulti
from .multilevel_reconstruct import MultilevelReconstruct
from .fusion.gru_fusion import GRUFusion
from utils import tocuda


class NeuralRecon(nn.Module):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, cfg):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.image_encoder = MnasMulti(alpha)
        self.neucon_net = MultilevelReconstruct(cfg.MODEL)
        # for fusing to global volume
        self.gru_fusion = GRUFusion(cfg.MODEL, direct_substitute=True)

    def normalizer(self, x):
        '''Normalization of input images'''
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, inputs, save_mesh=False):
        inputs = tocuda(inputs)
        outputs = {}
        imgs = torch.unbind(inputs['imgs'], 1)

        # image feature extraction
        # in: images; out: feature maps
        features = [self.image_encoder(self.normalizer(img)) for img in imgs]

        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)

        # fuse to global volume.
        if not self.training and 'coords' in outputs.keys():
            outputs = self.gru_fusion(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)

        # gather loss.
        print_loss = 'Loss: '
        for k, v in loss_dict.items():
            print_loss += f'{k}: {v} '

        weighted_loss = 0

        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.cfg.LW[i]

        loss_dict.update({'total_loss': weighted_loss})
        return outputs, loss_dict
