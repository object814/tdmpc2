"""Mirror decoder for autoencoder pretraining of the TD-MPC2 encoder.

Reconstructs (rgb, state) from the SimNorm latent z produced by
`layers.enc(cfg)`. The conv stack mirrors `layers.conv_deep` exactly (same
num_layers / channel schedule / kernel / stride / padding), so a stride-2
ConvTranspose with k=4,s=2,p=1 inverts each stride-2 conv's spatial halving.

This module is used ONLY during pretraining and is thrown away afterwards —
only the encoder weights are saved.
"""

import math

import torch
import torch.nn as nn


def _conv_deep_channels(img_channels, num_channels, num_layers):
    """Replicate conv_deep's per-layer (in_c, out_c) schedule."""
    in_cs, out_cs = [], []
    in_c, out_c = img_channels, num_channels
    for _ in range(num_layers):
        in_cs.append(in_c)
        out_cs.append(out_c)
        in_c = out_c
        out_c = min(out_c * 2, num_channels * 8)
    return in_cs, out_cs


class Decoder(nn.Module):
    """z (latent_dim) -> (rgb in [-0.5,0.5], state raw)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        image_size = int(cfg.image_size)
        minres = int(getattr(cfg, "cnn_minres", 4))
        num_channels = int(cfg.num_channels)
        # img_channels = 3 * num_cameras; obs_shape['rgb'] = (C, H, W).
        img_channels = int(cfg.obs_shape["rgb"][0])
        latent_dim = int(cfg.latent_dim)
        state_dim = int(cfg.obs_shape["state"][0])

        num_layers = int(math.log2(image_size // minres))
        assert 2 ** num_layers == (image_size // minres), \
            f"image_size {image_size} must downsample to minres={minres}"
        in_cs, out_cs = _conv_deep_channels(img_channels, num_channels, num_layers)
        feat_c = out_cs[-1]
        self._feat_shape = (feat_c, minres, minres)

        # --- rgb branch: project z -> feature map, then transposed convs ---
        self.rgb_fc = nn.Linear(latent_dim, feat_c * minres * minres)
        deconv = []
        # Reverse the forward layers: layer k maps in_cs[k]->out_cs[k] and
        # halves; its inverse maps out_cs[k]->in_cs[k] and doubles.
        for j, k in enumerate(reversed(range(num_layers))):
            in_ch = out_cs[k]
            out_ch = in_cs[k]
            deconv.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            if j < num_layers - 1:
                deconv.append(nn.GroupNorm(1, out_ch))
                deconv.append(nn.SiLU(inplace=False))
        self.rgb_deconv = nn.Sequential(*deconv)

        # --- state branch: small MLP ---
        self.state_head = nn.Sequential(
            nn.Linear(latent_dim, cfg.enc_dim),
            nn.SiLU(inplace=False),
            nn.Linear(cfg.enc_dim, state_dim),
        )

    def forward(self, z):
        b = z.shape[0]
        feat = self.rgb_fc(z).view(b, *self._feat_shape)
        rgb = self.rgb_deconv(feat)          # (B, C, H, W) in [-0.5, 0.5]-ish
        state = self.state_head(z)           # (B, state_dim) raw
        return rgb, state
