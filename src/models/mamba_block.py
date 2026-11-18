import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2
except ImportError:
    pass


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob if drop_prob is not None else 0.0

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            if torch.isnan(x).any():
                return torch.nan_to_num(x)
            return x
        if torch.isnan(x).any():
            x = torch.nan_to_num(x)
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


class SingleMambaBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.res_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.mixer(x)
        return shortcut + self.res_scale * self.drop_path1(x)


class CrossModalTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=1.0, drop_path=0.0, d_state=16, d_conv=2, expand=1):
        super().__init__()
        self.block_fusion_img = SingleMambaBlock(dim=dim, mlp_ratio=mlp_ratio, drop_path=drop_path, d_state=d_state, d_conv=d_conv, expand=expand)
        self.block_fusion_text = SingleMambaBlock(dim=dim, mlp_ratio=mlp_ratio, drop_path=drop_path, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, z_img, z_text, z_shared, mask=None):
        if torch.isnan(z_img).any():
            z_img = torch.nan_to_num(z_img)
        if torch.isnan(z_text).any():
            z_text = torch.nan_to_num(z_text)
        if torch.isnan(z_shared).any():
            z_shared = torch.nan_to_num(z_shared)

        _, n_shared, _ = z_shared.shape
        n_img = z_img.shape[1]
        n_text = z_text.shape[1]

        z_fusion_img_in = torch.cat([z_img, z_shared], dim=1)
        z_fusion_img_out = self.block_fusion_img(z_fusion_img_in)
        z_img_out = z_fusion_img_out[:, :n_img, :]
        z_shared_img = z_fusion_img_out[:, n_img:n_img + n_shared, :]

        z_fusion_text_in = torch.cat([z_text, z_shared], dim=1)
        z_fusion_text_out = self.block_fusion_text(z_fusion_text_in)
        z_text_out = z_fusion_text_out[:, :n_text, :]
        z_shared_text = z_fusion_text_out[:, n_text:n_text + n_shared, :]

        z_shared_out = (z_shared_img + z_shared_text) / 2.0
        return z_img_out, z_text_out, z_shared_out
