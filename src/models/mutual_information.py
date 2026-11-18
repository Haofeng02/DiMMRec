import torch
import torch.nn as nn


class CLUBEstimator(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=512, logvar_clamp=(-10, 5), eps=1e-06):
        super().__init__()
        self.eps = eps
        self.logvar_min, self.logvar_max = logvar_clamp
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, y_dim))
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def get_mu_logvar(self, x):
        mu = self.p_mu(x)
        logvar = torch.clamp(self.p_logvar(x), self.logvar_min, self.logvar_max)
        return mu, logvar

    def _apply_mask(self, tensor, mask):
        if mask is None:
            return tensor
        return tensor[mask]

    def forward(self, x, y, mask=None):
        if mask is not None:
            x = self._apply_mask(x, mask)
            y = self._apply_mask(y, mask)
        if x.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=x.device)
        mu, logvar = self.get_mu_logvar(x)
        var = torch.exp(logvar) + self.eps
        positive = -(y - mu) ** 2 / (2 * var)
        y_perm = y[torch.randperm(y.size(0))].detach()
        negative = -(y_perm - mu) ** 2 / (2 * var)
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def learning_loss(self, x, y, mask=None):
        if mask is not None:
            x = self._apply_mask(x, mask)
            y = self._apply_mask(y, mask)
        if x.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=x.device)
        mu, logvar = self.get_mu_logvar(x)
        var = torch.exp(logvar) + self.eps
        ll = -0.5 * ((mu - y) ** 2 / var + logvar).sum(dim=-1).mean()
        reg = 0.0001 * (mu.pow(2).mean() + logvar.pow(2).mean())
        return -(ll - reg)


class DisentangleMIEstimator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.mi_img_text = CLUBEstimator(dim, dim)
        self.mi_shared = CLUBEstimator(dim * 2, dim)

    def forward(self, z_img, z_text, z_shared, mask=None):
        mi_img_text = self.mi_img_text(z_img, z_text, mask)
        z_concat = torch.cat([z_img, z_text], dim=-1)
        mi_shared = self.mi_shared(z_concat, z_shared, mask)
        return mi_img_text + mi_shared

    def learning_loss(self, z_img, z_text, z_shared, mask=None):
        loss_img_text = self.mi_img_text.learning_loss(z_img, z_text, mask)
        z_concat = torch.cat([z_img, z_text], dim=-1)
        loss_shared = self.mi_shared.learning_loss(z_concat, z_shared, mask)
        return loss_img_text + loss_shared


class DisentangleMILoss(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.mi_estimator = DisentangleMIEstimator(dim)

    def forward(self, z_img, z_text, z_shared, mask=None):
        return self.mi_estimator(z_img, z_text, z_shared, mask)

    def learning_loss(self, z_img, z_text, z_shared, mask=None):
        return self.mi_estimator.learning_loss(z_img, z_text, z_shared, mask)
