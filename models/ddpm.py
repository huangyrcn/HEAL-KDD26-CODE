"""
M-CRD: Anchor-Conditioned Residual Diffusion
架构: Pre-LN + FiLM (AdaLN) + Residual
"""
import math
import torch
import torch.nn as nn


def cosine_beta_schedule(T, s=0.008):
    """Cosine beta schedule (Nichol & Dhariwal 2021). 确保 alpha_bar(T) → 0."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clamp(min=1e-6, max=0.999).float()


class FiLMBlock(nn.Module):
    """
    Pre-LN + FiLM + Residual 块

    公式: x = x + MLP( FiLM( Norm(x), condition ) )
    FiLM: x_norm * (1 + scale) + shift
    """
    def __init__(self, dim, dim_cond, dim_hidden=None):
        super().__init__()
        dim_hidden = dim_hidden or dim * 4

        # Pre-LN
        self.norm = nn.LayerNorm(dim)

        # FiLM 生成器: condition -> (scale, shift)
        self.film_gen = nn.Linear(dim_cond, dim * 2)

        # MLP: GELU + Linear
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim),
        )

    def forward(self, x, condition):
        """
        Args:
            x: [B, D] 输入特征
            condition: [B, dim_cond] 全局条件 (时间 + anchor)
        """
        # Pre-LN
        x_norm = self.norm(x)

        # FiLM 调制
        film_params = self.film_gen(condition)
        scale, shift = film_params.chunk(2, dim=-1)
        x_modulated = x_norm * (1 + scale) + shift

        # MLP + Residual
        return x + self.mlp(x_modulated)


class Backbone(nn.Module):
    """
    条件骨干网络: [xt; anchor] concat 输入，time 通过 AdaLN (FiLM) 注入
    """
    def __init__(
        self,
        dim_in,
        dim_hidden,
        num_layers,
        n_steps,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden

        # 输入投影: concat [xt, anchor]
        self.input_proj = nn.Linear(dim_in * 2, dim_hidden)

        # 时间嵌入
        self.time_embed = nn.Embedding(n_steps, dim_hidden)

        # 条件: 仅 time → dim_cond = dim_hidden
        dim_cond = dim_hidden

        # FiLM 块堆叠 (time-only conditioning)
        self.blocks = nn.ModuleList([
            FiLMBlock(dim_hidden, dim_cond) for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Linear(dim_hidden, dim_in)

    def forward(self, x, t_idx, anchor):
        """
        Args:
            x: [B, dim_in] 加噪的残差
            t_idx: [B] 时间步索引
            anchor: [B, dim_in] 锚点嵌入
        Returns:
            [B, dim_in] 预测
        """
        # Concat [xt, anchor] 输入
        h = self.input_proj(torch.cat([x, anchor], dim=-1))

        # Time-only conditioning
        cond = self.time_embed(t_idx)

        # 通过 FiLM 块
        for block in self.blocks:
            h = block(h, cond)

        # 输出投影
        return self.output_proj(h)


class ResidualDiffusion(nn.Module):
    """
    M-CRD: 以 Anchor 为条件的残差扩散模型

    训练目标: 学习 residual = neighbor - anchor 的噪声
    采样输出: anchor + generated_residual
    """
    def __init__(
        self,
        dim_in,
        dim_hidden,
        num_layers,
        T,
        beta_1,
        beta_T,
        anchor_scale=True,
        cfg_prob=0.0,
        pred_type="x0",
        schedule="cosine",
        cosine_weight=1.0,
        min_snr_gamma=5.0,
    ):
        super().__init__()
        self.T = T
        self.dim_in = dim_in
        self.anchor_scale = anchor_scale
        self.cfg_prob = cfg_prob
        self.pred_type = pred_type  # "eps", "x0", or "v"
        self.schedule = schedule
        self.cosine_weight = cosine_weight
        self.min_snr_gamma = min_snr_gamma

        # global_scale: per-dim std of residuals, set via set_global_scale()
        self.register_buffer('global_scale', None)
        # residual_mean: per-dim mean of residuals, set via set_residual_mean()
        self.register_buffer('residual_mean', None)

        if schedule == "cosine":
            beta = cosine_beta_schedule(T)
        else:
            beta = torch.linspace(beta_1, beta_T, T)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

        self.backbone = Backbone(
            dim_in,
            dim_hidden,
            num_layers,
            T,
        )

    def set_global_scale(self, scale: torch.Tensor):
        """Set per-dim std for residual normalization. scale: [D]"""
        self.global_scale = scale.clone()

    def set_residual_mean(self, mean: torch.Tensor):
        """Set per-dim mean for residual centering. mean: [D]"""
        self.residual_mean = mean.clone()

    def q_sample(self, x0, t, noise=None):
        """前向扩散: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t][:, None]
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def loss_per_sample(self, anchor, neighbor):
        batch_size = anchor.size(0)
        device = anchor.device

        x0 = neighbor - anchor
        if self.anchor_scale:
            anchor_norm = anchor.norm(dim=1, keepdim=True).clamp(min=1e-4)
            x0 = x0 / anchor_norm
        # Mean-center
        if self.residual_mean is not None:
            x0 = x0 - self.residual_mean.unsqueeze(0)
        if self.global_scale is not None:
            x0 = x0 / self.global_scale.unsqueeze(0)

        t = torch.randint(0, self.T, (batch_size,), device=device)
        xt, noise = self.q_sample(x0, t)

        # CFG: randomly drop anchor conditioning
        cond_anchor = anchor
        if self.training and self.cfg_prob > 0:
            drop_mask = torch.rand(batch_size, 1, device=device) < self.cfg_prob
            cond_anchor = anchor * (~drop_mask).float()

        pred = self.backbone(xt, t, cond_anchor)

        if self.pred_type == "x0":
            target = x0
        elif self.pred_type == "eps":
            target = noise
        else:  # v-prediction
            alpha_bar_t = self.alpha_bar[t][:, None]
            target = alpha_bar_t.sqrt() * noise - (1 - alpha_bar_t).sqrt() * x0

        # MSE loss
        mse = (pred - target).pow(2).mean(dim=1)

        # min-SNR-γ weighting: balance gradients across timesteps
        if self.min_snr_gamma > 0:
            alpha_bar_t = self.alpha_bar[t]
            snr = alpha_bar_t / (1 - alpha_bar_t).clamp(min=1e-8)  # [B]
            snr_clipped = snr.clamp(max=self.min_snr_gamma)
            if self.pred_type == "x0":
                w = snr_clipped / snr.clamp(min=1e-8)
            elif self.pred_type == "eps":
                w = snr_clipped
            else:  # v-prediction
                w = snr_clipped / (snr + 1).clamp(min=1e-8)
            mse = mse * w

        # Cosine direction loss (only meaningful for x0 prediction)
        if self.cosine_weight > 0 and self.pred_type == "x0":
            cos_sim = nn.functional.cosine_similarity(pred, target, dim=1)
            cos_loss = 1.0 - cos_sim  # [B]
            return mse + self.cosine_weight * cos_loss

        return mse

    def loss_fn(self, anchor, neighbor, weight=None):
        sample_loss = self.loss_per_sample(anchor, neighbor)
        if weight is not None:
            w = weight.detach()
            return (sample_loss * w).sum() / (w.sum() + 1e-8)
        return sample_loss.mean()

    def forward(self, x, t, anchor):
        """前向传播: 预测"""
        return self.backbone(x, t, anchor)


def _denormalize_residual(model, x, anchor, residual_scale=1.0):
    """反归一化残差: global_scale → mean_center → anchor_scale → residual_scale"""
    residual = x
    if model.global_scale is not None:
        residual = residual * model.global_scale.unsqueeze(0)
    if model.residual_mean is not None:
        residual = residual + model.residual_mean.unsqueeze(0)
    if model.anchor_scale:
        anchor_norm = anchor.norm(dim=1, keepdim=True).clamp(min=1e-4)
        residual = residual * anchor_norm
    if residual_scale != 1.0:
        residual = residual * residual_scale
    return residual


class DDPMSampler:
    """
    DDPM 采样器: 从 anchor 生成新样本，支持 Classifier-Free Guidance
    """
    def __init__(self, model, device, cfg_scale=0.0, residual_scale=1.0):
        """
        Args:
            model: 训练好的 ResidualDiffusion 模型
            device: 设备
            cfg_scale: CFG guidance scale (0=no guidance, >0=guided)
            residual_scale: 残差缩放因子 (>1 放大残差增加多样性)
        """
        self.model = model
        self.device = device
        self.T = model.T
        self.cfg_scale = cfg_scale
        self.residual_scale = residual_scale

        # 复制扩散系数
        self.beta = model.beta
        self.alpha = model.alpha
        self.alpha_bar = model.alpha_bar
        self.alpha_bar_prev = torch.cat([
            torch.tensor([1.0], device=device),
            model.alpha_bar[:-1]
        ])

    @torch.no_grad()
    def sample(self, anchor, n_samples=1, **kwargs):
        """
        从 anchor 生成新样本 (支持 CFG)

        Args:
            anchor: [B, D] 锚点嵌入
            n_samples: 每个 anchor 生成的样本数
        Returns:
            [B * n_samples, D] 生成的样本
        """
        batch_size, dim = anchor.shape

        if n_samples > 1:
            anchor = anchor.repeat_interleave(n_samples, dim=0)

        total_samples = anchor.size(0)
        use_cfg = self.cfg_scale > 0 and self.model.cfg_prob > 0
        null_cond = torch.zeros_like(anchor)

        # 初始化: 标准高斯噪声
        x = torch.randn(total_samples, dim, device=self.device)

        pred_type = self.model.pred_type

        for t in reversed(range(self.T)):
            t_batch = torch.full((total_samples,), t, device=self.device, dtype=torch.long)

            if use_cfg:
                pred_cond = self.model(x, t_batch, anchor)
                pred_uncond = self.model(x, t_batch, null_cond)
                pred = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = self.model(x, t_batch, anchor)

            alpha_bar_t = self.alpha_bar[t]
            alpha_t = self.alpha[t]
            beta_t = self.beta[t]

            # 从预测恢复 noise 和 x0
            if pred_type == "x0":
                pred_x0 = pred
                pred_noise = (x - alpha_bar_t.sqrt() * pred_x0) / (1 - alpha_bar_t).sqrt().clamp(min=1e-8)
            elif pred_type == "eps":
                pred_noise = pred
            else:  # v-prediction
                pred_noise = (1 - alpha_bar_t).sqrt() * x + alpha_bar_t.sqrt() * pred

            mu = (x - beta_t / (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_t.sqrt()

            if t > 0:
                alpha_bar_prev_t = self.alpha_bar_prev[t]
                sigma = ((1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * beta_t).sqrt()
                x = mu + sigma * torch.randn_like(x)
            else:
                x = mu

        # 反归一化 residual
        residual = x
        if self.model.global_scale is not None:
            residual = residual * self.model.global_scale.unsqueeze(0)
        # 反 mean-center
        if self.model.residual_mean is not None:
            residual = residual + self.model.residual_mean.unsqueeze(0)
        if self.model.anchor_scale:
            anchor_norm = anchor.norm(dim=1, keepdim=True).clamp(min=1e-4)
            residual = residual * anchor_norm

        # 残差缩放
        if self.residual_scale != 1.0:
            residual = residual * self.residual_scale

        generated = anchor + residual
        return generated


class SingleStepSampler:
    """单步采样器: 从纯噪声一步预测 x0，避免多步误差累积"""

    def __init__(self, model, device, cfg_scale=0.0, residual_scale=1.0):
        self.model = model
        self.device = device
        self.cfg_scale = cfg_scale
        self.residual_scale = residual_scale

    @torch.no_grad()
    def sample(self, anchor, n_samples=1, **kwargs):
        batch_size, dim = anchor.shape
        if n_samples > 1:
            anchor = anchor.repeat_interleave(n_samples, dim=0)
        total_samples = anchor.size(0)
        use_cfg = self.cfg_scale > 0 and self.model.cfg_prob > 0
        pred_type = self.model.pred_type

        x = torch.randn(total_samples, dim, device=self.device)
        t_batch = torch.full((total_samples,), self.model.T - 1, device=self.device, dtype=torch.long)

        if use_cfg:
            null_cond = torch.zeros_like(anchor)
            pred_cond = self.model(x, t_batch, anchor)
            pred_uncond = self.model(x, t_batch, null_cond)
            pred = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
        else:
            pred = self.model(x, t_batch, anchor)

        alpha_bar_t = self.model.alpha_bar[self.model.T - 1]
        if pred_type == "x0":
            pred_x0 = pred
        elif pred_type == "eps":
            pred_x0 = (x - (1 - alpha_bar_t).sqrt() * pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
        else:
            pred_x0 = alpha_bar_t.sqrt() * x - (1 - alpha_bar_t).sqrt() * pred

        residual = _denormalize_residual(self.model, pred_x0, anchor, self.residual_scale)
        return anchor + residual


class DenoiseSampler:
    """SDEdit 采样器: 从部分加噪的零残差开始去噪，利用模型在低 t 的强去噪能力"""

    def __init__(self, model, device, cfg_scale=0.0, residual_scale=1.0,
                 t_start=100, denoise_steps=20):
        self.model = model
        self.device = device
        self.cfg_scale = cfg_scale
        self.residual_scale = residual_scale
        self.t_start = min(t_start, model.T - 1)
        self.denoise_steps = denoise_steps
        self.alpha_bar = model.alpha_bar

    @torch.no_grad()
    def sample(self, anchor, n_samples=1, **kwargs):
        batch_size, dim = anchor.shape
        if n_samples > 1:
            anchor = anchor.repeat_interleave(n_samples, dim=0)
        total_samples = anchor.size(0)
        use_cfg = self.cfg_scale > 0 and self.model.cfg_prob > 0
        pred_type = self.model.pred_type

        # 从零残差加噪到 t_start: x_t = sqrt(alpha_bar_t) * 0 + sqrt(1-alpha_bar_t) * eps
        alpha_bar_start = self.alpha_bar[self.t_start]
        x = (1 - alpha_bar_start).sqrt() * torch.randn(total_samples, dim, device=self.device)

        # 构建从 t_start 到 0 的 DDIM 子序列
        steps = min(self.denoise_steps, self.t_start + 1)
        timesteps = torch.linspace(0, self.t_start, steps, dtype=torch.long, device=self.device).flip(0)
        null_cond = torch.zeros_like(anchor) if use_cfg else None

        for i, t in enumerate(timesteps):
            t_batch = torch.full((total_samples,), t, device=self.device, dtype=torch.long)

            if use_cfg:
                pred_cond = self.model(x, t_batch, anchor)
                pred_uncond = self.model(x, t_batch, null_cond)
                pred = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = self.model(x, t_batch, anchor)

            alpha_bar_t = self.alpha_bar[t]
            if pred_type == "x0":
                pred_x0 = pred
            elif pred_type == "eps":
                pred_x0 = (x - (1 - alpha_bar_t).sqrt() * pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
            else:
                pred_x0 = alpha_bar_t.sqrt() * x - (1 - alpha_bar_t).sqrt() * pred

            pred_x0 = pred_x0.clamp(-10, 10)

            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alpha_bar[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)

            # Deterministic DDIM step
            noise_pred = (x - alpha_bar_t.sqrt() * pred_x0) / (1 - alpha_bar_t).sqrt().clamp(min=1e-8)
            dir_xt = (1 - alpha_bar_prev).clamp(min=0).sqrt()
            x = alpha_bar_prev.sqrt() * pred_x0 + dir_xt * noise_pred

        residual = _denormalize_residual(self.model, x, anchor, self.residual_scale)
        return anchor + residual


class DDIMSampler:
    """DDIM 采样器: 从 T_train 步模型中用 S 步快速采样"""

    def __init__(self, model, device, cfg_scale=0.0, residual_scale=1.0, ddim_steps=50, eta=0.0):
        self.model = model
        self.device = device
        self.T = model.T
        self.cfg_scale = cfg_scale
        self.residual_scale = residual_scale
        self.eta = eta  # 0=deterministic DDIM, 1=DDPM

        # 构建子序列 timesteps
        self.ddim_steps = min(ddim_steps, self.T)
        self.timesteps = torch.linspace(0, self.T - 1, self.ddim_steps, dtype=torch.long, device=device)
        self.alpha_bar = model.alpha_bar

    @torch.no_grad()
    def sample(self, anchor, n_samples=1, **kwargs):
        batch_size, dim = anchor.shape
        if n_samples > 1:
            anchor = anchor.repeat_interleave(n_samples, dim=0)
        total_samples = anchor.size(0)
        use_cfg = self.cfg_scale > 0 and self.model.cfg_prob > 0
        pred_type = self.model.pred_type
        null_cond = torch.zeros_like(anchor)

        x = torch.randn(total_samples, dim, device=self.device)
        timesteps = self.timesteps.flip(0)  # 从大到小

        for i, t in enumerate(timesteps):
            t_batch = torch.full((total_samples,), t, device=self.device, dtype=torch.long)

            if use_cfg:
                pred_cond = self.model(x, t_batch, anchor)
                pred_uncond = self.model(x, t_batch, null_cond)
                pred = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = self.model(x, t_batch, anchor)

            alpha_bar_t = self.alpha_bar[t]

            # 恢复 pred_x0
            if pred_type == "x0":
                pred_x0 = pred
            elif pred_type == "eps":
                pred_x0 = (x - (1 - alpha_bar_t).sqrt() * pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
            else:  # v
                pred_x0 = alpha_bar_t.sqrt() * x - (1 - alpha_bar_t).sqrt() * pred

            pred_x0 = pred_x0.clamp(-10, 10)

            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alpha_bar[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)

            # DDIM update
            sigma = self.eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
            dir_xt = (1 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt()
            noise_pred = (x - alpha_bar_t.sqrt() * pred_x0) / (1 - alpha_bar_t).sqrt().clamp(min=1e-8)
            x = alpha_bar_prev.sqrt() * pred_x0 + dir_xt * noise_pred
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)

        # 反归一化
        residual = x
        if self.model.global_scale is not None:
            residual = residual * self.model.global_scale.unsqueeze(0)
        if self.model.residual_mean is not None:
            residual = residual + self.model.residual_mean.unsqueeze(0)
        if self.model.anchor_scale:
            anchor_norm = anchor.norm(dim=1, keepdim=True).clamp(min=1e-4)
            residual = residual * anchor_norm
        if self.residual_scale != 1.0:
            residual = residual * self.residual_scale

        return anchor + residual


class GaussianGenerator:
    """Ablation: simple Gaussian noise perturbation instead of DDPM."""

    def __init__(self, noise_scale=0.1, relative=True):
        self.noise_scale = noise_scale
        self.relative = relative  # scale noise relative to anchor norm

    def sample(self, anchor, n_samples=1, **kwargs):
        batch_size, dim = anchor.shape
        if n_samples > 1:
            anchor = anchor.repeat_interleave(n_samples, dim=0)
        noise = torch.randn_like(anchor) * self.noise_scale
        if self.relative:
            anchor_norm = anchor.norm(dim=1, keepdim=True).clamp(min=1e-4)
            noise = noise * anchor_norm
        return anchor + noise
