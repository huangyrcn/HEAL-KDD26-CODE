"""
双曲流形实现 - 数值稳定版本

Lorentz 模型数值稳定性问题：
  - expmap0 使用 sinh/cosh，当 ||v||/sqrt(k) > 50 时溢出
  - 解决方案：对大范数使用渐近近似 sinh(x) ≈ exp(x)/2
"""

import torch
from geoopt.manifolds.lorentz import math as L

# 数值稳定阈值
# Old code clamped v_norm to 50, which with sqrt(k)=2 gives theta_max=25.
# sinh(25) ≈ 3.6e10, still well within float32 range (3.4e38).
# We use the same effective limit for backward compatibility.
_SINH_THRESHOLD = 25.0
_EPS = 1e-7


class LorentzManifold:
    """Lorentz 双曲面 - 数值稳定实现"""

    name = "lorentz"

    def l_inner(self, x, y, keepdim=False):
        """Lorentz 内积: -x0*y0 + x1*y1 + ... + xn*yn"""
        return L.inner(x, y, keepdim=keepdim)

    def proj(self, x, k):
        """投影到流形：确保 -x0^2 + ||x_space||^2 = -k"""
        k = self._ensure_c(k, x.device)
        return L.project(x, k=k)

    def expmap0(self, v, k):
        """原点指数映射 - 数值稳定版本

        标准公式 (geoopt):
          x0 = sqrt(k) * cosh(||v||/sqrt(k))
          x_space = sqrt(k) * sinh(||v||/sqrt(k)) * v_space/||v||

        数值稳定：当 ||v||/sqrt(k) > threshold 时，clamp 输入避免溢出
        """
        k = self._ensure_c(k, v.device)
        sqrt_k = k.sqrt()

        # v = [v0, v_space]，v0 应该是 0（原点切空间）
        v_space = v[..., 1:]
        v_norm = v_space.norm(dim=-1, keepdim=True).clamp(min=_EPS)

        # theta = ||v_space|| / sqrt(k)
        theta = v_norm / sqrt_k

        # Clamp theta 避免 sinh/cosh 溢出
        # sinh(20) ≈ 2.4e8, sinh(30) ≈ 5.3e12, sinh(40) ≈ 1.2e17
        safe_theta = theta.clamp(max=_SINH_THRESHOLD)

        # 计算 sinh/cosh
        sinh_theta = torch.sinh(safe_theta)
        cosh_theta = torch.cosh(safe_theta)

        # 构建流形上的点 (注意：x_space 也要乘 sqrt_k)
        x0 = sqrt_k * cosh_theta
        x_space = sqrt_k * sinh_theta * v_space / v_norm

        x = torch.cat([x0, x_space], dim=-1)
        return self.proj(x, k)

    def logmap0(self, x, k):
        """原点对数映射 - 数值稳定版本"""
        k = self._ensure_c(k, x.device)
        sqrt_k = k.sqrt()

        x0 = x[..., :1]
        x_space = x[..., 1:]
        x_space_norm = x_space.norm(dim=-1, keepdim=True).clamp(min=_EPS)

        # theta = arccosh(x0/sqrt(k))
        # 数值稳定：arccosh(x) = log(x + sqrt(x^2 - 1))
        ratio = (x0 / sqrt_k).clamp(min=1.0 + _EPS)  # arccosh 定义域 [1, inf)
        theta = torch.acosh(ratio)

        # v_space = sqrt(k) * theta * x_space / ||x_space||
        # The geodesic distance is d = sqrt(k) * theta, and the tangent vector
        # should have norm equal to the geodesic distance.
        v_space = sqrt_k * theta * x_space / x_space_norm

        # v0 = 0（原点切空间）
        v0 = torch.zeros_like(x0)
        return torch.cat([v0, v_space], dim=-1)

    def dist(self, x, y, k):
        """测地距离 - 数值稳定版本

        d(x,y) = sqrt(k) * arccosh(-<x,y>_L / k)
        """
        k = self._ensure_c(k, x.device)
        sqrt_k = k.sqrt()

        inner = self.l_inner(x, y)
        # -<x,y>_L / k >= 1 对于流形上的点
        ratio = (-inner / k).clamp(min=1.0 + _EPS)
        return sqrt_k * torch.acosh(ratio)

    def ptransp(self, x, y, v, k):
        """平行传输"""
        k = self._ensure_c(k, x.device)
        return L.parallel_transport(x, y, v, k=k)

    def proj_tan0(self, v, k):
        """投影到原点切空间"""
        k = self._ensure_c(k, v.device)
        o = torch.zeros_like(v)
        o[..., 0] = k.sqrt()
        return L.project_u(o, v, k=k)

    def mobius_matvec(self, m, x, k):
        """Lorentz 线性变换"""
        d = x.size(-1) - 1
        v = self.logmap0(x, k)
        v_space = v.narrow(-1, 1, d) @ m.transpose(-1, -2)
        mv = torch.cat([v.narrow(-1, 0, 1), v_space], dim=-1)
        return self.expmap0(mv, k)

    def centroid(self, weights, x, k):
        """Lorentz 质心"""
        k = self._ensure_c(k, x.device)
        sum_x = torch.spmm(weights, x)
        inner = self.l_inner(sum_x, sum_x).abs().clamp(min=1e-6)
        return (k.sqrt() / inner.sqrt()).unsqueeze(-1) * sum_x

    def _ensure_c(self, c, device):
        if isinstance(c, torch.Tensor):
            return c.to(device)
        return torch.tensor(c, dtype=torch.float32, device=device)


class EuclideanManifold:
    """欧氏空间 — logmap0/expmap0 为恒等映射，dist 为 L2 距离"""

    name = "euclidean"

    def expmap0(self, v, k):
        return v

    def logmap0(self, x, k):
        return x

    def proj(self, x, k):
        return x

    def dist(self, x, y, k):
        return (x - y).norm(dim=-1)

    def l_inner(self, x, y, keepdim=False):
        return (x * y).sum(dim=-1, keepdim=keepdim)

    def centroid(self, weights, x, k):
        return torch.spmm(weights, x)


def get_manifold(name):
    if name.lower() in ("lorentz", "lorentzian"):
        return LorentzManifold()
    if name.lower() == "euclidean":
        return EuclideanManifold()
    raise ValueError(f"Unknown manifold: {name}")
