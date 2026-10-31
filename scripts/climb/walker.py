"""walker.py: a depth-based forward walk into one painting (Depth Anything V2 Small, z-buffer splat, hole fill).

Every pixel of the painting is a point at the depth Depth Anything V2 Small (Apache-2.0) reads for it; a step moves the
eye toward the point the path leads to, nearer points draw over farther ones, and the holes a step opens are filled
from their surroundings. The model is downloaded from Hugging Face on first use; it needs PyTorch and Transformers
(`pip install torch "transformers<5"`) and runs on the Mac's GPU (MPS) when there is one, otherwise on the CPU."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline

W, H, SS = 720, 1280, 2
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
_est = None


def depth(img: Image.Image) -> np.ndarray:
    global _est
    _est = _est or pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=DEV)
    d = _est(img)["predicted_depth"].squeeze().float()
    d = F.interpolate(d[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    d = np.clip((d - np.percentile(d, 2)) / (np.percentile(d, 99.5) - np.percentile(d, 2)), 0, 1)
    return (1.0 / (0.06 + 0.94 * d)).astype(np.float32)          # near ~1, far ~16


def hard_edges(z: np.ndarray, img: Image.Image, k: int = 21, slope: float = 0.05) -> np.ndarray:
    """Depth edges made hard, guided by colour (31 Oct 2026: the forest's near pines tore into specks and streaks as
    the days walked on, because the depth map's soft edges left pixels floating between a tree and the sky behind it).
    In a band around each steep change of disparity (a jump, not the ground's steady slope), every pixel takes the
    near side's or the far side's disparity, whichever side its colour is closer to: a dark branch goes with the pine,
    a patch of sky with the sky."""
    d = torch.from_numpy(1.0 / z)[None, None]
    c = torch.from_numpy(np.asarray(img.resize((W, H), Image.LANCZOS), dtype=np.float32) / 255).permute(2, 0, 1)[None]
    g = torch.zeros_like(d)
    g[..., :, 1:-1] = (d[..., :, 2:] - d[..., :, :-2]).abs() / 2
    g[..., 1:-1, :] = torch.maximum(g[..., 1:-1, :], (d[..., 2:, :] - d[..., :-2, :]).abs() / 2)
    band = F.max_pool2d(((g / d) > slope).float(), k, 1, k // 2) > 0
    near = (d > (F.max_pool2d(d, k, 1, k // 2) - F.max_pool2d(-d, k, 1, k // 2)) / 2).float()

    def mean(x, m):
        return F.avg_pool2d(x * m, k, 1, k // 2) / F.avg_pool2d(m, k, 1, k // 2).clamp(min=1e-6)

    nc, fc = mean(c, near), mean(c, 1 - near)
    to_near = ((c - nc) ** 2).sum(1, keepdim=True) <= ((c - fc) ** 2).sum(1, keepdim=True)
    out = torch.where(band, torch.where(to_near, mean(d, near), mean(d, 1 - near)), d)
    return (1.0 / out)[0, 0].numpy().astype(np.float32)


class Walker:
    def __init__(self, img: Image.Image, pp=(0.53, 0.575), plate: Image.Image | None = None, edges: bool = False):
        """`plate`: the same painting without its near objects, behind it (what a step reveals). `edges`: hard depth
        edges (hard_edges) and holes filled from their far side (the forest of the climb's week 1)."""
        self.edges = edges
        img = img.convert("RGB").resize((W, H), Image.LANCZOS)
        layers = [(img, depth(img), 1.0)] + ([(plate.convert("RGB").resize((W, H), Image.LANCZOS), None, 1.06)] if plate else [])
        xs, zs, cs = [], [], []
        yy, xx = torch.meshgrid(torch.arange(H * SS, device=DEV, dtype=torch.float32) / SS,
                                torch.arange(W * SS, device=DEV, dtype=torch.float32) / SS, indexing="ij")
        for im, z, bias in layers:
            z = depth(im) if z is None else z
            if edges:
                z = hard_edges(z, im)
            t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(DEV)
            t = F.interpolate(t, scale_factor=SS, mode="bicubic", align_corners=False).clamp(0, 1)
            zt = F.interpolate(torch.from_numpy(z)[None, None].to(DEV), scale_factor=SS, mode="bilinear", align_corners=False)
            xs.append(torch.stack([xx, yy], -1).reshape(-1, 2)); zs.append(zt.reshape(-1) * bias)
            cs.append(t[0].permute(1, 2, 0).reshape(-1, 3))
        self.xy, self.z, self.c = torch.cat(xs), torch.cat(zs), torch.cat(cs)
        self.n_main = zs[0].numel()                                  # the painting's points; the plate's follow
        self.pp = torch.tensor([pp[0] * W, pp[1] * H], device=DEV)

    def _splat(self, part: slice, tz: float):
        """One layer's points after a step of `tz`: the nearest per pixel (colour, how many, depth). Fixed sizes."""
        z0, xy, c = self.z[part], self.xy[part], self.c[part]
        z2 = z0 - tz
        ok = z2 > 0.12
        z2 = torch.where(ok, z2, torch.ones_like(z2))
        p = self.pp + (xy - self.pp) * (z0 / z2)[:, None]
        xi, yi = p[:, 0].round().long(), p[:, 1].round().long()
        ok = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = torch.where(ok, yi * W + xi, torch.full_like(xi, H * W))
        z2 = torch.where(ok, z2, torch.full_like(z2, 1e9))
        zbuf = torch.full((H * W + 1,), 1e9, device=DEV).scatter_reduce(0, idx, z2, reduce="amin")
        front = (ok & (z2 <= zbuf[idx] * 1.02)).float()
        acc = torch.zeros((H * W + 1, 3), device=DEV).index_add_(0, idx, c * front[:, None])[:H * W]
        cnt = torch.zeros((H * W + 1,), device=DEV).index_add_(0, idx, front)[:H * W]
        frame = (acc / cnt.clamp(min=1)[:, None]).reshape(H, W, 3).permute(2, 0, 1)[None]
        return frame, cnt.reshape(1, 1, H, W), zbuf[:H * W].reshape(1, 1, H, W)

    def _render_edges(self, tz: float) -> Image.Image:
        """render() with `edges`: the plate shows only where the painting left a gap (never over it: a plate's tree
        a little nearer than the painting's sky would show through), and what is still a gap takes an average leaning
        on its farther neighbours (weight (z/16)^2), never a nearer tree's colour smeared across."""
        frame, cnt, z = self._splat(slice(0, self.n_main), tz)
        # A pinhole in a near surface spreading as it comes closer (most of its 3x3 drawn) is that surface: closed from
        # its neighbours first, so the plate never stipples through the near ground.
        drawn = (cnt > 0).float()
        share = F.avg_pool2d(drawn, 3, stride=1, padding=1)
        pin = (cnt == 0) & (share >= 5 / 9)
        frame = torch.where(pin, F.avg_pool2d(frame * drawn, 3, stride=1, padding=1) / share.clamp(min=1e-6), frame)
        z = torch.where(pin, F.avg_pool2d(z * drawn, 3, stride=1, padding=1) / share.clamp(min=1e-6), z)
        cnt = torch.where(pin, torch.ones_like(cnt), cnt)
        if self.n_main < self.z.numel():
            pf, pc, pz = self._splat(slice(self.n_main, None), tz)
            use = (cnt == 0) & (pc > 0)
            frame, cnt, z = torch.where(use, pf, frame), torch.where(use, pc, cnt), torch.where(use, pz, z)
        gap = cnt == 0
        wgt = torch.where(gap, torch.zeros(1, device=DEV), (z.clamp(max=16) / 16) ** 2)
        for r in (1, 2, 3, 5, 8, 13, 21):
            num = F.avg_pool2d(frame * wgt, 2 * r + 1, stride=1, padding=r)
            den = F.avg_pool2d(wgt, 2 * r + 1, stride=1, padding=r)
            got = gap & (den > 1e-9)
            frame = torch.where(got, num / den.clamp(min=1e-9), frame)
            wgt = torch.where(got, den, wgt)
            gap = gap & ~got
        return Image.fromarray((frame[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    def render(self, tz: float) -> Image.Image:
        """The view after a step of `tz` toward `pp`. Every tensor keeps the same size in every frame (a point not
        drawn goes to one extra 'nowhere' pixel): with a different size each frame the Mac's GPU kept a compiled graph
        per size, ~7 MB a frame (30 Oct 2026: 5.2 GB for the 30-day climb; the same output, flat memory now)."""
        if self.edges:
            return self._render_edges(tz)
        z2 = self.z - tz
        ok = z2 > 0.12
        z2 = torch.where(ok, z2, torch.ones_like(z2))
        p = self.pp + (self.xy - self.pp) * (self.z / z2)[:, None]
        xi, yi = p[:, 0].round().long(), p[:, 1].round().long()
        ok = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = torch.where(ok, yi * W + xi, torch.full_like(xi, H * W))
        z2 = torch.where(ok, z2, torch.full_like(z2, 1e9))
        zbuf = torch.full((H * W + 1,), 1e9, device=DEV).scatter_reduce(0, idx, z2, reduce="amin")
        front = (ok & (z2 <= zbuf[idx] * 1.02)).float()
        acc = torch.zeros((H * W + 1, 3), device=DEV).index_add_(0, idx, self.c * front[:, None])[:H * W]
        cnt = torch.zeros((H * W + 1,), device=DEV).index_add_(0, idx, front)[:H * W]
        frame = (acc / cnt.clamp(min=1)[:, None]).reshape(H, W, 3).permute(2, 0, 1)[None]
        hole = (cnt == 0).reshape(1, 1, H, W).float()
        for r in (1, 2, 3, 5, 8, 13):
            k = 2 * r + 1
            num = F.avg_pool2d(frame * (1 - hole), k, stride=1, padding=r)
            den = F.avg_pool2d(1 - hole, k, stride=1, padding=r)
            got = (den > 1e-6).float() * hole
            frame = frame * (1 - got) + num / den.clamp(min=1e-6) * got
            hole = hole * (1 - got)
        return Image.fromarray((frame[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
