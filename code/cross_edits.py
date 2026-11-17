"""
CROSS edit operators (counterfactual completion of the label x source table), on 128 x 128 uint8 slices.

- transplant(host, donor, dmask): paste the donor's tumour (figshare, with mask) into a host slice (other-source
  no-tumour), at the donor lesion's relative position inside the host's brain box, intensity-matched to the host's
  brain tissue and blended with a feathered alpha. Label: the donor's class.
- transplant_control(...): the same alpha shape, placed at the same host position, but filled from a healthy region
  of the donor (same shape, shifted off the lesion). Label: no tumour. Same seams and blending as the real paste,
  so "was edited" cannot predict the label.
- erase(img, mask): fill the dilated tumour mask by harmonic (Laplace) inpainting plus noise matched to the ring
  around it. Label: no tumour (soft in training).
- erase_control(img, mask): the same fill applied to a same-shape region elsewhere in the brain. Label unchanged.

    python code/cross_edits.py      # writes results/cross_pilot/edit_examples.png
"""
import numpy as np
from scipy import ndimage as ndi

DILATE = 3        # px at 128 x 128
FEATHER = 1.0     # Gaussian sigma of the alpha edge


def brain_mask(img):
    """Rough brain/head foreground: smoothed intensity above 10% of its 99th percentile, largest component, filled."""
    s = ndi.gaussian_filter(img.astype(np.float32), 1.5)
    fg = s > 0.10 * np.percentile(s, 99)
    lab, n = ndi.label(fg)
    if n == 0:
        return np.ones_like(img, bool)
    big = 1 + int(np.argmax(ndi.sum(fg, lab, range(1, n + 1))))
    return ndi.binary_fill_holes(lab == big)


def inner_brain(img, erode=7):
    """The head mask eroded, as a proxy for brain tissue away from scalp and skull."""
    b = brain_mask(img)
    e = ndi.binary_erosion(b, iterations=erode)
    return e if e.sum() > 200 else b


def _bbox(m):
    ys, xs = np.where(m)
    return ys.min(), ys.max(), xs.min(), xs.max()


def _shift(m, dy, dx):
    out = np.zeros_like(m)
    H, W = m.shape
    ys, xs = np.where(m)
    ys, xs = ys + dy, xs + dx
    ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    out[ys[ok], xs[ok]] = True
    return out


def _tissue_stats(img, brain, exclude=None):
    sel = brain if exclude is None else brain & ~exclude
    v = img[sel].astype(np.float32)
    if v.size < 50:
        v = img.astype(np.float32).ravel()
    return float(v.mean()), float(v.std() + 1e-3)


def _healthy_offset(lesion, brain, rng, tries=40):
    """An offset that moves the lesion shape onto brain tissue that does not touch the lesion."""
    ring = ndi.binary_dilation(lesion, iterations=4)
    for _ in range(tries):
        dy, dx = rng.integers(-40, 41, 2)
        sm = _shift(lesion, dy, dx)
        if sm.sum() >= 0.9 * lesion.sum() and not (sm & ring).any() and (sm & brain).sum() >= 0.9 * sm.sum():
            return dy, dx
    return None


def _place(host_brain, donor_brain, lesion, rng):
    """Offset that puts the donor lesion's centroid at the same relative position inside the host's brain box."""
    hy0, hy1, hx0, hx1 = _bbox(host_brain)
    dy0, dy1, dx0, dx1 = _bbox(donor_brain)
    cy, cx = ndi.center_of_mass(lesion)
    for _ in range(10):
        ry = (cy - dy0) / max(dy1 - dy0, 1) + rng.normal(0, 0.04)
        rx = (cx - dx0) / max(dx1 - dx0, 1) + rng.normal(0, 0.04)
        ty, tx = hy0 + ry * (hy1 - hy0), hx0 + rx * (hx1 - hx0)
        dy, dx = int(round(ty - cy)), int(round(tx - cx))
        sm = _shift(lesion, dy, dx)
        if sm.sum() >= 0.95 * lesion.sum() and (sm & host_brain).sum() >= 0.95 * sm.sum():
            return dy, dx
    return None


def _paste(host, donor, src_region, dy, dx, host_brain, donor_brain, donor_lesion):
    """Paste donor pixels of src_region, moved by (dy, dx), intensity-matched host <- donor tissue statistics."""
    region = ndi.binary_dilation(src_region, iterations=DILATE)
    hm, hs = _tissue_stats(host, host_brain)
    dm, ds = _tissue_stats(donor, donor_brain, exclude=donor_lesion)
    patch = (donor.astype(np.float32) - dm) / ds * hs + hm
    patch = np.roll(np.roll(patch, dy, 0), dx, 1)
    alpha = ndi.gaussian_filter(_shift(region, dy, dx).astype(np.float32), FEATHER)
    out = host.astype(np.float32) * (1 - alpha) + patch * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def transplant(host, donor, dmask, rng):
    """Returns (edited image, new lesion mask) or None if no plausible placement exists."""
    lesion = dmask > 0
    if lesion.sum() < 4:
        return None
    hb, db = inner_brain(host), inner_brain(donor)
    off = _place(hb, db, lesion, rng)
    if off is None:
        return None
    dy, dx = off
    return _paste(host, donor, lesion, dy, dx, hb, db, lesion), _shift(lesion, dy, dx).astype(np.uint8)


def transplant_control(host, donor, dmask, rng):
    """Same shape, same host position as transplant, but filled from healthy donor tissue."""
    lesion = dmask > 0
    if lesion.sum() < 4:
        return None
    hb, db = inner_brain(host), inner_brain(donor)
    off = _place(hb, db, lesion, rng)
    hoff = _healthy_offset(lesion, db, rng)
    if off is None or hoff is None:
        return None
    healthy = _shift(lesion, *hoff)                       # healthy donor region with the lesion's shape
    dy, dx = off[0] - hoff[0], off[1] - hoff[1]           # move it to where the lesion would have gone
    return _paste(host, donor, healthy, dy, dx, hb, db, lesion)


def _harmonic_fill(img, region, iters=400):
    f = img.astype(np.float32).copy()
    f[region] = f[~region].mean()
    for _ in range(iters):
        avg = 0.25 * (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1))
        f[region] = avg[region]
    return f


def _fill(img, region, rng):
    f = _harmonic_fill(img, region)
    ring = ndi.binary_dilation(region, iterations=4) & ~region
    sd = float(img[ring].astype(np.float32).std()) if ring.any() else 0.0
    noise = ndi.gaussian_filter(rng.normal(0, 1, img.shape).astype(np.float32), 1.2)
    noise = noise / (noise.std() + 1e-6) * 0.25 * sd
    alpha = ndi.gaussian_filter(region.astype(np.float32), FEATHER)
    out = img.astype(np.float32) * (1 - alpha) + (f + noise) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def erase(img, mask, rng):
    lesion = mask > 0
    if lesion.sum() < 4:
        return None
    return _fill(img, ndi.binary_dilation(lesion, iterations=DILATE), rng)


def erase_control(img, mask, rng):
    lesion = mask > 0
    if lesion.sum() < 4:
        return None
    off = _healthy_offset(lesion, inner_brain(img), rng)
    if off is None:
        return None
    return _fill(img, ndi.binary_dilation(_shift(lesion, *off), iterations=DILATE), rng)


if __name__ == "__main__":
    import os
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_checkpoints import ROOT, load_img, load_mask
    k = pd.read_csv(os.path.join(ROOT, "results/audit/grouped_folds.csv"))
    rng = np.random.default_rng(0)
    tum = k[k.mask_path.notna()].sample(4, random_state=1)
    hosts = k[k.label == "notumor"].sample(4, random_state=2)
    fig, axs = plt.subplots(4, 6, figsize=(12, 8.4))
    for i, ((_, t), (_, h)) in enumerate(zip(tum.iterrows(), hosts.iterrows())):
        x = load_img(os.path.join(ROOT, "data/brain-tumor-mri-dataset", t.rel_path))
        m = load_mask(os.path.join(ROOT, "results/audit/masks", os.path.basename(t.mask_path)))
        hx = load_img(os.path.join(ROOT, "data/brain-tumor-mri-dataset", h.rel_path))
        tp = transplant(hx, x, m, rng)
        tc = transplant_control(hx, x, m, rng)
        panels = [(x, f"donor ({t.label})"), (hx, "host (no tumour)"), (tp[0] if tp else hx, "transplant"),
                  (tc if tc is not None else hx, "transplant control"), (erase(x, m, rng), "erase"),
                  (erase_control(x, m, rng), "erase control")]
        for j, (im, ttl) in enumerate(panels):
            axs[i, j].imshow(im if im is not None else x, cmap="gray", vmin=0, vmax=255)
            axs[i, j].set_title(ttl, fontsize=8); axs[i, j].axis("off")
    os.makedirs(os.path.join(ROOT, "results/cross_pilot"), exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results/cross_pilot/edit_examples.png"), dpi=110)
    print("saved results/cross_pilot/edit_examples.png")
