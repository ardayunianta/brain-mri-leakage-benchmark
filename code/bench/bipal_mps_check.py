"""Does the project's real bipal.py train on the Apple GPU (MPS), and how fast compared with CPU?

usage: python bipal_mps_check.py code    (path to the folder that holds bipal.py)
"""
import json
import sys
import time

import torch

sys.path.insert(0, sys.argv[1])
import bipal  # noqa: E402

B = 64
out = {}
for name, kw in [("bipal_full", {}), ("tpal_like", bipal.ABLATIONS["tpal_like"])]:
    for dev in ("cpu", "mps"):
        torch.manual_seed(0)
        m = bipal.BiPAL(nc=4, img=128, depth=2 if name == "tpal_like" else 4, **kw).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
        x = torch.randn(B, 1, 128, 128, device=dev)
        y = torch.randint(0, 4, (B,), device=dev)
        masks = torch.zeros(B, 1, 128, 128, device=dev)
        masks[:, :, 40:70, 50:80] = 1
        has = (torch.arange(B, device=dev) % 2).float()

        def step():
            opt.zero_grad(set_to_none=True)
            logits, aux = m(x)
            loss, parts = bipal.bipal_loss(m, logits, aux, x, y, masks, has)
            loss.backward()
            opt.step()
            return loss

        try:
            for _ in range(2):
                step()
            if dev == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                loss = step()
            loss.item()
            if dev == "mps":
                torch.mps.synchronize()
            ms = (time.perf_counter() - t0) / 5 * 1000
            finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in m.parameters())
            out[f"{name}/{dev}"] = {"train_step_ms": round(ms, 1), "loss": round(loss.item(), 4), "grads_finite": finite}
        except Exception as e:  # an op missing on MPS shows up here
            out[f"{name}/{dev}"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
print(json.dumps(out, indent=1))
