"""
Numerical checks behind the review's statements about T-PAL (Part B, M3-M8).

Restored from the 2 Nov 2026 session on the other computer, where it ran from a scratch folder and
was not copied into the project. Inputs in sections 3-5 are numbers logged in the T-PAL notebook
(source/1791390966256.pdf): the test-set size and accuracy, the epoch-30 loss terms and the test
confusion matrix.

    python tpal_checks.py

Output on 2 Nov 2026 (other computer):
    1) max |A~ - softmax(QK/sqrt(d) - d^2/2s^2)| = 5.36e-07
    2) prior mass within 3 / 6 / 9 tokens of a central query: 0.407 / 0.878 / 0.997
       mean token distance under uniform attention: 8.33 | under prior alone: 3.36
    3) test acc 0.9611, Wilson 95% CI [0.9479, 0.9711]
    4) 0.1*L_assoc = 0.0042, 0.01*L_div = 0.0086, share of total = 4.6%
    5) tumours called 'notumor': 12/810 -> tumour sensitivity 0.9852; healthy called tumour: 5/270
    6) L_div matmul FLOPs per training step at N=256: 25.8 GFLOP; at 256x256 input (N=1024): 1649 GFLOP
"""
import numpy as np
import torch

torch.manual_seed(0)
G, SIG = 16, 3.0
r, c = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
P = torch.stack([r.flatten(), c.flatten()], 1).float()
D = torch.cdist(P, P)
S = torch.exp(-D**2 / (2*SIG**2))

# 1) T-PAL's A*S renormalisation == softmax(logits + log S)
logits = torch.randn(2, 4, G*G, G*G) * 3
A = torch.softmax(logits, -1)
At = A * S; At = At / At.sum(-1, keepdim=True)
At2 = torch.softmax(logits - D**2/(2*SIG**2), -1)
print("1) max |A~ - softmax(QK/sqrt(d) - d^2/2s^2)| =", (At - At2).abs().max().item())

# 2) how local is the prior?
Sn = S / S.sum(-1, keepdim=True)
centre = (G//2)*G + G//2
for rad in [3, 6, 9]:
    print(f"2) prior mass within {rad} tokens of a central query: {Sn[centre][D[centre] <= rad].sum().item():.3f}")
for d in [4, 8, 12, 16]:
    print(f"   logit penalty at distance {d}: {-(d**2)/(2*SIG**2):.2f}  (x{np.exp(-(d**2)/(2*SIG**2)):.1e})")
print("   mean token distance under uniform attention:", round(D.mean().item(), 2),
      "| under prior alone:", round((Sn*D).sum(-1).mean().item(), 2))

# 3) uncertainty of the single test estimate
n, k = 1080, 1038
p = k/n; z = 1.96
den = 1 + z*z/n; centre_w = (p + z*z/(2*n))/den
half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
print(f"3) test acc {p:.4f}, Wilson 95% CI [{centre_w-half:.4f}, {centre_w+half:.4f}]")

# 4) share of the regularisers in the final loss (epoch 30 log)
loss, cls, assoc, div = 0.2799, 0.2671, 0.0418, 0.855
print(f"4) 0.1*L_assoc = {0.1*assoc:.4f}, 0.01*L_div = {0.01*div:.4f}, share of total = {(loss-cls)/loss:.1%}")

# 5) tumour-vs-no-tumour view of the confusion matrix (rows true, cols pred: glioma, meningioma, notumor, pituitary)
cm = np.array([[255,7,6,2],[4,253,5,8],[1,2,265,2],[1,3,1,265]])
fn = cm[[0,1,3],2].sum(); tumours = cm[[0,1,3]].sum()
fp = cm[2,[0,1,3]].sum()
print(f"5) tumours called 'notumor': {fn}/{tumours} -> tumour sensitivity {(tumours-fn)/tumours:.4f}; healthy called tumour: {fp}/270")
print("   glioma<->meningioma confusions:", cm[0,1] + cm[1,0], "| meningioma->pituitary:", cm[1,3])

# 6) cost of L_div: h(h-1)/2 products of NxN matrices per layer per sample
N, h, L, B = G*G, 4, 2, 64
fl = 2*N**3 * h*(h-1)//2 * L * B
print(f"6) L_div matmul FLOPs per training step at N={N}: {fl/1e9:.1f} GFLOP; at 256x256 input (N=1024): {fl*64/1e9:.0f} GFLOP")
