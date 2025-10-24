import numpy as np
from qutip import coherent, squeeze, basis, wigner
from PIL import Image
import csv, os

x = np.linspace(-4, 4, 64)
p = np.linspace(-4, 4, 64)

def make_wigner(psi):
    W = wigner(psi, x, p)
    W = (W - W.min()) / (W.max() - W.min())  # scale 0–1
    return (W * 255).astype(np.uint8)

def save_wigner(W, path):
    Image.fromarray(W).save(path)

def generate_dataset(outdir="data/wigner", n_per_class=100):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename","label"])
        for i in range(n_per_class):
            # coherent
            psi = coherent(40, np.random.uniform(-1.5,1.5)+1j*np.random.uniform(-1.5,1.5))
            W = make_wigner(psi)
            save_wigner(W, f"{outdir}/coh_{i}.png")
            writer.writerow([f"coh_{i}.png","coherent"])

            # squeezed
            psi = squeeze(40, np.random.uniform(0.2,1.0)*np.exp(1j*np.random.uniform(0,2*np.pi))) * basis(40,0)
            W = make_wigner(psi)
            save_wigner(W, f"{outdir}/sq_{i}.png")
            writer.writerow([f"sq_{i}.png","squeezed"])

            # cat
            alpha = np.random.uniform(0.8,1.6)
            psi = (coherent(40, alpha) + coherent(40, -alpha)).unit()
            W = make_wigner(psi)
            save_wigner(W, f"{outdir}/cat_{i}.png")
            writer.writerow([f"cat_{i}.png","cat"])
    print("Dataset generated in", outdir)

generate_dataset()
