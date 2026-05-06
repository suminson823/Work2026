import numpy as np
import matplotlib.pyplot as plt

image_path = "CHEST_sid 180_110kv 8mas_grid O.raw"

width = 3072
height = 3072

raw = np.fromfile(image_path, dtype=np.uint16)
raw = raw.reshape((height, width))
raw = raw.astype(np.float32)

low, high = np.percentile(raw, (0.5, 99.5))
base = np.clip(raw, low, high)
base = (base - low) / (high - low + 1e-8)

gammas = [0.35, 0.45, 0.60, 0.80, 0.25, 0.3]

plt.figure(figsize=(14, 8))

for i, gamma in enumerate(gammas):
    img_test = np.power(base, gamma)
    
    plt.subplot(2, 3, i + 1)
    plt.imshow(img_test, cmap="gray", vmin=0, vmax=1)
    plt.title(f"gamma = {gamma}")
    plt.axis("off")

plt.tight_layout()
plt.show()