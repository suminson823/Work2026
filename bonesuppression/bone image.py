import numpy as np
import matplotlib.pyplot as plt

image_path = "CHEST_sid 180_110kv 8mas_grid O.raw"

width = 3072
height = 3072

raw = np.fromfile(image_path, dtype=np.uint16)
raw = raw.reshape((height, width))
raw = raw.astype(np.float32)

low, high = np.percentile(raw, (0.5, 99.5))

img = np.clip(raw, low, high)
img = (img - low) / (high - low + 1e-8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("No invert")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(1.0 - img, cmap="gray")
plt.title("Invert")
plt.axis("off")

plt.tight_layout()
plt.show()