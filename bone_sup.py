# import itk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, binary_closing, disk

# 1. Load image
import numpy as np

image_path = "zFail_chest 86kv 200ma 8mas (grid).raw"

img = np.fromfile(image_path, dtype=np.uint16)

# image_path = "CHEST_sid 180_110kv 8mas_grid O.raw"

width = 3072
height = 3072

# img = np.fromfile('zFail_chest 86kv 200ma 8mas (grid).raw', dtype=np.uint16)
img = img.reshape((height, width))
img = img.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min())

# normalize
img = img.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

# img = img.astype(np.float32)
# img = (img - img.min()) / (img.max() - img.min() + 1e-8)

# 2. CED 대신 Gaussian + edge-preserving
ced = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
ced = (ced - ced.min()) / (ced.max() - ced.min() + 1e-8)

# 3. Gradient
Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

grad_mag = np.sqrt(Ix**2 + Iy**2)
grad_mag = grad_mag / (grad_mag.max() + 1e-8)

# 4. Structure tensor
sigma = 5
Jxx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma)
Jxy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma)
Jyy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma)

trace = Jxx + Jyy
det = Jxx * Jyy - Jxy * Jxy

temp = np.sqrt(np.maximum(trace**2 - 4 * det, 0))

mu1 = 0.5 * (trace + temp)
mu2 = 0.5 * (trace - temp)

coherency = ((mu1 - mu2) / (mu1 + mu2 + 1e-8)) ** 2
coherency = coherency / (coherency.max() + 1e-8)

# 5. Bone mask
bone_mask = (coherency > 0.05)

bone_mask = remove_small_objects(bone_mask.astype(bool), min_size=80)
bone_mask = binary_closing(bone_mask, disk(2))
bone_mask = bone_mask.astype(np.float32)

# 6. Bone component / soft mask
bone_component = coherency * bone_mask
bone_component = bone_component / (bone_component.max() + 1e-8)

# bone 위치를 부드럽게 만든 mask
soft_mask = cv2.GaussianBlur(bone_mask, (0, 0), 5)
soft_mask = soft_mask / (soft_mask.max() + 1e-8)

# 7. Bone suppression
# 원본보다 훨씬 많이 blur된 이미지 생성
blurred = cv2.GaussianBlur(img, (0, 0), 12)

alpha = 0.7

# 뼈 위치만 blurred 이미지로 섞기
soft = img * (1 - alpha * soft_mask) + blurred * (alpha * soft_mask)
soft = np.clip(soft, 0, 1)

# 8. Visualization
plt.figure(figsize=(12, 8))

titles = [
    "Original", "CED", "Gradient",
    "Coherency", "Bone", "Suppressed"
]

images = [img, ced, grad_mag, coherency, bone_component, soft]

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# 9. Save
cv2.imwrite("ced_result.png", (ced * 255).astype(np.uint8))
cv2.imwrite("bone_component.png", (bone_component * 255).astype(np.uint8))
cv2.imwrite("bone_suppressed.png", (soft * 255).astype(np.uint8))