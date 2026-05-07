import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ====================================================
# 0. Settings
# ====================================================

# image_path = "CHEST_sid 180_110kv 8mas_grid O.raw"
# image_path = "zFail_chest 86kv 200ma 8mas (grid).raw"
image_path = "zFail_chest 75kv 200ma 5mas.raw"

width = 3072
height = 3072

# RAW display parameters
low_percentile = 0.5
high_percentile = 99.5
gamma = 0.30          # 0.25, 0.30, 0.35 중에서 조정 가능
contrast = 1.05
brightness = 0.00

# Bone mask parameters
tau_c = 0.10          # coherency threshold
tau_g = 0.02          # gradient threshold
min_size = 80

# Suppression parameter
alpha = 0.30          # 뼈 억제 강도. 0.20~0.45 사이에서 조정 가능


# ====================================================
# 1. Load RAW image
# ====================================================

raw = np.fromfile(image_path, dtype=np.uint16)

if raw.size != width * height:
    raise ValueError(
        f"RAW size mismatch: got {raw.size}, expected {width * height}. "
        f"Check width, height, dtype, or file path."
    )

raw = raw.reshape((height, width))
raw = raw.astype(np.float32)

# print("RAW loaded")
# print("raw dtype:", raw.dtype)
# print("raw shape:", raw.shape)
# print("raw min:", raw.min())
# print("raw max:", raw.max())
# print("raw percentiles:", np.percentile(raw, [0.5, 1, 5, 50, 95, 99, 99.5]))


# ====================================================
# 2. Windowing + gamma correction
# ====================================================

low, high = np.percentile(raw, (low_percentile, high_percentile))

img = np.clip(raw, low, high)
img = (img - low) / (high - low + 1e-8)

# 반전은 사용하지 않음
# img = 1.0 - img

# gamma < 1이면 어두운 영역이 밝아짐
img = np.power(img, gamma)

# contrast / brightness fine tuning
img = contrast * (img - 0.5) + 0.5 + brightness
img = np.clip(img, 0, 1)

img = img.astype(np.float32)
img = np.ascontiguousarray(img)

# print("Processed image")
# print("img dtype:", img.dtype)
# print("img min:", img.min())
# print("img max:", img.max())


# ====================================================
# 3. CED substitute: edge-preserving smoothing
# ====================================================
# 원래 PDF의 CED 대신 bilateral filter 사용
# img가 0~1 float32이므로 sigmaColor는 0.05~0.15 정도가 적절함

ced = cv2.bilateralFilter(
    img,
    d=9,
    sigmaColor=0.08,
    sigmaSpace=75
)

ced = ced.astype(np.float32)
ced = (ced - ced.min()) / (ced.max() - ced.min() + 1e-8)


# ====================================================
# 4. Gradient computation
# ====================================================

Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

grad_mag = np.sqrt(Ix**2 + Iy**2)
grad_mag = grad_mag / (grad_mag.max() + 1e-8)


# ====================================================
# 5. Structure tensor and coherency map
# ====================================================

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


# ====================================================
# 6. Bone mask generation
# ====================================================
# 방향성이 강하고, gradient도 어느 정도 있는 부분만 bone 후보로 선택

bone_mask = (coherency > tau_c) & (grad_mag > tau_g)

bone_mask_uint8 = bone_mask.astype(np.uint8)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    bone_mask_uint8,
    connectivity=8
)

clean_mask = np.zeros_like(bone_mask_uint8)

for i in range(1, num_labels):  # 0번은 배경
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_size:
        clean_mask[labels == i] = 1

# Morphological closing
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
bone_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
bone_mask = bone_mask.astype(np.float32)


# ====================================================
# 7. Bone component estimation
# ====================================================
# coherency만 쓰면 너무 넓게 잡히므로 gradient를 함께 사용

bone_component = coherency * grad_mag * bone_mask
bone_component = cv2.GaussianBlur(bone_component.astype(np.float32), (0, 0), 2)
bone_component = bone_component / (bone_component.max() + 1e-8)


# ====================================================
# 8. Bone suppression
# ====================================================
# 현재 영상에서는 뼈가 밝은 edge 성분으로 잡히므로 bone_component를 빼는 방식 사용

soft = img - alpha * bone_component
soft = np.clip(soft, 0, 1)

# 비교용 blur-mixing 방식이 필요하면 아래를 대신 사용 가능
# soft_mask = cv2.GaussianBlur(bone_mask, (0, 0), 5)
# soft_mask = soft_mask / (soft_mask.max() + 1e-8)
# blurred = cv2.GaussianBlur(img, (0, 0), 12)
# soft = img * (1 - alpha * soft_mask) + blurred * (alpha * soft_mask)
# soft = np.clip(soft, 0, 1)


# ====================================================
# 9. Visualization
# ====================================================

plt.close("all")

plt.figure(figsize=(14, 9))

titles = [
    "Original",
    "CED substitute",
    "Gradient magnitude",
    "Coherency map",
    "Estimated bone component",
    "Bone-suppressed image"
]

images = [
    img,
    ced,
    grad_mag,
    coherency,
    bone_component,
    soft
]

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray", vmin=0, vmax=1)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


# ====================================================
# 10. Save outputs
# ====================================================

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

output_files = [
    output_dir / "original_windowed.png",
    output_dir / "ced_result.png",
    output_dir / "gradient_magnitude.png",
    output_dir / "coherency_map.png",
    output_dir / "bone_component.png",
    output_dir / "bone_suppressed.png"
]

# 기존 png 삭제 후 새로 저장
for file in output_files:
    if file.exists():
        file.unlink()

cv2.imwrite(str(output_dir / "original_windowed.png"), (img * 255).astype(np.uint8))
cv2.imwrite(str(output_dir / "ced_result.png"), (ced * 255).astype(np.uint8))
cv2.imwrite(str(output_dir / "gradient_magnitude.png"), (grad_mag * 255).astype(np.uint8))
cv2.imwrite(str(output_dir / "coherency_map.png"), (coherency * 255).astype(np.uint8))
cv2.imwrite(str(output_dir / "bone_component.png"), (bone_component * 255).astype(np.uint8))
cv2.imwrite(str(output_dir / "bone_suppressed.png"), (soft * 255).astype(np.uint8))

print("Saved outputs to:", output_dir.resolve())
print("Done")