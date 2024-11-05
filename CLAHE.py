
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the original image
#路徑可能要改一下
image_path = R'C:\Users\user\Desktop\meeting\Comparison of Pre-Trained YOLO Models on Steel Surface\remove-reflection\img1.png'
source_image = cv2.imread(image_path)

# Step 2: Convert the image to grayscale for brightness thresholding
gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

# Step 3: Create a mask using both brightness and color
# 3.1: Brightness-based mask (detect bright areas)
_, brightness_mask = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)

# 3.2: Color-based mask (detecting strong highlights)
# Convert to HSV and threshold the saturation and value channels to detect reflective highlights
hsv_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
lower_saturation = np.array([0, 0, 230])
upper_saturation = np.array([180, 40, 255])
color_mask = cv2.inRange(hsv_image, lower_saturation, upper_saturation)

# Combine both masks to improve reflection detection
combined_mask = cv2.bitwise_or(brightness_mask, color_mask)

# Step 4: Refine the mask using morphological operations
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

# Step 5: Invert the mask (so reflective areas are 1, and others are 0)
mask_inverted = cv2.bitwise_not(mask_cleaned)

# Step 6: Apply Poisson blending (seamlessClone)
mask_3ch = cv2.merge([mask_inverted, mask_inverted, mask_inverted])
center = (source_image.shape[1] // 2, source_image.shape[0] // 2)

# Perform Poisson fusion (seamless cloning)
poisson_result = cv2.seamlessClone(source_image, source_image, mask_3ch, center, cv2.NORMAL_CLONE)

# Step 7: If reflection still exists, apply inpainting to remaining areas
# Detect any remaining bright spots in the Poisson result
gray_poisson = cv2.cvtColor(poisson_result, cv2.COLOR_BGR2GRAY)
_, remaining_mask = cv2.threshold(gray_poisson, 220, 255, cv2.THRESH_BINARY)

# Inpaint remaining reflective areas
final_image = cv2.inpaint(poisson_result, remaining_mask, 3, cv2.INPAINT_TELEA)

# Step 8: Display the images using matplotlib
# Convert BGR images to RGB for displaying correctly with matplotlib
source_image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
poisson_result_rgb = cv2.cvtColor(poisson_result, cv2.COLOR_BGR2RGB)
final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

# Plot the images
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(source_image_rgb)
plt.title('Source Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mask_cleaned, cmap='gray')
plt.title('Combined Mask')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(poisson_result_rgb)
plt.title('Poisson Fusion Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(final_image_rgb)
plt.title('Final Image (Inpainted)')
plt.axis('off')

plt.tight_layout()
plt.show()


"""增強圖像對比度：
CLAHE 可以顯著提高圖像的局部對比度，這對於在低對比度或光照條件不佳的環境下（例如夜間拍攝、醫學圖像等）特別有效。
通過局部增強圖像的對比度，物體的細節會變得更加明顯。
2. 避免過度增強噪聲：
相比於自適應直方圖均衡化（AHE），CLAHE 在增強對比度的同時，能有效限制過度放大的噪聲。
在AHE中，圖像中低亮度區域的噪聲有可能被過度增強，而CLAHE 會通過設置對比度的增強上限來抑制這些噪聲的放大，從而保持圖像的質量。
3. 自適應性強：
CLAHE 將圖像分成若干局部區域，對每個區域分別進行直方圖均衡化，因此對於有不均勻亮度的圖像，
CLAHE可以在不同的區域中增強對比度，這比全局直方圖均衡化更靈活。例如，一張有陰影和光斑的圖像可以在陰影區域增強亮度，而不影響其他區域。
4. 適合於多種應用：
CLAHE 廣泛應用於多個領域，例如：
醫學圖像處理：增強X光、CT或MRI圖像中的細節，有助於提高診斷的準確性。
衛星圖像處理：強化衛星圖像中的細節，使得目標物（如建築物、道路、地形等）更加清晰。
工業檢測：在自動化檢測系統中，增強產品表面的細節，有助於檢測缺陷或異常。
5. 改善物體檢測和識別：
在計算機視覺應用中，CLAHE 增強了物體的邊緣和細節，這可以提高物體檢測和識別的準確性。
對於使用YOLO等模型進行物體檢測的系統，CLAHE 可以幫助模型更好地識別低對比度或光照不均的物體。
6. 靈活調整：
CLAHE 允許用戶通過兩個參數靈活調整其效果：
clipLimit：控制對比度增強的限制。較高的 clipLimit 會允許更多的對比度增強，較低的值會抑制過度增強。
tileGridSize：決定圖像被分割成的小區塊的大小。較小的區塊會進行更細粒度的局部對比度增強，較大的區塊則更接近於全局均衡化。
7. 保留自然細節：
由於CLAHE是局部處理圖像的對比度，它保留了更多的自然細節，避免全局均衡化帶來的過度扭曲和不自然感。
這使得CLAHE在增強圖像的同時，不會引入明顯的視覺瑕疵。"""

"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖像
image = cv2.imread('C:/Users/user/Desktop/meeting/Comparison of Pre-Trained YOLO Models on Steel Surface/_MG_5035.jpeg')

# 將圖像轉換為灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 應用高斯平滑（Gaussian Blur）
gaussian_blur_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

# 應用 AHE (自適應直方圖均衡化)
ahe = cv2.equalizeHist(gaussian_blur_image)

# 應用 CLAHE (對比度限制自適應直方圖均衡化)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gaussian_blur_image)

# 顯示原圖、高斯平滑圖像、AHE 和 CLAHE 結果
plt.figure(figsize=(12, 8))

# 原圖
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# 高斯平滑結果
plt.subplot(1, 4, 2)
plt.imshow(gaussian_blur_image, cmap='gray')
plt.title('Gaussian Blur')
plt.axis('off')

# AHE 結果
plt.subplot(1, 4, 3)
plt.imshow(ahe, cmap='gray')
plt.title('AHE Result')
plt.axis('off')

# CLAHE 結果
plt.subplot(1, 4, 4)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE Result')
plt.axis('off')

plt.show()
"""


