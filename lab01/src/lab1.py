import cv2

img = cv2.imread("photo.jpg")
print("shape:", img.shape)
print("dtype:", img.dtype)
print("min:", img.min())
print("max:", img.max())