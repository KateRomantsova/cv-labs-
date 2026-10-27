import cv2
import matplotlib.pyplot as plt

img = cv2.imread("photo.jpg")

# Версия 1: без перетворення (цвета искажены)
plt.figure()
plt.imshow(img)
plt.title("Без перетворення (BGR показано як RGB)")
plt.axis("off")
plt.savefig("step2_bgr.png", dpi=150)

# Версия 2: после BGR -> RGB (цвета правильные)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img_rgb)
plt.title("Після cv2.COLOR_BGR2RGB")
plt.axis("off")
plt.savefig("step2_rgb.png", dpi=150)

plt.show()