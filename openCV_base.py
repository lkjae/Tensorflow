import cv2

print("openCV version: ")
print(cv2.__version__)

img = cv2.imread("image/bts-01.jpg")
print("image shape: {} pixels".format(img.shape))
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))
cv2.imshow("bts", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
