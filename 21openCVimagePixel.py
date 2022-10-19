import cv2

print("openCV version: ")
print(cv2.__version__)

img = cv2.imread("image/bts-02.jpg")
print("image shape: {} pixels".format(img.shape))
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))
cv2.imshow("bts", img)

(b,g,r) = img[0,0]
print("Pixel at (0,0) - Red: {}, Green: {}, Blue: {}".format(r,g,b))
cv2.waitKey(0)

dot = img[50:100, 200:400]
cv2.imshow("Dot", dot)

cv2.waitKey(0)

img[50:100, 200:400] = (255,0,0)
img[100:200, 200:400] = (100,100,0)

cv2.imshow("bts - dootted", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
