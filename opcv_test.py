import cv2

img = cv2.imread("./image/ieu-01.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("iu", img)
cv2.imshow("iu - gray", gray_img)

cv2.waitKey(0)  #아무키를 누를 때까지 delay
cv2.destroyAllWindows()