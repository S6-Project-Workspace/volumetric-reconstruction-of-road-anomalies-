import cv2

l = cv2.imread("data/dataset1/rgb/left.png")
r = cv2.imread("data/dataset1/rgb/right.png")

print(l is not None, r is not None)
print(l.shape, r.shape)
