import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

def rotatedRect2Points(rotatedRect):
  center, sizes, angle = rotatedRect
  angle = np.deg2rad(angle)
  b = np.cos(angle)/2
  a = np.sin(angle)/2
  x,y = center
  w,h = sizes
  v1 = int(x - a*h - b*w), int(y + b*h - a*w)
  v2 = int(x + a*h - b*w), int(y - b*h - a*w)
  v3 = int(2*x - v1[0]), int(2*y - v1[1])
  v4 = int(2*x - v2[0]), int(2*y - v2[1])
  return v1, v2, v3, v4

img_file = sys.argv[1]
img = cv2.imread(img_file)

aspect_ratio = img.shape[1] / img.shape[0]
size = 500
img = cv2.resize(img, (int(size*aspect_ratio), size))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

_, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

for c in cnts:
  peri = cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, 0.02*peri, True)
  if len(approx) == 4:
    displayCnt = approx
    break

cnt_img = img.copy()
cv2.drawContours(cnt_img, [displayCnt], 0, (0,255,0), 5)

M = cv2.getPerspectiveTransform(np.array(displayCnt.reshape(4,2), dtype='float32'), np.float32([[0,0], [0, 500], [500, 500], [500, 0]]))
warped = cv2.warpPerspective(gray, M, (500, 500))

_, thresh = cv2.threshold(warped, 128, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
base = thresh
for i in range(3):
  eroded = cv2.erode(base, kernel)
  dilated = cv2.dilate(eroded, kernel)
  base = dilated

l,r,t,b = 500, 0, 0, 500
for i in range(base.shape[0]):
  for j in range(base.shape[1]):
    if base[j][i]:
      l = min(l, j)
      r = max(r, j)
      t = max(t, i)
      b = min(b, i)

M2 = cv2.getPerspectiveTransform(np.array([[t,l], [t,r], [b,r], [b,l]], dtype='float32'), np.float32([[500,0], [500, 500], [0, 500], [0,0]]))
warped2 = cv2.warpPerspective(base, M2, (500,500))

_, warped_cnts, _ = cv2.findContours(warped2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
big_cnts = list(filter(lambda c: cv2.contourArea(c) > 2000, warped_cnts))
warped_cnt_img = np.zeros_like(warped2)
cv2.fillPoly(warped_cnt_img, big_cnts, 1)
warped_cnt_img = cv2.bitwise_and(warped_cnt_img, warped2)

l,r,t,b = 500, 0, 0, 500
for i in range(warped_cnt_img.shape[0]):
  for j in range(warped_cnt_img.shape[1]):
    if warped_cnt_img[j][i]:
      l = min(l, j)
      r = max(r, j)
      t = max(t, i)
      b = min(b, i)

M3 = cv2.getPerspectiveTransform(np.array([[t,l], [t,r], [b,r], [b,l]], dtype='float32'), np.float32([[500,0], [500, 500], [0, 500], [0,0]]))
warped3 = cv2.warpPerspective(warped_cnt_img, M3, (500,500))

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
base = warped3
for i in range(3):
  dilated2 = cv2.dilate(base, kernel)
  eroded2 = cv2.erode(dilated2, kernel)
  base = eroded2
warped4 = base

_, num_cnts, _ = cv2.findContours(warped4.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
minRects = []
for c in num_cnts:
  minRects.append(cv2.minAreaRect(c))

nums_img = cv2.cvtColor(warped3.astype('float32'), cv2.COLOR_GRAY2BGR)
verts_list = [rotatedRect2Points(rect) for rect in minRects]
nums = []
for verts in verts_list:
  for i in range(len(verts)):
    cv2.line(nums_img, verts[i], verts[(i+1)%4], (0,1,0), thickness=3)
  Mn = cv2.getPerspectiveTransform(np.array(verts).astype('float32'), np.float32([[500,0], [500,500], [0,500], [0,0]]))
  num = cv2.warpPerspective(warped3, Mn, (500,500))
  nums.append(num)

M5 = cv2.getPerspectiveTransform(np.array(verts_list[1]).astype('float32'), np.float32([[500,0], [500,500], [0,500], [0,0]]))
num2 = cv2.warpPerspective(warped3, M5, (500,500))

fig = plt.figure(1)

ax1 = plt.subplot(251)
ax1.imshow(img)

ax2 = plt.subplot(252)
ax2.imshow(edged)

ax3 = plt.subplot(253)
ax3.imshow(cnt_img)

ax4 = plt.subplot(254)
ax4.imshow(warped)

ax5 = plt.subplot(255)
ax5.imshow(thresh)

ax8 = plt.subplot(256)
ax8.imshow(nums_img)

for i,num in enumerate(nums):
  ax = plt.subplot(2,5,7+i)
  ax.imshow(num)

plt.show()
