import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv2.imread('../static/images/Test5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

# Apply filter and find edges for localization # ปรับใช้ตัวกรอง และค้นหาขอบเพื่อตำแหน่ง

bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

# Find Contours and Apply Mask # ค้นหาเส้นรอบรูป และมาร์ค
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
mask = np.zeros(gray.shape, np.uint8)
if location is not None:
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
else:
    largest_area = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = 4 * w * h
        if area > largest_area:
            largest_area = area
            largest_area_contour = c
    new_image = cv2.drawContours(mask, [largest_area_contour], 0, 255, -1)

# Apply OCR # ใช้งาน OCR เพื่ออ่านข้อความ
new_image = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
reader = easyocr.Reader(['th', 'en'])
result = reader.readtext(cropped_image)
text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
                  color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

# Show Results # แสดงผลลัพธ์  และบันทึกไฟล์
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
with open('%s.txt' % text, 'a', encoding="utf-8") as f:
    for i in (result):
        if type(i[-2]) == int:
            f.write(str(i[-2]))
        else:
            f.write(i[-2] + '\n')

cv2.imwrite('%s.jpg' % (text), cropped_image)
