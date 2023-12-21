import easyocr
import cv2
import time

img_path = "C:\\Users\\mekal\\Downloads\\basler.png"
img = cv2.imread(img_path)
# img = cv2.resize(img,(500,500))
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img,(7,7),1)
# cv2.imshow('stop',img)
reader = easyocr.Reader(['en'])
result = reader.readtext(img, detail = 1,paragraph=True)
result_1 = reader.readtext(img,detail =0)
# print(type(result))
print(result_1)
for (coord, text) in result:

  (topleft, topright, bottomright, bottomleft) = coord
  tx,ty = (int(topleft[0]), int(topleft[1]))
  bx,by = (int(bottomright[0]), int(bottomright[1]))
  cv2.rectangle(img, (tx,ty), (bx,by), (0, 0, 255), 2)
  cv2.putText(img,text, (tx, ty),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

print("The extracted word is ",result_1,end="\n")
# cv2.putText(img, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
cv2.imshow('basler',img)
# cv2.imwrite('C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\output_folder\\80.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

