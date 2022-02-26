import cv2 

img_path = "image.png"
txt_path = "bounding_box.txt"

img = cv2.imread(img_path)
lines = []
points = []


with open(txt_path, "r") as file:
    lines = [line for line in file] 

for line in lines:
    if line[-1] == '\n':
        line = line[:-1]
    points = line.split(" ")
    points_int = list(map(int, points))
    print(points_int)
    cv2.rectangle(img, (points_int[0], points_int[1]), (points_int[2], points_int[3]), (0, 0, 255), thickness=3)


cv2.imwrite("hw0_110550088_1.png", img)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()