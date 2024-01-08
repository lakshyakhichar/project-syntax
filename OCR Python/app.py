import pytesseract
import cv2
from PIL import Image
import numpy as np
pytesseract.pytesseract.tesseract_cmd = "D:\\Tesseract\\tesseract.exe"


def rotate_to_horizontal(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Use HoughLinesP to find lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Calculate the angle of the most dominant line
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append(angle)

    # Calculate the most dominant angle
    dominant_angle = np.median(angles)

    # Rotate the image to align text horizontally
    rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), np.degrees(dominant_angle), 1), (image.shape[1], image.shape[0]))

    return rotated_image


def get_text(img_name):
    image = cv2.imread(img_name)
    rotated_img = rotate_to_horizontal(image)
    cv2.imshow("rotated image", rotated_img)
    cv2.waitKey(100)
    cv2.imwrite("rotated_image.jpg", rotated_img)
    grey_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
    inv_img = cv2.bitwise_not(grey_img)
    thresh, bw_img = cv2.threshold(inv_img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", bw_img)
    cv2.waitKey(1110)
    text = pytesseract.image_to_string(bw_img)
    return text


output = get_text("image1.jpeg")


print(output)
