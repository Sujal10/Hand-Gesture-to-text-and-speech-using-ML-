import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import os
from keras.models import load_model
import traceback

# Path to your trained model file
model_path = "Sign-Language-To-Text-and-Speech-Conversion-master\\cnn8grps_rad1_model.h5"
model = load_model(model_path)

# Initialize the video capture
capture = cv2.VideoCapture(0)

# Initialize hand detectors
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Directory for storing processed images for each letter
dataset_dir = "Sign-Language-To-Text-and-Speech-Conversion-master\\AtoZ_3.1"

offset = 30
step = 1
flag = False
suv = 0

# Create a white background image for displaying hand skeleton
white = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("Sign-Language-To-Text-and-Speech-Conversion-master\\white.jpg", white)

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)
        img_final = img_final1 = img_final2 = 0

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            roi = image  # RGB image without drawing

            # Convert to grayscale and apply Gaussian blur for simple preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (1, 1), 2)

            # Binary threshold image
            gray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur2 = cv2.GaussianBlur(gray2, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, test_image = cv2.threshold(th3, 27, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Resize and prepare image for output
            test_image1 = blur
            img_final1 = np.ones((400, 400), np.uint8) * 148
            h, w = test_image1.shape
            img_final1[((400 - h) // 2):((400 - h) // 2) + h, ((400 - w) // 2):((400 - w) // 2) + w] = test_image1

            img_final = np.ones((400, 400), np.uint8) * 255
            h, w = test_image.shape
            img_final[((400 - h) // 2):((400 - h) // 2) + h, ((400 - w) // 2):((400 - w) // 2) + w] = test_image

        hands = hd.findHands(frame, draw=False, flipType=True)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            white = cv2.imread("Sign-Language-To-Text-and-Speech-Conversion-master\\white.jpg")
            handz = hd2.findHands(image, draw=False, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']
                os = ((400 - w) // 2) - 15
                os1 = ((400 - h) // 2) - 15
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)

                # Additional points and connections
                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                cv2.imshow("skeleton", white)
            
            # Draw bounding box for hand
            hands = hd.findHands(white, draw=False, flipType=True)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                cv2.rectangle(white, (x - offset, y - offset), (x + w, y + h), (3, 255, 25), 3)

            # Capture grayscale image for model input
            image1 = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            roi1 = image1
            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.GaussianBlur(gray1, (1, 1), 2)

            test_image2 = blur1
            img_final2 = np.ones((400, 400), np.uint8) * 148
            h, w = test_image2.shape
            img_final2[((400 - h) // 2):((400 - h) // 2) + h, ((400 - w) // 2):((400 - w) // 2) + w] = test_image2

            cv2.imshow("binary", img_final)

        cv2.imshow("frame", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            # Esc key
            break
        if interrupt & 0xFF == ord('n'):
            flag = False
        if interrupt & 0xFF == ord('a'):
            flag = not flag
            suv = 0 if flag else suv

        # Save processed image
        if flag and suv == 50:
            flag = False
        if flag and step % 2 == 0:
            img_path = os.path.join(dataset_dir, "Captured_Images")
            os.makedirs(img_path, exist_ok=True)
            cv2.imwrite(f"{img_path}/image_{suv}.jpg", img_final1)

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()

capture.release()
cv2.destroyAllWindows()






















































# img_final=cv2.resize(img_final,(224,224));
# img_finalf=np.ones((400,400,3),np.uint8)*255;
# print("img final shape= ", img_final)
#  for i in range(400):
#      for j in range(400):
#          if(img_final[i][j]==255):
#              img_finalf[i][j]=[255,255,255]
#          else:
#              img_finalf[i][j]=[0,0,0];
# print("img final f shape= ", img_finalf)
# image = cv2.medianBlur(test_image, 5)
# kernel = np.ones((3, 3), np.uint8)
# kernel1 = np.ones((1, 1), np.uint8)
# dilate = cv2.dilate(image, kernel, iterations=1)
# dilate = cv2.erode(dilate, kernel1, iterations=1)

# cv2.imshow("gray",gray)
# cv2.imshow("blurr",blur)
# cv2.imshow("adapt threshold",th3)
# cv2.imshow("roi",test_image)

# white increase

# if flag:
#     if step % 2 == 0:
#         cv2.imwrite("D:\\sign_data\\B\\b" + str(count) + ".jpg", img_final)
#         print(count)
#         count += 1
#     step += 1