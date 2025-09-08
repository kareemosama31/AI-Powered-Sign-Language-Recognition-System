# ------------------------------
# ASL Real-Time Webcam Recognition (Python 3.13 compatible)
# ------------------------------

import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ------------------------------
# 1. Load the trained model
# ------------------------------
model = tf.keras.models.load_model("asl_letters_numbers_model.h5")

# ------------------------------
# 2. Class labels mapping
# ------------------------------
labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
          10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j',
          20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't',
          30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'}

# ------------------------------
# 3. Initialize webcam and parameters
# ------------------------------
cap = cv2.VideoCapture(0)
img_size = 64
pred_queue = deque(maxlen=5)  # smoothing over last 5 predictions

# ------------------------------
# 4. Real-time loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror effect

    # ROI for hand
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # ------------------------------
    # Hand detection using skin color
    # ------------------------------
    # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    # upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    # mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # hand_area = cv2.countNonZero(mask)

    # Draw ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert ROI to grayscale and blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 7
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:
            x, y, w, h = cv2.boundingRect(max_contour)
            pad = 20
            x1_crop = max(x - pad, 0)
            y1_crop = max(y - pad, 0)
            x2_crop = min(x + w + pad, roi.shape[1])
            y2_crop = min(y + h + pad, roi.shape[0])
            hand_img = roi[y1_crop:y2_crop, x1_crop:x2_crop]
            hand_img = cv2.resize(hand_img, (img_size, img_size))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            prediction = model.predict(hand_img)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            pred_queue.append(predicted_class)
            smooth_class = max(set(pred_queue), key=pred_queue.count)

            if confidence > 0.5:
                label = labels[smooth_class]
                cv2.putText(frame, f'{label} ({confidence*100:.1f}%)', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No confident prediction', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Draw bounding box
            cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            cv2.putText(frame, 'No hand detected', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'No hand detected', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Threshold", thresh)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
