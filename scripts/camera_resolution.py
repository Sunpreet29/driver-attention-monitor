import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("The camera resolution is", end=" ")
print(f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH)} * {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
cap.release()
cv2.destroyAllWindows()