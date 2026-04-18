import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
	num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def main():
	# Open webcam (0 = default camera)
	cap = cv2.VideoCapture(0)
	
	while True:
		# Read one frame from camera
		ret, frame = cap.read()
		
		
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
		
		detection_result = detector.detect(mp_image)
		
		if not ret:
			break
		
		# Show raw frame in a window
		cv2.imshow("Camera Feed", frame)
		
		# Wait 1ms and check if 'q' is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	# Cleanup
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
