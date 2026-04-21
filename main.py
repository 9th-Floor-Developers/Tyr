import threading
import time

import cv2
import mediapipe as mp
from cv2 import VideoCapture
from mediapipe import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import \
	NormalizedLandmark
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

import pyautogui
import math

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)
BUFFER_DISTANCE: float = .15
SCREEN_SIZE = (1920, 1080)
SMOOTHING = .8
prev_x, prev_y = 0, 0


def handle_mouse(hand: list[NormalizedLandmark]):
	thumb = hand[4]
	index = hand[8]
	middle = hand[12]
	ring = hand[16]
	
	if not any(
			[
				thumb.x, thumb.y,
				index.x, index.y,
				middle.x, middle.y,
				ring.x, ring.y
			]
	):
		return
	
	global prev_x, prev_y
	
	# print(f"{index.y} {thumb.y}")
	
	if distance(middle, thumb) < BUFFER_DISTANCE:  # left click
		pass
		# pyautogui.mouseDown(button='left')
	elif distance(ring, thumb) < BUFFER_DISTANCE:  # right click
		pass
		# pyautogui.mouseDown(button='right')
	
	if index.y < thumb.y:  # movement
		target_x = (1 - index.x) * SCREEN_SIZE[0]
		target_y = index.y * SCREEN_SIZE[1]
		
		smooth_x = prev_x + (target_x - prev_x) * SMOOTHING
		smooth_y = prev_y + (target_y - prev_y) * SMOOTHING
		
		pyautogui.moveTo(smooth_x, smooth_y)
		
		prev_x, prev_y = smooth_x, smooth_y


def distance(
	landmark1: NormalizedLandmark,
	landmark2: NormalizedLandmark
) -> float:
	if not any([landmark1.x, landmark1.y, landmark2.x, landmark2.y]):
		return math.inf
	
	return abs(landmark2.y - landmark1.y) + abs(landmark2.x - landmark1.x)


def detect_hand(mp_image: Image) -> None:
	detection_result: HandLandmarkerResult = detector.detect(mp_image)
	
	if len(detection_result.hand_landmarks) != 0:
		hand: list[NormalizedLandmark] = detection_result.hand_landmarks[0]
		handle_mouse(hand)


def main() -> None:
	cap: VideoCapture = cv2.VideoCapture(0)
	
	threads: list[Thread] = []
	
	while True:
		_, frame = cap.read()
		
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image: Image = mp.Image(
			image_format=mp.ImageFormat.SRGB, data=frame_rgb
		)
		
		thread = threading.Thread(target=detect_hand, args=(mp_image,))
		threads.append(thread)
		thread.start()
		
		while len(threads) > 20:
			threads.remove(threads[0])
		
		cv2.imshow("Camera Feed", frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
