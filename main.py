import dataclasses
import json
from dataclasses import dataclass

import cv2
import mediapipe as mp
# noinspection PyProtectedMember
from cv2 import VideoCapture
from mediapipe import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import \
	NormalizedLandmark
from mediapipe.tasks.python.vision import face_detector
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

@dataclass
class GestureData:
	is_moving: bool
	is_left_click: bool
	is_right_click: bool

	def reset(self):
		self.is_moving = False
		self.is_left_click = False
		self.is_right_click = False

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)
BUFFER_DISTANCE: float = .05
SCREEN_SIZE = (1920, 960)

gesture_data = GestureData(False, False, False)

def handle_mouse(hand: list[NormalizedLandmark]):
	thumb = hand[4]
	index = hand[8]
	middle = hand[12]
	ring = hand[16]

	if (thumb.x is None) or (index.x is None) or (middle.x is None) or (ring.x is None):
		return

	if (thumb.y is None) or (index.y is None) or (middle.y is None) or (ring.y is None):
		return

	print(f"{index.y} {thumb.y}")

	if distance(middle, thumb) < BUFFER_DISTANCE:
		gesture_data.is_left_click = True
	elif distance(ring, thumb) < BUFFER_DISTANCE:
		gesture_data.is_right_click = True
	if index.y < thumb.y:
		gesture_data.is_moving = True
		#pyautogui.moveTo(index.x * SCREEN_SIZE[0], index.y * SCREEN_SIZE[1])

	save_tasks_data()


def distance(
	landmark1: NormalizedLandmark,
	landmark2: NormalizedLandmark
) -> float:
	if (landmark1.x is None) or (landmark2.x is None):
		return 10000 #If 1 of the 2 numbers are null

	if (landmark1.y is None) or (landmark2.y is None):
		return 10000

	return abs(landmark2.y - landmark1.y) + abs(landmark2.x - landmark1.x)


def main():
	cap: VideoCapture = cv2.VideoCapture(0)

	while True:
		ret, frame = cap.read()

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image: Image = mp.Image(
			image_format=mp.ImageFormat.SRGB, data=frame_rgb
		)

		detection_result: HandLandmarkerResult = detector.detect(mp_image)

		gesture_data.reset()

		if len(detection_result.hand_landmarks) != 0:
			hand: list[NormalizedLandmark] = detection_result.hand_landmarks[0]
			handle_mouse(hand)

		cv2.imshow("Camera Feed", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


def save_tasks_data() -> None:
	file = open("gesture_data/data.txt", "w")
	json.dump(dataclasses.asdict(gesture_data), file)


if __name__ == '__main__':
	main()
