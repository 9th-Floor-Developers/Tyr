import cv2
import mediapipe as mp
from cv2 import VideoCapture
from mediapipe import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import \
	NormalizedLandmark
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)
BUFFER_DISTANCE: float = .15
SCREEN_SIZE = (1920, 960)


def handle_mouse(hand: list[NormalizedLandmark]):
	thumb = hand[4]
	index = hand[8]
	middle = hand[12]
	ring = hand[16]
	
	print(f"{index.y} {thumb.y}")
	
	# if distance(middle, thumb) < BUFFER_DISTANCE:
	# 	print("left click")
	# elif distance(ring, thumb) < BUFFER_DISTANCE:
	# 	print("right click")
	if (index.y < thumb.y):
		print("HI")
		#pyautogui.moveTo(index.x * SCREEN_SIZE[0], index.y * SCREEN_SIZE[1])


def distance(
	landmark1: NormalizedLandmark,
	landmark2: NormalizedLandmark
) -> float:
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
		
		if len(detection_result.hand_landmarks) != 0:
			hand: list[NormalizedLandmark] = detection_result.hand_landmarks[0]
			handle_mouse(hand)
		
		cv2.imshow("Camera Feed", frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
