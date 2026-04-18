import cv2


def main():
	# Open webcam (0 = default camera)
	cap = cv2.VideoCapture(0)
	
	while True:
		# Read one frame from camera
		ret, frame = cap.read()
		
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
