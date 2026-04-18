import pyautogui
from pynput import keyboard
import time

# Safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

def on_press(key):
	try:
		# Press ESC to stop script
		if key.char == 'q':
			print("Stopping...")
			return False
	except:
		pass

# Example movement loop
def demo_mouse_movement():
	screen_width, screen_height = pyautogui.size()
	
	for i in range(100):
		x = int(screen_width * (i / 100))
		y = int(screen_height / 2)
		
		pyautogui.moveTo(x, y)
		time.sleep(0.01)
	
	pyautogui.click()

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

demo_mouse_movement()
listener.join()
