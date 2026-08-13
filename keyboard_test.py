import pyautogui
import time

print("Keyboard controller test")
print("Starting in 3 seconds...")
time.sleep(3)

print("Holding D for 2 seconds...")
pyautogui.keyDown("d")
time.sleep(2)
pyautogui.keyUp("d")

time.sleep(1)

print("Holding A for 2 seconds...")
pyautogui.keyDown("a")
time.sleep(2)
pyautogui.keyUp("a")

print("Test complete.")