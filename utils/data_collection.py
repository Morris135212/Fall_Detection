import time

import cv2
from IPython.display import display, clear_output
from PIL import Image

DATA_SHAPE = (640, 480)


def save_images(frame, path, i):
    idx = str(i)
    idx = "".join(["0" for _ in range(6-len(idx))]) + idx
    save_path = f"{path}_{idx}.png"
    frame.save(save_path)


def collect_data(path):
    cam = cv2.VideoCapture(0) # start the video feed
    i = 0
    try:
        while True:
            success, frame_array = cam.read()  # read frame from camera feed
            if not success:
                raise Exception("Camera initialization failed or was not released from previous stream. Restart kernel.")

            frame_array = cv2.resize(frame_array, DATA_SHAPE)
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame_array)
            display(frame)
            # get user input
            val = input("Save image? (y/n) or continue to next class? (c) or exit? (x)")

            # clear current frame
            clear_output(wait=True)
            print(flush=True)
            if val == "y":
                save_images(frame, path, i)
                i += 1
            elif val == 'x':
                break  # continue to next class or exit
            time.sleep(5)
        cam.release()  # release camera
        print("Data collection finished.")
    except KeyboardInterrupt:
        print("Data collection finished.")
        cam.release()