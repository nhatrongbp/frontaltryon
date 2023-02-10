import mediapipe as mp
# from rotate import *
from tkinter import *
from detector import *
from PIL import Image, ImageTk

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("girl.mp4")
# write mp4 file
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
# result = cv2.VideoWriter('filename.mp4',
#                          cv2.VideoWriter_fourcc(*'MP4V'),
#                          10, size)
# result = cv2.VideoWriter('filename.avi',
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)

# kit selection
current_kit = "homekit/"

# Create a GUI app
app = Tk()
app.title('Frontal spotify try-on')
# Bind the app with Escape keyboard to
# quit app whenever pressed
app.bind('<Escape>', lambda e: app.quit())
# Create label
l = Label(app, text="FRONTAL TRY-ON")
l.config(font=("TkDefaultFont", 24))
l.grid(column=0, row=0, rowspan=1, columnspan=3)
# Create label
l = Label(app, text="Frontally try our new 2022/2023 Spotify kit online")
l.config(font=("TkDefaultFont", 12))
l.grid(column=0, row=1, rowspan=1, columnspan=3)
# Create a label and display it on app
label_widget = Label(app)
label_widget.grid(column=0, row=2, rowspan=2, columnspan=3)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)


def my_main_loop():
    global cap
    if cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            cap = cv2.VideoCapture("girl.mp4")
            success, image = cap.read()
        # print("new frame")
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        # )
        image = check_landmark(results, image, current_kit)
        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        # result.write(image)
        opencv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        label_widget.photo_image = photo_image
        label_widget.configure(image=photo_image)
        label_widget.after(10, my_main_loop)
        # cv2.imshow('MediaPipe Pose', image)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break


def set_home_kit():
    global current_kit
    current_kit = "homekit/"


def set_away_kit():
    global current_kit
    current_kit = "awaykit/"


def set_third_kit():
    global current_kit
    current_kit = "thirdkit/"


photo1 = PhotoImage(file=r"homekit\baby.png")
photoimage1 = photo1.subsample(3, 3)
button1 = Button(app, text="Home baby kit", image=photoimage1, compound=TOP, command=set_home_kit)
button1.grid(column=0, row=4)

photo2 = PhotoImage(file=r"awaykit\baby.png")
photoimage2 = photo2.subsample(3, 3)
button2 = Button(app, text="Away baby kit", image=photoimage2, compound=TOP, command=set_away_kit)
button2.grid(column=1, row=4)

photo3 = PhotoImage(file=r"thirdkit\smallbaby.png")
photoimage3 = photo3.subsample(3, 3)
button3 = Button(app, text="Third baby kit", image=photoimage3, compound=TOP, command=set_third_kit)
button3.grid(column=2, row=4)

# Create label
l = Label(app, text="Available on https://store.spotify.es/22-23-barca-kit")
l.config(font=("TkDefaultFont", 12))
l.grid(column=0, row=5, rowspan=1, columnspan=3)

my_main_loop()
app.mainloop()
cap.release()
# result.release()
