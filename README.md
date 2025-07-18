# 👁️ Real-Time Face Detection using OpenCV

This Python script uses OpenCV's Haar Cascade Classifier to detect faces from a webcam in real time. It draws rectangles around detected faces and shows the video in a live window.


---

## 🧠 Full Python Code with Explanation (When PC contain Web-Camera)

```python
import cv2

# Load the pre-trained Haar cascade for frontal face detection
face_cap = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Try to open the default webcam (index 0)
video_cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, video_data = video_cap.read()

    # If frame not captured properly, skip to next iteration
    if not ret or video_data is None:
        print("Failed to capture video frame.")
        break

    # Convert the frame to grayscale (required by Haar cascades)
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the video frame with rectangles
    cv2.imshow("video_live", video_data)

    # Break the loop if 'a' key is pressed
    if cv2.waitKey(10) == ord("a"):
        break

# Release the camera and close any open windows
video_cap.release()
cv2.destroyAllWindows()
```
_________________________________________________________________________________________________

> 📌 If your PC doesn't have a camera, you can easily modify it to use a video or image file instead.

## 🧠 Full Python Code with Explanation (When PC have no Web-Camera)
```python
import cv2  # Import the OpenCV library for image processing

# Load the Haar cascade classifier for frontal face detection
# Make sure the file 'haarcascade_frontalface_default.xml' is in the same directory or provide the full path
face_cap = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load an image from the file system
# Replace "test.jpg" with the path to your actual image file
image = cv2.imread("test.jpg")

# Check if the image was successfully loaded
if image is None:
    print("Error: Image not found or could not be loaded.")
else:
    # Convert the image to grayscale (Haar cascades work on grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the loaded Haar cascade
    # Parameters:
    # - scaleFactor: Specifies how much the image size is reduced at each scale (1.1 = 10% reduction per step)
    # - minNeighbors: How many neighbors each candidate rectangle should have to retain it
    # - minSize: Minimum size of detected faces
    # - flags: Specifies the detection mode
    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Loop over each detected face and draw a green rectangle around it
    for (x, y, w, h) in faces:
        # (x, y) is the top-left coordinate, (x+w, y+h) is the bottom-right
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with rectangles drawn around detected faces
    cv2.imshow("Face Detection", image)

    # Wait for any key to be pressed before closing the image window
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

```


## 📁 Requirements
* `Python 3.x`
* `opencv(pip install opencv-python)`
* `haarcascade_frontalface_default.xml` - this file is exist in `cv2` file when `opencv` is installed in the `PC`.. 