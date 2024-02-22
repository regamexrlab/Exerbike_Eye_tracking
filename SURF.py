import cv2

# Load the template image
template = cv2.imread('/Users/kaviprakashramalingam/Desktop/Exerbike Python script/test.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if template is None:
    print("Error: Failed to load the template image")
    exit()

# Initialize the video capture
video_capture = cv2.VideoCapture('/Users/kaviprakashramalingam/Desktop/Exerbike Python script/testvideo.mp4')

# Get the width and height of the template image
w, h = template.shape[::-1]

# Initialize the method for template matching
method = cv2.TM_CCORR_NORMED

# Loop through the frames
while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply template matching
    result = cv2.matchTemplate(gray_frame, template, method)

    # Get the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Draw a rectangle around the best match
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
