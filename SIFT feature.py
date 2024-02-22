import cv2

# Load the template image
template = cv2.imread('/Users/kaviprakashramalingam/Desktop/Exerbike Python script/test.jpg')

# Check if the image is loaded successfully
if template is None:
    print("Error: Failed to load the template image")
    exit()

# Convert the template image to grayscale
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Convert the template image to 8-bit unsigned integer depth
template_8u = cv2.convertScaleAbs(template_gray)

# Initialize the SIFT detector
detector = cv2.SIFT_create()

# Detect keypoints and compute descriptors for the template image
keypoints_template, descriptors_template = detector.detectAndCompute(template_8u, None)

# Check if keypoints and descriptors are computed successfully
# if keypoints_template is None:
#     print("Error: Failed to compute keypoints and descriptors")
#     exit()

# Initialize the feature matcher (e.g., Brute-Force matcher, FLANN)
matcher = cv2.BFMatcher()

# Initialize the video capture
video_capture = cv2.VideoCapture('/Users/kaviprakashramalingam/Desktop/Exerbike Python script/testoutput.mp4')

# Loop through the frames
while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the current frame
    keypoints_frame, descriptors_frame = detector.detectAndCompute(gray_frame, None)

    # Match descriptors between the template and the current frame
    matches = matcher.match(descriptors_template, descriptors_frame)

    # Apply ratio test to filter good matches
    good_matches = [m for m in matches if m.distance < 0.5 * min(len(matches), 100)]

    # Extract matched keypoints
    matched_keypoints_template = [keypoints_template[m.queryIdx] for m in good_matches]
    matched_keypoints_frame = [keypoints_frame[m.trainIdx] for m in good_matches]

    # Calculate transformation matrix using RANSAC
    if len(good_matches) > 4:
        src_pts = np.float32([kp.pt for kp in matched_keypoints_template]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp.pt for kp in matched_keypoints_frame]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        M = None

    # If transformation matrix is found, update ROI
    if M is not None:
        # Update the ROI using perspective transform
        landmark_corners = np.float32([[0, 0], [0, template.shape[0]], [template.shape[1], template.shape[0]], [template.shape[1], 0]]).reshape(-1, 1, 2)
        landmark_corners_transformed = cv2.perspectiveTransform(landmark_corners, M)
        x, y, w, h = cv2.boundingRect(landmark_corners_transformed)
        landmark_roi = {'x': x, 'y': y, 'width': w, 'height': h}

        # Draw rectangle around the landmark ROI
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
