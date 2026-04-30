import cv2
import numpy as np

video = cv2.VideoCapture('KandinskyBook.mp4')
ret, frame = video.read()
# show first frame
cv2.imshow("First Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Global variable to store corner points
corners = []

# Mouse callback function
def select_corners(event, x, y, flags, param):
    global corners
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x, y))
        print(f"Corner {len(corners)} selected at ({x}, {y})")

# Set up the window and callback
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", select_corners)

# Display the first frame to select corners
while len(corners) < 4:
    cv2.imshow("Select ROI", frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()

# Convert corners to numpy array
corners = np.array(corners, dtype=np.float32)
print("Selected Corners:", corners)

# Load replacement image
replacement_img = cv2.imread('photo.jpg')


# Define the replacement image corners (top-left, top-right, bottom-right, bottom-left)
replacement_h, replacement_w = replacement_img.shape[:2]
replacement_corners = np.array([
    [0, 0],
    [replacement_w, 0],
    [replacement_w, replacement_h],
    [0, replacement_h]
], dtype=np.float32)

# Convert first frame to grayscale
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Prepare tracking parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

prev_corners = corners  # Initialize with manually selected corners

# Output video setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output4.mp4', fourcc, video.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert current frame to grayscale
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Track corners
    next_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_corners, None, **lk_params)

    # If tracking fails
    if np.sum(status) < 4:
        print("Tracking failed. Stopping.")
        break

    # Compute homography
    h, _ = cv2.findHomography(replacement_corners, next_corners)

    # Warp replacement image
    warped_img = cv2.warpPerspective(replacement_img, h, (frame.shape[1], frame.shape[0]))

    # Create a mask for blending
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(next_corners), (255, 255, 255))
    # Overlay the warped image directly
    frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))  # Remove original region
    frame = cv2.add(frame, cv2.bitwise_and(warped_img, mask))  # Add the replacement
    # Save the processed frame
    out.write(frame)

    # Update for next frame
    prev_gray = curr_gray
    prev_corners = next_corners

# Release resources
video.release()
out.release()
cv2.destroyAllWindows()
cv2.imwrite('debug_warped_image.jpg', warped_img)
