import cv2
import numpy as np

# Define the lower and upper boundaries for skin color in HSV
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Define the region of interest (ROI) for hand detection
    roi = frame[50:400, 50:400]  # Larger ROI for better tracking

    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (50, 50), (400, 400), (0, 255, 0), 2)

    # Convert ROI to HSV for skin color segmentation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological transformations to remove noise
    mask = cv2.dilate(mask, None, iterations=3)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours of the hand
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)

        # Ignore small contours to avoid noise
        area = cv2.contourArea(max_contour)
        if area > 5000:  # Dynamically adjust threshold
            # Find the convex hull and defects
            hull = cv2.convexHull(max_contour, returnPoints=False)
            if len(hull) > 3:  # Ensure there are enough points for convexity defects
                defects = cv2.convexityDefects(max_contour, hull)

                # Count the number of fingers raised
                finger_count = 0
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        # Use the angle between fingers to count
                        a = np.linalg.norm(np.array(start) - np.array(far))
                        b = np.linalg.norm(np.array(end) - np.array(far))
                        c = np.linalg.norm(np.array(start) - np.array(end))
                        angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

                        # Check if the defect forms a valid finger
                        if angle <= np.pi / 2 and d > 5000:  # Adjust depth threshold
                            finger_count += 1

                    # Display the finger count
                    cv2.putText(frame, f"Fingers: {finger_count + 1}", (50, 450),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Display the processed frame
    cv2.imshow("Hand Gesture Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
