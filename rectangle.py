import cv2
import numpy as np

def callback(_):
    pass

def init_trackbars():
    cv2.namedWindow('HSV Trackbars')
    cv2.createTrackbar('LH', 'HSV Trackbars', 0, 255, callback)
    cv2.createTrackbar('LS', 'HSV Trackbars', 0, 255, callback)
    cv2.createTrackbar('LV', 'HSV Trackbars', 0, 255, callback)
    cv2.createTrackbar('UH', 'HSV Trackbars', 255, 255, callback)
    cv2.createTrackbar('US', 'HSV Trackbars', 255, 255, callback)
    cv2.createTrackbar('UV', 'HSV Trackbars', 255, 255, callback)

def get_lower_hsv():
    lower_hue = cv2.getTrackbarPos('LH', 'HSV Trackbars')
    lower_sat = cv2.getTrackbarPos('LS', 'HSV Trackbars')
    lower_val = cv2.getTrackbarPos('LV', 'HSV Trackbars')
    return (lower_hue, lower_sat, lower_val)

def get_upper_hsv():
    upper_hue = cv2.getTrackbarPos('UH', 'HSV Trackbars')
    upper_sat = cv2.getTrackbarPos('US', 'HSV Trackbars')
    upper_val = cv2.getTrackbarPos('UV', 'HSV Trackbars')
    return (upper_hue, upper_sat, upper_val)

def main(capture):
    while True:
        ret, frame = capture.read()

        lower_hsv = get_lower_hsv()
        upper_hsv = get_upper_hsv()
        upper_hsv = get_upper_hsv()

        upper_hue = cv2.getTrackbarPos('UH', 'HSV Trackbars')
        # Thresholding based on color (HSV)
        lower_val = cv2.getTrackbarPos
        

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Thresholding based on color (HSV)
        thresh = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Edge detection using Canny
        edges = cv2.Canny(thresh, 50, 150)

        # Morphological operations to improve edge detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest rectangle
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Draw the rectangle
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        cv2.imshow('Edges', edges)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    init_trackbars()
    camera = cv2.VideoCapture(0)  # Adjust the camera index as needed
    main(camera)
