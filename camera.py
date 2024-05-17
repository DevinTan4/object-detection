import cv2

def main():
    # Open the first webcam available
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        return

    # Set the frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Loop to continuously read frames from the webcam
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            print("Error: Unable to read frame")
            break

        # Display the frame
        cv2.imshow('Webcam Feed', frame)

        # Check for the 'q' key pressed to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
