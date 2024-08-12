
# pip install opencv-python==4.5.2

import cv2 

# Initialize video capture from default camera
video = cv2.VideoCapture(0)

# Load the Haar cascade classifier for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ask the user to input an ID
id = input("Enter Your ID: ")

# Initialize a counter for the number of collected images
count = 0

# Main loop for capturing and processing frames
while True:
    # Read a frame from the video feed
    ret, frame = video.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Increment the counter
        count += 1
        
        # Save the cropped face image to the dataset directory
        cv2.imwrite('datasets/User.' + str(id) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])
        
        # Draw a rectangle around the detected face on the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Display the frame with rectangles drawn around the faces
    cv2.imshow("Frame", frame)

    # Wait for a key press
    k = cv2.waitKey(1)

    # Break the loop if the count of collected images exceeds 500
    if count > 500:
        break

# Release the video capture object
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Print a message indicating the completion of dataset collection
print("Dataset Collection Done..................")
