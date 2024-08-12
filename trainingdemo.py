import cv2
import numpy as np
from PIL import Image
import os

# Create LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Dataset path
path = "datasets"

def getImageID(path):
    # Get list of image file paths in the dataset directory
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    
    # Initialize lists to store face images and corresponding IDs
    faces = []
    ids = []
    
    # Loop through each image path
    for imagePath in imagePath:
        # Open image, convert to grayscale
        faceImage = Image.open(imagePath).convert('L')
        # Convert PIL image to numpy array
        faceNP = np.array(faceImage)
        # Extract ID from image file name
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        # Append face image and ID to lists
        faces.append(faceNP)
        ids.append(Id)
        
        # Display image for training (debugging purpose)
        cv2.imshow("Training", faceNP)
        cv2.waitKey(1)
    
    # Close OpenCV window
    cv2.destroyAllWindows()
    
    return ids, faces

# Get IDs and face data from images in the dataset directory
IDs, facedata = getImageID(path)

# Train the recognizer using the face data and corresponding IDs
recognizer.train(facedata, np.array(IDs))

# Write trained recognizer to file
recognizer.write("Trainer.yml")

# Print completion message
print("Training Completed............")
