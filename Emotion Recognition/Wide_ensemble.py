import cv2
from controller import cvision
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Read an image
image = cv2.imread('D:/ConvolutionaNeuralNetwork/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks-master/media/jackie.jpg', cv2.IMREAD_COLOR)

# Recognize a facial expression if a face is detected. The boolean argument set to False indicates that the process runs on CPU
fer = cvision.recognize_facial_expression(image, False)

# Print list of emotions (individual classification from 9 convolutional branches and the ensemble classification)	
print(fer.list_emotion)











