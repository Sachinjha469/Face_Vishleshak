import ImageEnhance as IE
import dlib
import scipy.misc
import numpy as np

import cv2

haar_detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Get Face Detector from dlib
# This allows us to detect faces in images
face_detector = dlib.get_frontal_face_detector()

# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# Get Pose Predictor from dlib
# This allows us to detect landmark points in faces and understand the pose/angle of the face
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Get the face recognition model
# This is what gives us the face encodings (numbers that identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


# This function will take an image and return its face encodings using the neural network
def get_face_encodings(path_to_image):

    #Equalise image using Histogram equalization in cv2
    path_to_image=IE.avg_brightness(path_to_image)

    # Load image using scipy
    image = scipy.misc.imread(path_to_image)

    # Detect faces using the face detector
    detected_faces_using_SVM =face_detector(image, 1)
    #detected_faces_using_CNN = cnn_face_detector(image, 1)

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #detected_faces_using_Haar = haar_detector.detectMultiScale(gray, 1.8, 2)
    
        


    # Get pose/landmarks of those faces
    # Will be used as an input to the function that computes face encodings
    # This allows the neural network to be able to produce similar numbers for faces of the same people, regardless of camera angle and/or face positioning in the image
    shapes_faces = [shape_predictor(image, face) for face in detected_faces_using_SVM]
    #shapes_faces = [shape_predictor(image, dlib.rectangle(face.rect.left(),face.rect.top(),face.rect.right(),face.rect.bottom())) for face in detected_faces_using_CNN]
    #shapes_faces = [shape_predictor(image,dlib.rectangle(x,y,x+w,y+h)) for (x,y,w,h) in detected_faces_using_Haar]

    # For every face detected, compute the face encodings
    face_encoding= [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]
    
    return face_encoding


get_face_encodings('test_images/virenderrangasolti.jpg')