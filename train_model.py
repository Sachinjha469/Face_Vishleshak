import scipy.misc
import numpy as np
import os
import time
import GetFaceEncodings as GFE

   
# This function takes a list of known faces
def train_model():


    # Get path to all the known images
    # Filtering on .jpg extension - so this will only work with JPEG images ending with .jpg
    image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('images/'))

    # Sort in alphabetical order
    image_filenames = sorted(image_filenames)

    # Get full paths to images
    paths_to_images = ['images/' + x for x in image_filenames]

    # List of face encodings we have
    face_encodings = []

    #List of names we have
    names = []

    # Loop over images to get the encoding one by one
    for path_to_image in paths_to_images:
        # Get face encodings from the image
        face_encodings_in_image = GFE.get_face_encodings(path_to_image)

        # Make sure there's exactly one face in the image
        if len(face_encodings_in_image) != 1:
            print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
            continue;

        # Append the face encoding found in that image to the list of face encodings we have
        face_encodings.append(face_encodings_in_image[0])
        names.append(path_to_image[7:-4])
    

    np.save('face_encodings',face_encodings)
    np.save('name',names)

    print(len(names),len(face_encodings))

start =time.clock()
   
train_model()
print("Time to train_model",time.clock()-start)