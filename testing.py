import dlib
import scipy.misc
import numpy as np
import time
import os
import GetFaceEncodings as GFE


# This is the tolerance for face comparisons
# The lower the number - the stricter the comparison
# To avoid false matches, use lower value
# To avoid false negatives (i.e. faces of the same person doesn't match), use higher value
# 0.5-0.6 works well
TOLERANCE = 0.5

# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
    # Finds the difference between each known face and the given face (that we are comparing)
    # Calculate norm for the differences with each known face
    # A match occurs when the (norm) difference between a known face and the given face is less than or equal to the TOLERANCE value
    return(np.linalg.norm(known_faces - face,axis=1))

# This function returns the name of the person whose image matches with the given face (or 'Not Found')
# known_faces is a list of face encodings
# names is a list of the names of people (in the same order as the face encodings - to match the name with an encoding)
# face is the face we are looking for
def find_match(known_faces, names, face):
    # Call compare_face_encodings to get a list of True/False values indicating whether or not there's a match
    matches = compare_face_encodings(known_faces, face)

    i = np.argmin(matches)

    # for k in range(len(matches)):
    #     print(names[k]+"  "+str(matches[k]))

    if matches[i] < TOLERANCE :
        return names[i],1
    else:
        return 'Not Found',0

start = time.clock()
face_encodings = np.load('face_encodings.npy')
names = np.load('name.npy')


# Get path to all the test images
# Filtering on .jpg extension - so this will only work with JPEG images ending with .jpg
test_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('test_images/'))

# Get full paths to test images
paths_to_test_images = ['test_images/' + x for x in test_filenames]

# Iterate over test images to find match one by one
eff=0
total=185
for path_to_image in paths_to_test_images:
    # Get face encodings from the test image
    face_encodings_in_image = GFE.get_face_encodings(path_to_image)


    #Looping over each and every face of the image and matching it with the image in the database

    for i in range(len(face_encodings_in_image)):
        # Find match for the face encoding found in this test image
        match,flag = find_match(face_encodings, names, face_encodings_in_image[i])

        if match in path_to_image:
            eff=eff+1
        

        # Print the path of test image and the corresponding match
        print(path_to_image[5:], match)
    

print(eff/total)
print("Time to preprocess image",time.clock()-start)