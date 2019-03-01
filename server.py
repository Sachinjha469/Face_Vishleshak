import numpy as np
import os
import time
from flask import Flask,jsonify,request
import base64
from PIL import Image
from io import BytesIO
import GetFaceEncodings as GFE

app = Flask(__name__)

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


@app.route("/predict",methods=['POST'])
def predict():

    start =time.clock()
    
    content = request.get_json()
    im = Image.open(BytesIO(base64.b64decode(content["image"])))
    im.save('test/image.jpg')
    

    print("Image Recieved")

    print("Time to preprocess image",time.clock()-start)
    start =time.clock()



    face_encodings = np.load('face_encodings.npy')
    names = np.load('name.npy')


    print("Time to load train_model",time.clock()-start)
    start =time.clock()

    

    face_encodings_in_image = GFE.get_face_encodings('test/image.jpg')
    s=""
    found=0

    if len(face_encodings_in_image)==0:
        return "No face"


    for i in range(len(face_encodings_in_image)):
        # Find match for the face encoding found in this test image
        match,flag= find_match(face_encodings, names, face_encodings_in_image[i])

        if(flag==1):
            found=1
            s+=match+" "

    print(s)

    print("time to match",time.clock()-start)

    if(found==0):
        return "No match"
    else:
        return s


if __name__ == '__main__':
    app.run(host='0.0.0.0')
 

