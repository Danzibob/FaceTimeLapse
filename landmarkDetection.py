import cv2, os, logging
import numpy as np
from config import *

# used for accessing url to download files
import urllib.request as urlreq

def loadImgGray(path):
    img = cv2.imread(path)
    # convert image to grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def existsOrDownload(path, URL):
    if os.path.exists(path):
        logging.info(f"File {path} already exists, skipping download")
    else:
        logging.info(f"File {path} does not exist - downloading...")
        urlreq.urlretrieve(URL, path)
        logging.info(f"File {path} retrieved!")

def loadHaar():
    """Create the Haar cascade detector, downloading the model if needed"""
    existsOrDownload(HAAR_CASCADE_PATH, HAAR_CASCADE_URL)
    return cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def loadLBF():
    """Create the LBF facial landmark detector, downloading the model if needed"""
    existsOrDownload(LBF_MODEL_PATH, LBF_MODEL_URL)
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBF_MODEL_PATH)
    return landmark_detector


def faceMoreThanFraction(face_coords, image_shape):
    """Accepts a detected face (x,y,w,h) and image dimensions
    returns True if image is larger than fraction specified in config"""
    (_,_,face_w,face_h) = face_coords
    (image_h, image_w) = image_shape
    return face_w > image_w/FACE_FRACT and face_h > image_h/FACE_FRACT

def detectFaces(image, detector=False):
    """Accepts a grayscale image and returns [(x,y,w,h), ...] of detected faces"""
    # Instanciate a detector if one wasn't passed in
    if not detector: detector = loadHaar()
    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(image)
    # Filter out faces that take up too little of either dimension of the image
    faces = np.array(list(filter(lambda f: faceMoreThanFraction(f, image.shape), faces)))

    # Print coordinates of detected faces
    logging.info(f"Detected {len(faces)} Faces")
    for f in faces: logging.debug("\tx: {} y: {} width: {} height: {}".format(*f))
    return faces

def detectLandmarks(image, faces=False, detector=False):
    """Accepts a grayscale image and list of detected faces, returns facial landmarks for each face"""
    # Detect faces if they weren't passed in
    if faces is False: 
        faces = detectFaces(image)
    # Instanciate a detector if one wasn't passed in
    if not detector: detector = loadLBF()
    _, landmarks = detector.fit(image, faces)
    return landmarks

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    img = loadImgGray("SelfTest/PXL_20211222_181147848.PORTRAIT.jpg")
    detectLandmarks(img)