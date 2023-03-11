from landmarkDetection import *
from parseImages import *
from config import *
import numpy as np

logging.basicConfig(level=logging.DEBUG)
PIL_logger = logging.getLogger('PIL')
PIL_logger.setLevel(logging.WARN)

# Get a list of (date, image_path) pairs
images = getImagesWithDates(INPUT_FOLDER)
print(images)

# Find canvas size
img = loadImgGray(images[0][1])
CANVAS_SIZE = img.shape[:2][::-1]

# Pre-emptively create the detectors for faces and landmarks
haar = loadHaar()
lbf = loadLBF()

face_landmarks = np.empty((len(images),68,2))

i = 0

for date, path in images:
    # Load image and detect landmarks
    img = loadImgGray(path)
    faces = detectFaces(img, detector=haar)

    try:
        face_landmarks[i] = np.array(detectLandmarks(img, faces, detector=lbf))
    except ValueError:
        logging.warn(f"Skipping {path} because no faces were detected!")
        continue

    # Select target points (result of adjusting last image)
    targetpts = face_landmarks[max(0, i-1)]
    sourcepts = face_landmarks[i]
    # Estimate best Affine transformation
    M = cv2.estimateAffinePartial2D(sourcepts, targetpts)[0]

    # Calculate new positions of landmarks for next iteration
    hom = np.append(sourcepts, np.ones((68,1)), axis=1)
    warped = (M@hom.T).T
    face_landmarks[i] = warped

    # Load the colour image and warp
    colour_image = cv2.imread(path)
    result = cv2.warpAffine(colour_image, M, CANVAS_SIZE)

    # Write the result to the output folder
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{i}.jpg"), result)
    
    logging.info(f"Completed frame {i+1}/{len(images)}")
    i += 1