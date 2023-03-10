import cv2, os, logging
import numpy as np
from config import *
from dateutil import parser as dateParser

from PIL import Image, ExifTags

def parseImageDate(full_path):
    img = Image.open(full_path)
    if "exif" not in img.info.keys():
        logging.warning(f"Image {full_path} has no EXIF data! Skipping...")
        return False
    date_time_original = img.getexif()._get_merged_dict()[36867]
    return dateParser.parse(date_time_original)

def getImagesWithDates(folder):
    """Retrieves the paths for any images in the specified folder, returns [(date, image_path), ...]"""
    paths = []
    for image_path in os.listdir(folder):
        if image_path.split(".")[-1] in IMAGE_FORMATS:
            full_path = os.path.join(INPUT_FOLDER, image_path)
            if img_date := parseImageDate(full_path):
                paths.append((img_date, full_path))
    logging.info(f"Detected {len(paths)} images with valid metadata")
    paths.sort()
    return paths

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    PIL_logger = logging.getLogger('PIL')
    PIL_logger.setLevel(logging.WARN)
    print(getImagesWithDates())