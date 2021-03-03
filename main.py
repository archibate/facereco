from reco import process_image
from parse import get_landmarks


res, keys = process_image('001.jpg')
lms = get_landmarks(res)
