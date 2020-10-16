from PIL import Image
import numpy as np

elevfile = 'data/worldelev.npy'
data = np.load(elevfile, 'r')
resolution = 120  # points per degree


def get_elevation(lat, lon):
    x = int(round((lon + 180) * resolution))
    y = int(round((90 - lat) * resolution)) - 1
    try:
        return max(0, data[y, x])
    except:
        return 0


def jpg_to_npy(jpg, npy):
    array = jpg_to_nparray(jpg)
    np.save(npy, array)


def jpg_to_nparray(jpg):
    im = Image.open(jpg)
    array = np.array(im)
    array = np.int16(array)
    return array


def get_coordinates(img_id):
    """
    for an img_id "14_{x coordinates}_{y coordinates}"

    returns ints of the x and y coordinates
    """
    segments = img_id.split("_")
    return int(segments[1]), int(segments[2])
