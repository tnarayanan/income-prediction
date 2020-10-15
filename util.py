from PIL import Image
import numpy as np

elevfile = 'worldelev.npy'
data = np.load(elevfile, 'r')
resolution = 120 ## points per degree

def getElevation(lat, lon):
    x = int(round((lon + 180) * resolution))
    y = int(round((90 - lat) * resolution)) - 1
    try: return max(0, data[y, x])
    except: return 0

def jpg_to_npy(jpg, npy):
    im = Image.open(jpg)
    array = np.array(im)
    array = np.int16(array)
    np.save(npy, array)

