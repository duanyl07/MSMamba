import spectral as spy
from skimage.segmentation import felzenszwalb
from utils.find_pca import find_pca

def get_false_color(img):
  return spy.get_rgb(img, [30, 20, 10])

def segmentation(img, size=50):
  return felzenszwalb(find_pca(img, 0.999), sigma=0.95, min_size=size, channel_axis=2)