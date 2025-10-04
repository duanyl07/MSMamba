import numpy as np

from sklearn.decomposition import PCA

def find_pca(img, n_components):
  X = np.array(img)
  X = X.reshape((-1, img.shape[2]))
  pca = PCA(n_components = n_components)
  reduced_X = pca.fit_transform(X)
  img = reduced_X.reshape((img.shape[0], img.shape[1], reduced_X.shape[1]))
  return img
