# -*- coding:utf-8 -*-

import os

import numpy as np

import cv2

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def load_images_from_folder(folder_path):
    images = []
    original_shape = ()
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if original_shape == ():
                original_shape = img.shape
            images.append(img.flatten())
    
    # Check that all images are of the same size
    img_size = images[0].shape[0]
    for img in images[1:] :
        assert img.shape[0] == img_size
    
    # Return a numpy array
    return np.array(images), original_shape

def calculate_pca(matrix, original_shape, num_components=10):
    # PCA
    # TODO: select component number on the GUI
    pca = PCA(n_components=num_components)
    pca.fit(matrix)

    explained_variance = 0
    for i in range(num_components):
        if explained_variance < .8 :
            explained_variance += pca.explained_variance_ratio_[i]
        else:
            break
    
    eigenimages = pca.components_[:i]
    selected_components = i+1

    print(f"First {i+1} components explain {explained_variance} of variance.")

    # Visualize
    for i, eigenimg in enumerate(eigenimages):
        plt.subplot(3, selected_components // 2, i + 1)
        plt.imshow(eigenimg.reshape(original_shape), cmap='gray')
        plt.title(f'Eigenimage {i+1}')
        plt.axis('off')
   
    plt.subplot(3, selected_components // 2, selected_components + 1)
    single_eigenimage = eigenimages.sum(axis=0)
    plt.imshow(single_eigenimage.reshape(original_shape), cmap='gray')
    plt.title(f'Eigenimage (all)')

    plt.tight_layout()
    plt.show()
