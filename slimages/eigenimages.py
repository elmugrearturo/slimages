# -*- coding:utf-8 -*-

import os

import numpy as np

import cv2

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def load_images_from_folder(folder_path, resize_to_percentage=0.1,
                            static_resizing=False):
    images = []
    original_shape = ()
    small_shape = ()
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if original_shape == ():
                original_shape = img.shape
                if static_resizing:
                    small_shape = (100, 100)
                else:
                    small_shape = (
                        int(original_shape[0] * resize_to_percentage),
                        int(original_shape[1] * resize_to_percentage))
                print(f"Original size: {original_shape}")
                print(f"Processing size: {small_shape}")
            small_img = cv2.resize(img, small_shape[::-1])
            images.append(small_img.flatten())
    
    # Check that all images are of the same size
    img_size = images[0].shape[0]
    for img in images[1:] :
        assert img.shape[0] == img_size
    
    # Return a numpy array
    return np.array(images), original_shape, small_shape

def calculate_pca(matrix, shape, num_components=10, visualize=False):
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
    single_eigenimage = eigenimages.sum(axis=0)
    selected_components = i+1

    print(f"First {i+1} components explain {explained_variance} of variance.")

    # Visualize
    if visualize:
        for i, eigenimg in enumerate(eigenimages):
            plt.subplot(3, selected_components // 2, i + 1)
            plt.imshow(eigenimg.reshape(shape), cmap='gray')
            plt.title(f'Eigenimage {i+1}')
            plt.axis('off')
   
        plt.subplot(3, selected_components // 2, selected_components + 1)
        plt.imshow(single_eigenimage.reshape(shape), cmap='gray')
        plt.title(f'Eigenimage (all)')

        plt.tight_layout()
        plt.show()

    return single_eigenimage

def calculate_scores(single_eigenimage):
    # Calculate two scores
    # one just being a simple sum
    # the other just the positive values

    first_score = single_eigenimage.sum()
    second_score = ((single_eigenimage > 0) * single_eigenimage).sum()

    return (first_score, 
            second_score) 
