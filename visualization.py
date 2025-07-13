import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import random

def plot_class_collage(df, class_names, num_images=3):
    """
    Plot sample images from different classes
    """
    fig, axes = plt.subplots(num_images, len(class_names), 
                figsize=(15, num_images*3))
    
    for col_idx, class_name in enumerate(class_names):
        class_df = df[df['labels'] == class_name].sample(n=num_images)
        for row_idx, (_, row) in enumerate(class_df.iterrows()):
            img = mpimg.imread(row['filepaths'])
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis('off')
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(class_name)
    plt.tight_layout()
    plt.show()

def show_image_samples(gen):
    """
    Display sample images from generator with labels
    """
    images, labels = next(gen)
    class_names = list(gen.class_indices.keys())
    
    plt.figure(figsize=(15, 10))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i]/255)
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_image_samples_with_predictions(gen, model):
    """
    Show images with actual vs predicted labels
    """
    images, labels = next(gen)
    class_names = list(gen.class_indices.keys())
    preds = model.predict(images)
    
    plt.figure(figsize=(15, 10))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i]/255)
        true_label = class_names[np.argmax(labels[i])]
        pred_label = class_names[np.argmax(preds[i])]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()