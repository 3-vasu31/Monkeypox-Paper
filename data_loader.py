import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(dataset_dir, train_ratio=0.75, valid_ratio=0.5, random_state=123):
    """
    Loads image file paths and labels, then splits data into train, validation, and test sets.
    """
    filepaths, labels = [], []
    classlist = os.listdir(dataset_dir)
    
    for class_name in classlist:
        class_path = os.path.join(dataset_dir, class_name)
        for file in os.listdir(class_path):
            filepaths.append(os.path.join(class_path, file))
            labels.append(class_name)
    
    df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    train_df, temp_df = train_test_split(df, train_size=train_ratio, stratify=df['labels'], random_state=random_state)
    valid_df, test_df = train_test_split(temp_df, train_size=valid_ratio, stratify=temp_df['labels'], random_state=random_state)
    
    return train_df, valid_df, test_df

def balance_data(df, target_samples, working_dir, img_size):
    """
    Balances dataset by augmenting underrepresented classes.
    """
    df = df.copy()
    aug_dir = os.path.join(working_dir, 'augmented')
    
    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir)
    os.makedirs(aug_dir)
    
    # Create subdirectories for each class
    for label in df['labels'].unique():
        os.makedirs(os.path.join(aug_dir, label))
    
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2
    )
    
    total_augmented = 0
    for label, group in df.groupby('labels'):
        sample_count = len(group)
        if sample_count < target_samples:
            delta = target_samples - sample_count
            target_dir = os.path.join(aug_dir, label)
            
            aug_generator = datagen.flow_from_dataframe(
                group, x_col='filepaths', y_col=None, target_size=img_size,
                class_mode=None, batch_size=1, shuffle=False,
                save_to_dir=target_dir, save_prefix='aug', save_format='jpg'
            )
            
            count = 0
            while count < delta:
                next(aug_generator)
                count += 1
            
            total_augmented += count
    
    print(f'Total augmented images created: {total_augmented}')
    
    # Merge original and augmented data
    aug_filepaths, aug_labels = [], []
    for label in os.listdir(aug_dir):
        for file in os.listdir(os.path.join(aug_dir, label)):
            aug_filepaths.append(os.path.join(aug_dir, label, file))
            aug_labels.append(label)
    
    aug_df = pd.DataFrame({'filepaths': aug_filepaths, 'labels': aug_labels})
    df = pd.concat([df, aug_df]).reset_index(drop=True)
    
    print(f'Final dataset size after augmentation: {len(df)}')
    return df

def create_generators(train_df, valid_df, test_df, img_size, batch_size):
    """
    Creates ImageDataGenerators for training, validation, and testing.
    """
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2
    )
    
    test_valid_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='filepaths', y_col='labels',
        target_size=img_size, class_mode='categorical',
        batch_size=batch_size, shuffle=True
    )
    
    valid_generator = test_valid_datagen.flow_from_dataframe(
        valid_df, x_col='filepaths', y_col='labels',
        target_size=img_size, class_mode='categorical',
        batch_size=batch_size, shuffle=False
    )
    
    # Adjust test batch size to be optimal
    test_batch_size = max([i for i in range(1, len(test_df) + 1) if len(test_df) % i == 0 and len(test_df) / i <= 80])
    test_steps = len(test_df) // test_batch_size
    
    test_generator = test_valid_datagen.flow_from_dataframe(
        test_df, x_col='filepaths', y_col='labels',
        target_size=img_size, class_mode='categorical',
        batch_size=test_batch_size, shuffle=False
    )
    
    return train_generator, valid_generator, test_generator, test_steps