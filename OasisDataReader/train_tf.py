import os
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.data_utils import ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import image_dataset_from_directory
from sklearn.model_selection import KFold
import numpy as np
import evaluation_metrics as custom_metrics

# Load the pre-trained model
root_dir = os.path.expanduser('~/Projects/AlzheimerDiagnosisAssist')
models_dir = os.path.join(root_dir, 'Models')
deep_brain_net_path = os.path.Join(models_dir, 'DeepBrainNet_model.h5')
deep_brain_net_model = load_model(deep_brain_net_path)
Alzheimer_model_name = 'DeepBrainNet_Alzheimer'
Alzheimer_model_dir = os.path.Join(models_dir, Alzheimer_model_name)

# Prepare the data
# Define the directories
train_data_dir = os.path.join(root_dir, 'Data', 'OASIS', 'slices')

# Load the dataset and labels from directory structure where subdirectories represent different classes
dataset = image_dataset_from_directory(
    train_data_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary',
    seed=123
)

# Convert dataset to numpy arrays for KFold
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for image, label in dataset:
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

images, labels = dataset_to_numpy(dataset)

# KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=123)
performance = []

for fold, (train_index, test_index) in enumerate(kf.split(images), 1):
    train_images, test_images = images[train_index], images[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    # Further split train set into train and validation sets
    val_split = int(len(train_images) * 0.2)
    val_images, val_labels = train_images[:val_split], train_labels[:val_split]
    train_images, train_labels = train_images[val_split:], train_labels[val_split:]

    # Convert numpy arrays back to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
    validation_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Only rescaling for validation and test
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
    validation_generator = validation_datagen.flow(val_images, val_labels, batch_size=32)
    test_generator = test_datagen.flow(test_images, test_labels, batch_size=32)

    # Compile the model
    deep_brain_net_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[custom_metrics.F1Score()])

    # Retrain the model
    deep_brain_net_model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )

    # Evaluate the model on the test set
    test_loss, test_acc = deep_brain_net_model.evaluate(test_generator, steps=len(test_generator))
    print(f'Test accuracy: {test_acc}')

    # Save the retrained model for each fold
    deep_brain_net_model.save(os.path.join(Alzheimer_model_dir, f'{Alzheimer_model_name}_fold_{fold}.h5'))

    # Store the performance of each fold
    performance.append((fold, test_loss, test_acc))
    with open(os.path.Join(Alzheimer_model_dir,'performance.txt', 'a')) as f:
        f.write(f'Fold {fold} - Test Loss: {test_loss}, Test Accuracy: {test_acc}\n')

# Determine the best performing fold
best_fold = max(performance, key=lambda x: x[2])
best_fold_index = best_fold[0]
best_model_path = os.path.join(Alzheimer_model_dir, f'{Alzheimer_model_name}_fold_{best_fold_index}.h5')
# Save the best model as the final model
final_model_path = os.path.join(Alzheimer_model_dir, f'{Alzheimer_model_name}_best.h5')
os.rename(best_model_path, final_model_path)
print(f'Best model saved as {final_model_path} with accuracy {best_fold[2]}')