import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
from keras import layers
from keras import models
from keras import preprocessing
from keras import optimizers
from sklearn.model_selection import KFold
import numpy as np
from brainhealth.metrics.evaluation_metrics import F1Score
from PIL import Image
from scipy.stats import kurtosis
import cv2

class AlzheimerDetectionModel:
    def __init__(self):
        self.root_dir = os.path.expanduser('~/Projects/AlzheimerDiagnosisAssist')
        self.models_repo = os.path.join(self.root_dir, 'base_models')
        self.pretrained_model_path = os.path.join(self.models_repo, 'DBN_model.h5')
        self.pretrained_model = models.load_model(self.pretrained_model_path, compile=False)
        self.model_name = 'DeepBrainNet_Alzheimer'
        self.model_dir = os.path.join(self.models_repo, self.model_name)
        self.train_data_dir = os.path.join(self.root_dir, 'data', 'OASIS', 'slices')

    def load_data(self, data_dir: str) -> tf.data.Dataset:
        """
        Load the dataset and labels from directory structure where subdirectories represent different classes.

        Parameters:
        data_dir (str): The directory where the data is stored.

        Returns:
        tf.data.Dataset: A TensorFlow dataset containing the images and labels.
        """
        dataset = preprocessing.image_dataset_from_directory(
            data_dir,
            image_size=(32, 32),
            batch_size=32,
            color_mode='grayscale',  # Convert images to RGB
            seed=123
        )
        return dataset

    def dataset_to_numpy(self, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        images = []
        labels = []
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label.numpy())
        return np.array(images), np.array(labels)

    def define_model(self, pretrained_model: tf.keras.Model) -> tf.keras.Model:
        # Define the augmentation layers
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip horizontally and vertically
            layers.RandomRotation(0.2),                    # Randomly rotate by 20%
            layers.RandomZoom(0.2),                        # Random zoom by 20%
            layers.RandomContrast(0.2),                    # Random contrast adjustment
            layers.RandomBrightness(0.2)                   # Random brightness adjustment
        ])

        # Freeze all layers except the last one in pre-trained model
        for layer in pretrained_model.layers[:-1]:
            layer.trainable = False

        pretrained_model.summary()
        
        # Define the model
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(32, 32, 1)),
            # data_augmentation,  # Add the augmentation layer to the model
            # pretrained_model
        ])
        # for layer in data_augmentation.layers:
        #     model.add(layer)

        for layer in pretrained_model.layers:
            model.add(layer)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[F1Score()]
        )
        model.summary()
        return model

    def train(self, data_dir: str, model: tf.keras.Model):

        # Prepare the data
        dataset = self.load_data(data_dir)
        images, labels = self.dataset_to_numpy(dataset)

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

            # Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.CategoricalCrossEntropy(),
                metrics=[F1Score()]
            )

            # Retrain the model
            model.fit(
                train_dataset,
                steps_per_epoch=len(train_dataset),
                epochs=5,
                shuffle=True,
                validation_data=validation_dataset,
                validation_steps=len(validation_dataset)
            )

            # Evaluate the model on the test set
            test_loss, test_acc = model.evaluate(test_dataset, steps=len(test_dataset))
            print(f'Test accuracy: {test_acc}')

            # Save the retrained model for each fold
            model.save(os.path.join(self.model_dir, f'{self.model_name}_fold_{fold}.h5'))

            # Store the performance of each fold
            performance.append((fold, test_loss, test_acc))
            with open(os.path.Join(self.model_dir,'performance.txt', 'a')) as f:
                f.write(f'Fold {fold} - Test Loss: {test_loss}, Test Accuracy: {test_acc}\n')

        # Determine the best performing fold
        best_fold = max(performance, key=lambda x: x[2])
        best_fold_index = best_fold[0]
        best_model_path = os.path.join(self.model_dir, f'{self.model_name}_fold_{best_fold_index}.h5')
        # Save the best model as the final model
        final_model_path = os.path.join(self.model_dir, f'{self.model_name}_best.h5')
        os.rename(best_model_path, final_model_path)
        print(f'Best model saved as {final_model_path} with accuracy {best_fold[2]}')

    def predict(self, model: tf.keras.Model, input: np.ndarray) -> np.ndarray:
        prediction = model.predict(input)
        return prediction

    def decode_predictions(self, model: tf.keras.Model, predictions: np.ndarray) -> list:
        # Decode the predictions
        # Analyze skewness of the predictions
        # Return prediction for each image in the series and overall probability of the disease.
        return []
    
if __name__ == "__main__":
    model_instance = AlzheimerDetectionModel()
    pretrained_model = model_instance.pretrained_model
    model = model_instance.define_model(pretrained_model)

    # image = Image.open(os.path.expanduser('~/Projects/AlzheimerDiagnosisAssist/data/OASIS/slices/demented/OAS1_0003_MR1_mpr_n4_anon_111_t88_gfc.img_ASL_40.jpg'))
    image = Image.open(os.path.expanduser('~/Projects/AlzheimerDiagnosisAssist/data/OASIS/slices/nondemented/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.img_ASL_39.jpg'))

    input_data = np.array(image)
    input_data.resize(32, 32)
    reshaped_input = np.stack([input_data] * 3, axis=-1)  # Stack along the last axis to create 3 channels
    reshaped_input = np.expand_dims(input_data, axis=0)  # Add batch dimension
    print(f"test image reshaped: {input_data.shape}")

    test_data_dir = os.path.expanduser('~/Projects/AlzheimerDiagnosisAssist/tests/data')
    ds = model_instance.load_data(test_data_dir)
    images, labels = model_instance.dataset_to_numpy(ds)
    for i in images[:1]:
        i = i.flatten()
        print(f"image n° {i}: {i.shape}")
    print(f"image: {images[0].shape}")
    # kurtosis_vals = kurtosis(images, axis=0, fisher=True,bias=False)
    # print(f"kurtosis: {kurtosis_vals.shape}")
    # for i, k_value in enumerate(kurtosis_vals[:10]):
    #     print(f"Feature {i}: Kurtosis = {k_value}")

    for label in ds.class_names:
        print(f"label: {label}")
    # model.fit(ds, verbose=1)
    # predictions = model_instance.predict(model, ds)
    # print(predictions.shape)
