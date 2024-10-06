import os
import tensorflow as tf
from keras import layers
from keras import models
from keras import preprocessing
from keras import optimizers
from sklearn.model_selection import KFold
import numpy as np
from metrics.evaluation_metrics import F1Score
import brainhealth.models.misc.conf as config
import brainhealth.models.misc.model_params as params
import brainhealth.models.misc.enums as enums
from brainhealth.models.misc.brain_result import BrainResult
from utilities.model_converters import convert_pytorch_model_to_tf


class BrainMri:
    def __init__(self, model_params: params.ModelParams, training_params: params.TrainingParams):
        self.model_params = model_params
        self.training_params = training_params
        self.models_repo = os.getenv(model_params.models_repo_path)
        self.pretrained_model_path = os.getenv(model_params.base_model_path)
        self.pretrained_model = models.load_model(self.pretrained_model_path, compile=False)
        self.model_name = 'DeepBrainNet_Alzheimer'
        self.model_dir = os.path.join(self.models_repo, self.model_name)
        self.train_data_dir = os.getenv(config.TRAIN_DATA_DIR)

    def load_data(self, data_dir: str) -> tf.data.Dataset:
        """
        Load the dataset and labels from directory structure where subdirectories represent different classes.
        Preprocess the images to a proper format.

        Parameters:
        data_dir (str): The directory where the data is stored.

        Returns:
        tf.data.Dataset: A TensorFlow dataset containing the images and labels.
        """
        dataset = preprocessing.image_dataset_from_directory(
            data_dir,
            image_size=(32, 32),
            color_mode='rgb',
            batch_size=32,
            seed=123
        )
        return dataset
    
    def load_pretrained_model(self, model_type: enums.ModelType, model_path: str) -> tf.keras.Model:
        """
        Load a pre-trained model from a file path.

        Parameters:
        model_path (str): The file path to the pre-trained model.

        Returns:
        tf.keras.Model: The pre-trained model.
        """
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found at {model_path}')
        
        if model_type == enums.ModelType.Keras:
            return models.load_model(model_path, compile=False)
        elif model_type == enums.ModelType.PyTorch:
            return convert_pytorch_model_to_tf(model_path)
        else:
            raise ValueError(f'Unsupported model type: {model_type}')
    
    def dataset_to_numpy(self, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        images = []
        labels = []
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label.numpy())
        return np.array(images), np.array(labels)

    def define_model(self, pretrained_model: tf.keras.Model, 
                     model_params: params.ModelParams,
                     training_params: params.TrainingParams) -> tf.keras.Model:

        # Define the augmentation layers
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip horizontally and vertically
            layers.RandomRotation(0.2),                    # Randomly rotate by 20%
            layers.RandomZoom(0.2),                        # Random zoom by 20%
            layers.RandomContrast(0.2),                    # Random contrast adjustment
            layers.RandomBrightness(0.2)                   # Random brightness adjustment
        ])
        
        # Define the model
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(32, 32, 3))
        ])
        for layer in data_augmentation.layers:
            model.add(layer)

        for layer in pretrained_model.layers:
            model.add(layer)

        optimizer=None

        if self.training_params.optimizer == enums.ModelOptimizers.Adam:
            optimizer = optimizers.Adam(learning_rate=training_params.learning_rate)
        elif self.training_params.optimizer == enums.ModelOptimizers.SGD:
            optimizer = optimizers.SGD(learning_rate=training_params.learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {training_params.optimizer}')

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[F1Score()]
        )
        model.summary()
        return model

    def train(self, 
              data_dir: str, 
              model: tf.keras.Model, 
              model_params: params.ModelParams, 
              training_params: params.TrainingParams):

        # Prepare the data
        dataset = self.load_data(data_dir)
        images, labels = self.dataset_to_numpy(dataset)

        # KFold cross-validation
        kf = KFold(n_splits=training_params.kfold, shuffle=True, random_state=123)
        performance = []

        for fold, (train_index, test_index) in enumerate(kf.split(images), 1):
            train_images, test_images = images[train_index], images[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            # Further split train set into train and validation sets
            val_split = int(len(train_images) * 0.2)
            val_images, val_labels = train_images[:val_split], train_labels[:val_split]
            train_images, train_labels = train_images[val_split:], train_labels[val_split:]

            # Convert numpy arrays back to tf.data.Dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(training_params.batch_size)
            validation_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(training_params.batch_size)
            test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(training_params.batch_size)

            # Freeze all layers except the last one in pre-trained model
            for layer in model.layers[:-1]:
                layer.trainable = False

            # Retrain the model
            model.fit(
                train_dataset,
                steps_per_epoch=len(train_dataset),
                epochs=1,
                shuffle=True,
                validation_data=validation_dataset,
                validation_steps=len(validation_dataset)
            )

            # Unfreeze all layers and train the model until convergence
            for layer in model.layers:
                layer.trainable = True

            model.fit(
                train_dataset,
                steps_per_epoch=len(train_dataset),
                epochs=training_params.num_epoch,
                shuffle=True,
                validation_data=validation_dataset,
                validation_steps=len(validation_dataset)
            )

            # Evaluate the model on the test set
            test_loss, test_acc = model.evaluate(test_dataset, steps=len(test_dataset))
            print(f'Test accuracy: {test_acc}')

            # Save the retrained model for each fold
            model.save(os.path.join(model_params.model_dir, f'{model_params.model_name}_fold_{fold}.h5'))

            # Store the performance of each fold
            performance.append((fold, test_loss, test_acc))
            with open(os.path.Join(model_params.model_dir,'performance.txt', 'a')) as f:
                f.write(f'Fold {fold} - Test Loss: {test_loss}, Test Accuracy: {test_acc}\n')

        # Determine the best performing fold
        best_fold = max(performance, key=lambda x: x[2])
        best_fold_index = best_fold[0]
        best_model_path = os.path.join(model_params.model_dir, f'{model_params.model_name}_fold_{best_fold_index}.h5')

        # Save the best model as the final model
        final_model_path = os.path.join(model_params.model_dir, f'{model_params.model_name}_best.h5')
        os.rename(best_model_path, final_model_path)
        print(f'Best model saved as {final_model_path} with accuracy {best_fold[2]}')

    def predict(self, model: tf.keras.Model, input: np.ndarray) -> np.ndarray:
        predictions = model.predict(input)
        return self.decode_predictions(model, predictions)
    
    def decode_predictions(self, model: tf.keras.Model, predictions: np.ndarray) -> BrainResult:
        # Decode the predictions
        # Analyze skewness of the predictions
        # Return prediction for each image in the series and overall probability of the disease.
        result = BrainResult()
        return result
