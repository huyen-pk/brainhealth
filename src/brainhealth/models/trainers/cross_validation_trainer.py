import os
import tensorflow as tf
from keras import preprocessing
from sklearn.model_selection import KFold
import numpy as np
from brainhealth.models import params
import copy

class CrossValidationTrainer:

    def load_data(self, data_dir: str) -> tf.data.Dataset:
        """
        Load the dataset and labels from directory structure where subdirectories represent different classes.
        Preprocess the images to a proper format.

        Parameters:
        data_dir (str): The directory where the data is stored.

        Returns:
        tf.data.Dataset: A TensorFlow dataset containing the images and labels.
        """

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'Directory not found: {data_dir}')
        
        dataset = preprocessing.image_dataset_from_directory(
            data_dir,
            image_size=(32, 32),
            color_mode='rgb',
            batch_size=32,
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

    def train(self, 
              data_dir: str, 
              model: tf.keras.Model, 
              model_params: params.ModelParams, 
              training_params: params.TrainingParams,
              evaluation_metric: str) -> tuple[tf.keras.Model, str]:
        
        if model is None:
            raise ValueError('Model is required for training')
        
        if model_params is None:
            raise ValueError('Model parameters are required for training')
        
        if training_params is None:
            raise ValueError('Training parameters are required for training')

        tuned_model = copy.deepcopy(model)

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
            for layer in tuned_model.layers[:-1]:
                layer.trainable = False

            # Retrain the model for one epoch
            tuned_model.fit(
                train_dataset,
                steps_per_epoch=len(train_dataset),
                epochs=1,
                shuffle=True,
                validation_data=validation_dataset,
                validation_steps=len(validation_dataset)
            )

            # Unfreeze all layers and train the model until convergence
            for layer in tuned_model.layers:
                layer.trainable = True

            tuned_model.fit(
                train_dataset,
                steps_per_epoch=len(train_dataset),
                epochs=training_params.num_epoch,
                shuffle=True,
                validation_data=validation_dataset,
                validation_steps=len(validation_dataset)
            )

            # Evaluate the model on the test set
            results = tuned_model.evaluate(test_dataset, steps=len(test_dataset), return_dict=True)
            test_loss = results["loss"]
            test_acc = {'metric': None, 'value' : None}
            metrics = []
            for key, value in results.items()[1:]:
                metrics.append({'metric': key, 'value': value})
                if(key == evaluation_metric):
                    test_acc = {'metric': key, 'value': value}
            
            if(test_acc['metric'] == None):
                test_acc = metrics[0]

            # Save the retrained model for each fold
            tuned_model.save(os.path.join(model_params.model_dir, f'{model_params.model_name}_fold_{fold}.h5'))

            # Store the performance of each fold
            performance.append((fold, test_loss, test_acc))
            with open(os.path.join(model_params.model_dir,'performance.txt', 'a')) as f:
                f.write(f'Fold {fold} - Test Loss: {test_loss}, Test {test_acc['metric']}: {test_acc['value']}\n')

        # Determine the best performing fold
        best_fold = max(performance, key=lambda x: x[2])
        best_fold_index = best_fold[0]
        best_model_path = os.path.join(model_params.model_dir, f'{model_params.model_name}_fold_{best_fold_index}.h5')

        # Save the best model as the final model
        final_model_path = os.path.join(model_params.model_dir, f'{model_params.model_name}_best.h5')
        os.rename(best_model_path, final_model_path)
        print(f'Best model saved as {final_model_path} with accuracy {best_fold[2]}')
        return (tuned_model, final_model_path)
