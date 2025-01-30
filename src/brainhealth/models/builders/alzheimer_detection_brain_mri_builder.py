import tensorflow as tf
from keras import layers, models, optimizers, Optimizer, metrics, losses, Model
from keras import initializers
from brainhealth.models import enums, params
from brainhealth.models.builders.builder_base import ModelBuilderBase
from brainhealth.metrics.evaluation_metrics import F1Score
from infrastructure.units_of_work import ModelTrainingDataDomain
import numpy as np
from typing import override
import uuid
import copy

class AlzheimerDetectionBrainMriModelBuilder(ModelBuilderBase):

    def __init__(self, data_domain: ModelTrainingDataDomain) -> None:
        self.data_domain = data_domain

    @override
    def load_base_model(self, model_name: str) -> Model:
        """
        Load a pre-trained model from a file path.

        Parameters:
        model_name (str): The name of the model to load.

        Returns:
        tf.keras.Model: The pre-trained model.
        """
        return self.data_domain.get_model(model_name=model_name, file_type="h5")
    
    @override
    def define_model(self, 
                     base_model: models.Model,
                     model_params: params.ModelParams) -> Model:
        """
        Define the model architecture based on the foundation model and the training parameters.

        Parameters:
        base_model (tf.keras.Model): The foundation model to build upon.
        training_params (params.TrainingParams): The training parameters to use for the model.

        Returns:
        tf.keras.Model: The compiled model.
        """
        
        return base_model
    
    @override
    def init_model(self, model: models.Model, 
                   model_params: params.ModelParams,
                   training_params: params.TrainingParams) -> tuple[models.Model, tf.train.Checkpoint, Optimizer]:
        """
        Initialize the model with the training parameters.

        Parameters:
        model (tf.keras.Model): The model to initialize.
        model_params (params.ModelParams): The model parameters to use for initialization.

        Returns:
        tf.keras.Model: The initialized model.
        tf.train.Checkpoint: The model checkpoint.
        """
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=training_params.learning_rate,
            decay_steps=training_params.steps_per_epoch,
            decay_rate=0.85,
            staircase=False  # If True, learning rate decays in discrete steps
        )

        # Define the optimizer
        optimizer=None

        if training_params.optimizer == enums.ModelOptimizers.Adam:
            optimizer = optimizers.Adam(learning_rate=lr_schedule)
        elif training_params.optimizer == enums.ModelOptimizers.SGD:
            optimizer = optimizers.SGD(learning_rate=lr_schedule)
        else:
            raise ValueError(f'Unsupported optimizer: {training_params.optimizer}')

        # Load checkpoints if available
        checkpoint = self.load_checkpoint(
            model_name=model_params.model_name,
            model=model, 
            optimizer=optimizer)
        
        # Initialize the model's variables and inputs to avoid unknown variable error
        input_shape = model.input_shape # Get the shape of the input layer, which will be (None, shape)
        dummy_input = tf.zeros((1, *input_shape[1:]))
        __ = model(dummy_input)

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=losses.BinaryCrossentropy(),
            metrics=[metrics.Precision(), metrics.Recall(), F1Score()]
        )
        return model, checkpoint, optimizer

    @override
    def fetch_data(self, 
                   page_index: int, 
                   training_params: params.TrainingParams,
                   model_params: params.ModelParams,
                   **kwargs) -> tuple[np.ndarray, np.ndarray]:
        images, labels = self.data_domain.get_dataset(
                    model_name=model_params.model_name,
                    page_size=training_params.batch_size,
                    page_index=page_index,
                    page_count=1,
                    continuation_token=kwargs.get('continuation_token', None))
        return images, labels
    
    @override
    def train(self, 
              model: Model, 
              model_params: params.ModelParams, 
              training_params: params.TrainingParams,
              checkpoint: tf.train.Checkpoint) -> tuple[Model, str]:
        
        if model is None:
            raise ValueError('Model is required for training')
        
        if model_params is None:
            raise ValueError('Model parameters are required for training')
        
        if training_params is None:
            raise ValueError('Training parameters are required for training')

        tuned_model = copy.deepcopy(model)
        for layer in model.layers:
            layer.trainable = True
        optimizer = optimizers.get(tuned_model.optimizer)  
        save_every_n_batches = training_params.save_every_n_batches
        # Prepare the data
        steps_per_epoch = training_params.steps_per_epoch # Set according to the streaming data availability
        continuation_token  = None
        cycle_identifier = str(uuid.uuid4())
        for epoch in range(1, training_params.num_epoch + 1):
            print(f"Epoch {epoch}/{training_params.num_epoch}")

            for step in range(1, steps_per_epoch  + 1):
                # Fetch a batch of data from the stream
                batches, labels = self.fetch_data(
                    page_index=step, 
                    training_params=training_params, 
                    model_params=model_params,
                    continuation_token=continuation_token
                )
                batchX = batches[0]
                batchX_labels = labels[0]
                learning_rate = optimizer._get_current_learning_rate() 
                descriptions={"step": step, "learning_rate": learning_rate}
                # Validate the model on the last step of an epoch
                if step % steps_per_epoch == 0:
                    results = tuned_model.evaluate(x=batchX, y=batchX_labels, steps=1, return_dict=True, verbose=0)
                    for metric, value in results.items():
                        print(f"{metric}: {value}")
                    self.data_domain.save_performance_metrics(
                        epoch=epoch, model_name=model_params.model_name, metrics=results, descriptions=descriptions , identifier=f"{cycle_identifier}_validation")
                else:
                    # Perform a single training step
                    optimizer, loss, metrics = self.apply_gradients(model=tuned_model, optimizer=optimizer, input=batchX, labels=batchX_labels)

                self.data_domain.save_performance_metrics(
                    epoch=epoch, model_name=model_params.model_name, metrics=metrics, descriptions=descriptions, identifier=f"{cycle_identifier}_training")
                
                if step % save_every_n_batches == 0:
                    # Send command to save checkpoint to storage
                    self.data_domain.save_checkpoint(model_name=model_params.model_name, checkpoint=checkpoint)
                    print(f"Checkpoint saved at batch {step} in epoch {epoch} in {self.data_domain.get_checkpoint_local_path(model_params.model_name)}")
                
                self.data_domain.purge_dataset(model_name=model_params.model_name, batch_index=step)
            self.data_domain.save_model(model=tuned_model, model_name=f"{model_params.model_name}_{cycle_identifier}", type='keras')

        # Save the model
        final_model_path = self.data_domain.save_model(model=tuned_model, model_name=model_params.model_name, type='keras')
        return (tuned_model, final_model_path)