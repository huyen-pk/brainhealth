import os
import copy
from abc import ABC, abstractmethod
import tensorflow as tf
from keras import models
from keras import Model as Model, optimizers as ops, losses as ls, metrics as mt
from brainhealth.models import enums, params
from brainhealth.metrics.evaluation_metrics import F1Score
from infrastructure.units_of_work import ModelTrainingDataDomain
from concurrent.futures import ThreadPoolExecutor

class ModelBuilderBase(ABC):

    def __init__(self, data_domain: ModelTrainingDataDomain) -> None:
        self.data_domain = data_domain
        self.model = None
        self.checkpoint = None
        self.optimizer = None
        self.model_params = None
        self.training_params = None

    @abstractmethod
    def fetch_data(self, 
                   page_index: int, 
                   training_params: params.TrainingParams,
                   **kwargs) -> tuple[tf.Tensor, tf.Tensor]:
        pass
    
    @abstractmethod
    def define_model(self, 
                     base_model: models.Model,
                     model_params: params.ModelParams) -> tf.keras.Model:
        """
        Define a model for training.

        Parameters:
        base_model (models.Model): The base model to use for training.
        model_params (params.ModelParams): The parameters to use for training.

        Returns:
        tf.keras.Model: The defined model.
        """
        pass

    def load_base_model(self, 
                        model_type: enums.ModelType, 
                        model_file_path: str) -> Model:
        """
        Load a pre-trained model from a file path.

        Parameters:
        model_type (enums.ModelType): The type of the model to load.
        model_path (str): The file path to the pre-trained model.

        Returns:
        tf.keras.Model: The pre-trained model.
        """
        if model_file_path is None:
            raise ValueError('Model file path is required.')
        
        # Attempt to download the model if the file does not exist
        if not os.path.exists(model_file_path):
            model_url = model_file_path
            model_file_path = self.data_domain.get_model(model_url)
            if not os.path.exists(model_file_path) or os.path.getsize(model_file_path) == 0:
                raise FileNotFoundError(f'Model not found at {model_file_path}')
        
        if model_type == enums.ModelType.Keras:
            return models.load_model(model_file_path, compile=False)
        elif model_type == enums.ModelType.PyTorch:
            raise NotImplementedError('PyTorch model conversion to TensorFlow is not supported yet.')
        else:
            raise ValueError(f'Unsupported model type: {model_type}')

    def save_checkpoint(self) -> None:
        chkpt_local_path = self.data_domain.get_checkpoint_local_path()
        def log_progress(future):
            print(f"Checkpoint saved to {self.data_domain.checkpoint_directory}/{key}")
        # Set up the thread pool executor
        executor = ThreadPoolExecutor(max_workers=1)
        # Submit the background task to the executor
        future = executor.submit(self.checkpoint.save, chkpt_local_path)
        future = executor.submit(self.data_domain.save_checkpoint, chkpt_local_path, key)
        # Add the callback to be called when the task is done
        future.add_done_callback(log_progress)        
    
    def load_checkpoint(self, 
                        model_name: str,
                        model: Model, 
                        optimizer: ops.Optimizer) -> tf.train.Checkpoint:
        """
        Load a model from a checkpoint directory.

        Parameters:
        model_name (str): name of the model to load checkpoints for.
        model: The model to load the checkpoint into.
        optimizer: The optimizer to load the checkpoint into.

        Returns:
        tf.train.Checkpoint: checkpoint.
        """
        checkpoint_dir = self.data_domain.get_latest_checkpoint(model_name=model_name)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print(f"Restored from {latest_checkpoint}")
        else:
            print("No checkpoint found, initializing from scratch.")
        return checkpoint
    
    def init_model(self, model: models.Model, 
                   model_params: params.ModelParams,
                   training_params: params.TrainingParams) -> tuple[models.Model, tf.train.Checkpoint]:
        """
        Initialize the model with the training parameters.

        Parameters:
        model (tf.keras.Model): The model to initialize.
        model_params (params.ModelParams): The model parameters to use for initialization.

        Returns:
        tf.keras.Model: The initialized model.
        tf.train.Checkpoint: The model checkpoint.
        """
        # Load checkpoints if available
        checkpoint = self.load_checkpoint(
            model_name=model_params.model_name,
            model=model, 
            optimizer=optimizer)
        # Load weights if available

        # Define the optimizer
        optimizer=None

        if training_params.optimizer == enums.ModelOptimizers.Adam:
            optimizer = ops.Adam(learning_rate=training_params.learning_rate)
        elif training_params.optimizer == enums.ModelOptimizers.SGD:
            optimizer = ops.SGD(learning_rate=training_params.learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {training_params.optimizer}')

        # Define the loss function
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=ls.CategoricalCrossentropy(),
            metrics=[mt.Precision(), mt.Recall(), F1Score()]
        )
        return model, checkpoint, optimizer
    
    def save_model(self, model: models.Model, model_dir: str) -> str:
        """
        Save the model to a directory.

        Parameters:
        model (tf.keras.Model): The model to save.
        model_dir (str): The directory to save the model to.
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(model_dir)

    def build(self, 
              model_file_path: str,
              model_type: enums.ModelType,
              model_params: params.ModelParams,
              training_params: params.TrainingParams
              ) -> tuple[models.Model, tf.train.Checkpoint]:
        # Load the base model
        base_model = self.load_base_model(
            model_type=model_type,
            model_file_path=model_file_path
        )
        # Define the model
        model = self.define_model(base_model=base_model, model_params=model_params)
        # Initialize the model
        model, checkpoint, optimizer = self.init_model(model=model, model_params=model_params, training_params=training_params)
        self.model = model
        self.checkpoint = checkpoint
        self.optimizer = optimizer
        self.model_params = model_params
        self.training_params = training_params
        return model, checkpoint
    
    def apply_gradients(self,
                            model: Model, 
                            optimizer: ops.Optimizer, 
                            data: tf.Tensor, 
                            labels: tf.Tensor) -> tuple[ops.Optimizer, tf.Tensor, Model]:
        """
        Perform a single training step.

        Parameters:
        model (Model): The model to train.
        optimizer (ops.Optimizer): The optimizer to use for training.
        data (tf.Tensor): The input data.
        labels (tf.Tensor): The target labels.

        Returns:
        None
        """
        with tf.GradientTape() as tape:
            predictions = model(data, training=True)
            loss = model.compute_loss(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return optimizer, loss

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

        repo = self.data_domain.get_model_repository_local(model_name=model_params.model_name)
        tuned_model = copy.deepcopy(model)

        optimizer = ops.get(model.optimizer)       
        save_every_n_batches = 50
        # Prepare the data
        steps_per_epoch = training_params.steps_per_epoch  # Set according to the streaming data availability
        continuation_token  = None
        for epoch in range(training_params.num_epoch):
            print(f"Epoch {epoch + 1}/{training_params.num_epoch}")
            if(epoch == 0):
                # Freeze all layers except the last one in pre-trained model
                for layer in tuned_model.layers[:-1]:
                    layer.trainable = False
            else:
                # Unfreeze all layers and train the model until convergence
                for layer in tuned_model.layers:
                    layer.trainable = True

            for step in range(steps_per_epoch):
                # Fetch a batch of data from the stream
                batchX, labels = self.fetch_data(
                    page_index=step, 
                    training_params=training_params, 
                    continuation_token=continuation_token
                )
                # Validate the model on the last step of an epoch
                if step == steps_per_epoch - 1:
                    results = tuned_model.evaluate(x=batchX, y=labels, steps=1, return_dict=True)
                    self.data_domain.save_performance_metrics(epoch, model_params.model_name, results, description='Validation')
                else:
                    # Perform a single training step
                    optimizer, loss = self.apply_gradients(model=tuned_model, optimizer=optimizer, data=batchX, labels=labels)
                if step % save_every_n_batches == 0:
                    # Send command to save checkpoint to storage
                    self.data_domain.save_checkpoint(model_name=model_params.model_name, checkpoint=checkpoint)
                    print(f"Checkpoint saved at batch {step} in epoch {epoch + 1}")

                # Log progress
                if step % 10 == 0:
                    print(f"Step {step + 1}/{steps_per_epoch}, Loss: {loss.numpy():.4f}")
                
                self.data_domain.purge_dataset(model_name=model_params.model_name, batch_index=step)

        # Save the model
        final_model_path = os.path.join(repo, f'{model_params.model_name}.h5')
        tuned_model.save(final_model_path, overwrite=True)
        return (tuned_model, final_model_path)


