import os
import tensorflow as tf
from keras import preprocessing as pp, optimizers as ops
from keras import metrics as mt, losses as ls
from keras import Model as Model
from brainhealth.models import enums, params
from brainhealth.metrics.evaluation_metrics import F1Score
from infrastructure.storage import Storage
from brainhealth.models import conf as config
import copy
import asyncio

class StreamingData_TF_Trainer:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
    
    async def __upload_checkpoint__(self, 
                                    chkpt_local_path:str, 
                                    bucket_name:str, 
                                    key:str) -> None:
        await self.storage.upload(chkpt_local_path,)
                    
    def __load_checkpoint__(self, 
                            checkpoint_dir: str,
                            model: Model, 
                            optimizer: ops.Optimizer) -> tf.train.Checkpoint:
        """
        Load a model from a checkpoint directory.

        Parameters:
        checkpoint_dir (str): The directory where the model checkpoint is saved.

        Returns:
        Model: The loaded model.
        """
        # TODO: download latest checkpoint from storage
        # Create a Checkpoint instance
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print(f"Restored from {latest_checkpoint}")
        else:
            print("No checkpoint found, initializing from scratch.")
        return checkpoint

    def train(self, 
              model: Model, 
              model_params: params.ModelParams, 
              training_params: params.TrainingParams,
              evaluation_metric: str) -> tuple[Model, str]:
        
        if model is None:
            raise ValueError('Model is required for training')
        
        if model_params is None:
            raise ValueError('Model parameters are required for training')
        
        if training_params is None:
            raise ValueError('Training parameters are required for training')

        repo = os.path.join(model_params.models_repo_path, model_params.model_dir)
        tuned_model = copy.deepcopy(model)

        optimizer=None

        if training_params.optimizer == enums.ModelOptimizers.Adam:
            optimizer = ops.Adam(learning_rate=training_params.learning_rate)
        elif training_params.optimizer == enums.ModelOptimizers.SGD:
            optimizer = ops.SGD(learning_rate=training_params.learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {training_params.optimizer}')

        # Compile the model
        tuned_model.compile(
            optimizer=optimizer,
            loss=ls.CategoricalCrossentropy(),
            metrics=[mt.Precision(), mt.Recall(), F1Score()]
        )
        checkpoint_prefix = os.path.join(repo, 'ckpt')
        checkpoint = self.__load_checkpoint__(
            checkpoint_dir=repo, 
            model=tuned_model, 
            optimizer=optimizer)
        save_every_n_batches = 50
        storage = self.storage
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
                images, labels = storage.download(
                    page_size=training_params.batch_size,
                    page_index=step,
                    page_count=1,
                    continuation_token=continuation_token
                )

                # Validate the model on the last step of an epoch
                if step == steps_per_epoch - 1:
                    results = tuned_model.evaluate(x=images, y=labels, steps=1, return_dict=True)
                    test_loss = results["loss"]
                    test_accuracy = {'metric': None, 'value' : None}
                    metrics = []
                    for key, value in results.items()[1:]:
                        metrics.append({'metric': key, 'value': value})
                        if(key == evaluation_metric):
                            test_accuracy = {'metric': key, 'value': value}
                    
                    if(test_accuracy['metric'] == None):
                        test_accuracy = metrics[0]
                    with open(os.path.join(repo,'performance.txt', 'a')) as f:
                        f.write(f'Epoch {epoch} - Test Loss: {test_loss}, Test {test_accuracy['metric']}: {test_accuracy['value']}\n')

                else:
                    # Perform a single training step
                    with tf.GradientTape() as tape:
                        predictions = tuned_model(images, training=True)
                        loss = tuned_model.compute_loss(labels, predictions)
                    gradients = tape.gradient(loss, tuned_model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, tuned_model.trainable_variables))

                if step % save_every_n_batches == 0:
                    chkpt_local_path = checkpoint.save(file_prefix=checkpoint_prefix) # Start async task to save the checkpoint to storage
                    # Run the async upload task
                    asyncio.run(self.__upload_checkpoint__(chkpt_local_path,))
                    print(f"Checkpoint saved at batch {step} in epoch {epoch + 1}")

                # Log progress
                if step % 10 == 0:
                    print(f"Step {step + 1}/{steps_per_epoch}, Loss: {loss.numpy():.4f}")

        # Save the model
        final_model_path = os.path.join(repo, f'{model_params.model_name}.h5')
        tuned_model.save(final_model_path)
        return (tuned_model, final_model_path)
