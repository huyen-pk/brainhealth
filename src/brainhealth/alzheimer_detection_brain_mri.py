from brainhealth.models.builders.brain_mri_builder import BrainMriModelBuilder
from brainhealth.models.params import ModelParams, TrainingParams
from brainhealth.models.enums import ModelOptimizers, ModelType

class AlzheimerDetectionBrainMri:

    def __init__(self, builder: BrainMriModelBuilder) -> None:
        self.builder = builder

    def train(self, 
              base_model_path: str,
              models_repo_dir_path: str,
              train_data_dir: str) -> None:
        """
        Trains the Alzheimer detection model using brain MRI data.
        Args:
            base_model_path (str): Local path or download url to the pre-trained base model file.
            models_repo_dir_path (str): Directory path where the trained models will be saved.
            train_data_dir (str): Directory path containing the training data.
        Returns:
            None
        """
        training_params = TrainingParams(
            dataset_path=train_data_dir,
            batch_size=32,
            num_epoch=10,
            learning_rate=0.001,
            optimizer=ModelOptimizers.Adam,
            kfold=5
        )
        model_params = ModelParams(
            model_name='AlzheimerDetectionBrainMRI',
            base_model_path=base_model_path,
            base_model_type=ModelType.Keras,
            models_repo_path=models_repo_dir_path)
        
       
        model, checkpoint = self.builder.build(
            model_file_path=base_model_path,
            model_type=ModelType.Keras,
            model_params=model_params,
            training_params=training_params
        )
        # Train & evaluate the model
        tuned_model, tuned_model_path = self.builder.train(
            model=model, model_params=model_params, training_params=training_params, checkpoint=checkpoint)