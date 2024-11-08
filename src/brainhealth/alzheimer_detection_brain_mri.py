from brainhealth.models.builders.brain_mri_builder import BrainMriModelBuilder
from brainhealth.models.trainers.trainer_base import Trainer
from brainhealth.models.params import ModelParams, TrainingParams
from brainhealth.models.enums import ModelOptimizers, ModelType
from brainhealth.metrics.evaluation_metrics import F1Score

class AlzheimerDetectionBrainMri:
    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer


    def train(self, 
              base_model_path: str,
              models_repo_dir_path: str,
              train_data_dir: str) -> None:

        self.training_params = TrainingParams(
            dataset_path=train_data_dir,
            batch_size=32,
            num_epoch=10,
            learning_rate=0.001,
            optimizer=ModelOptimizers.Adam,
            kfold=5
        )
        self.model_params = ModelParams(
            model_name='AlzheimerDetectionBrainMRI',
            base_model_path=base_model_path,
            base_model_type=ModelType.Keras,
            models_repo_path=models_repo_dir_path)
        
        # Define the model
        builder = BrainMriModelBuilder()
        base_model = builder.load_base_model(model_type=self.model_params.base_model_type, 
                                            model_file_path=self.model_params.base_model_path)
        self.model = builder.define_model(base_model=base_model, 
                                          model_params=self.model_params)
        # Train & evaluate the model
        self.tuned_model, self.tuned_model_path = self.trainer.train(self.model, self.model_params, self.training_params, F1Score.__name__)