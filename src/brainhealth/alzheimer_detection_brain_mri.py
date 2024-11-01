import os
from brainhealth.models.builders.brain_mri_builder import BrainMriModelBuilder
from brainhealth.models.trainers.cross_validation_trainer import CrossValidationTrainer_TF as Trainer
from brainhealth.models.params import ModelParams, TrainingParams
from brainhealth.models.enums import ModelOptimizers, ModelType
from brainhealth.metrics.evaluation_metrics import F1Score
from brainhealth.models.conf import VariableNames
import numpy as np
import argparse

class AlzheimerDetectionBrainMri:
    def __init__(self) -> None:
        self.trainer = Trainer()


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

    # def predict(self, dir) -> tuple[np.ndarray, np.ndarray]:
    #     dataset = self.trainer.__load_data__(dir, shuffle=False)
    #     file_names = dataset.list_files(shuffle=False)
    #     predictions = self.tuned_model.predict(dataset)
    #     return (file_names, predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Alzheimer Detection on Brain MRI')
    parser.add_argument('--data', type=str, help='Directory containing MRI images for training')
    parser.add_argument('--base', type=str, help='Path to foundation model to build our model upon')
    parser.add_argument('--repo', type=str, help='Model repository')

    args = parser.parse_args()
    
    train_data_dir = args.data if args.data else os.getenv(VariableNames.TRAIN_DATA_DIR)
    base_model_path = args.base if args.base else os.getenv(VariableNames.BASE_MODEL_PATH)
    models_repo_dir_path = args.repo if args.repo else os.getenv(VariableNames.MODELS_REPO_DIR_PATH)

    model = AlzheimerDetectionBrainMri()
    model.train(base_model_path=base_model_path, 
                models_repo_dir_path=models_repo_dir_path, 
                train_data_dir=train_data_dir)