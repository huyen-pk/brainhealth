import os
from brainhealth.models.conf import VariableNames
import argparse
from brainhealth.models.builders.brain_mri_builder import BrainMriModelBuilder
from brainhealth.models.params import ModelParams, TrainingParams
from brainhealth.models.enums import ModelOptimizers, ModelType
from infrastructure.denpendency_container import DependencyContainer

parser = argparse.ArgumentParser(description='Alzheimer Detection on Brain MRI')
parser.add_argument('--data', type=str, help='Directory containing MRI images for training')
parser.add_argument('--base', type=str, help='Path to foundation model to build our model upon')
parser.add_argument('--repo', type=str, help='Model repository')
args = parser.parse_args()

print("data: ", os.getenv(VariableNames.TRAIN_DATA_DIR))
print("base model:", os.getenv(VariableNames.BASE_MODEL_PATH))
print("model repo: ", os.getenv(VariableNames.MODELS_REPO_DIR_PATH))
print("storage: ", os.getenv(VariableNames.S3_BUCKET_NAME))

train_data_dir = args.data if args.data else os.getenv(VariableNames.TRAIN_DATA_DIR)
base_model_path = args.base if args.base else os.getenv(VariableNames.BASE_MODEL_PATH)
models_repo_dir_path = args.repo if args.repo else os.getenv(VariableNames.MODELS_REPO_DIR_PATH)
di_container = DependencyContainer.configure_injector()
builder = di_container.get(BrainMriModelBuilder)

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
        
       
model, checkpoint = builder.build(
            model_file_path=base_model_path,
            model_type=ModelType.Keras,
            model_params=model_params,
            training_params=training_params
        )
# # Train & evaluate the model
# tuned_model, tuned_model_path = builder.train(
#             model=model, model_params=model_params, training_params=training_params, checkpoint=checkpoint)
