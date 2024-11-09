import os
from brainhealth.models.conf import VariableNames
import argparse
from brainhealth.alzheimer_detection_brain_mri import AlzheimerDetectionBrainMri

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
model = AlzheimerDetectionBrainMri()
model.train(base_model_path=base_model_path, 
            models_repo_dir_path=models_repo_dir_path, 
            train_data_dir=train_data_dir)