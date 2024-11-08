from abc import ABC, abstractmethod
from brainhealth.models import params
from keras import Model

class Trainer(ABC):
    @abstractmethod
    def train(self, 
              model: Model, 
              model_params: params.ModelParams, 
              training_params: params.TrainingParams,
              evaluation_metric: str) -> tuple[Model, str]:
        pass