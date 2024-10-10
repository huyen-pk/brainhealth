import brainhealth.models.misc.enums as enums

class ModelParams:
    """
    ModelParams is a class that encapsulates the parameters required for configuring a machine learning model.
    Attributes:
        model_name (str): The name of the model.
        base_model_path (str): The file path to the base model.
        base_model_type (enums.ModelType): The type of the base model.
        dataset_path (str): The file path to the dataset.
        models_repo_path (str): The file path to the repository containing models.
        model_dir (str): The directory where the model will be saved.
    Methods:
        __init__(model_name: str, base_model_path: str, base_model_type: enums.ModelType, dataset_path: str, models_repo_path: str, model_dir: str):
            Initializes the ModelParams instance with the provided parameters.
    """
    def __init__(self, 
                 model_name: str, 
                 base_model_path: str, 
                 base_model_type: enums.ModelType,
                 models_repo_path: str,
                 model_dir: str):
        
        self.model_name = model_name
        self.base_model_path = base_model_path
        self.base_model_type = base_model_type
        self.models_repo_path = models_repo_path
        self.model_dir = model_dir

class TrainingParams:
    def __init__(self,
                 dataset_path: str,
                 optimizer: enums.ModelOptimizers = enums.ModelOptimizers.Adam,
                 batch_size=32, 
                 learning_rate=0.001, 
                 num_epoch=10,
                 kfold=5):
        self.dataset_path = dataset_path
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.kfold = kfold
