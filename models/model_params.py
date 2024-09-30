import enums

class ModelParams:
    def __init__(self, 
                 model_name: str, 
                 base_model_path: str, 
                 base_model_type: enums.ModelType,
                 dataset_path: str):
        self.model_name = model_name
        self.base_model_path = base_model_path
        self.base_model_type = base_model_type
        self.dataset_path = dataset_path

class TrainingParams:
    def __init__(self, 
                 optimizer: enums.ModelOptimizers = enums.ModelOptimizers.Adam,
                 batch_size=32, 
                 learning_rate=0.001, 
                 num_epoch=10,
                 kfold=5):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.kfold = kfold