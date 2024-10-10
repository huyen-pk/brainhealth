from enum import Enum

class ModelOptimizers(Enum):
    Adam = 'Adam'
    SGD = 'SGD'
    RMSprop = 'RMSprop'
    Adagrad = 'Adagrad'
    AdaMax = 'AdaMax'

class ModelType(Enum):
    PyTorch = 'PyTorch'
    Keras = 'Keras'