class BrainResult:
    def __init__(self, predictions: dict, description: str, statistics: dict):
        self.predictions = predictions
        self.description = description
        self.statistics = statistics