import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update precision and recall
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        # Calculate F1 score
        precision_value = self.precision.result()
        recall_value = self.recall.result()
        return 2 * (precision_value * recall_value) / (precision_value + recall_value + tf.keras.backend.epsilon())

    def reset_states(self):
        # Reset the metrics
        self.precision.reset_states()
        self.recall.reset_states()

