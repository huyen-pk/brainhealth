import cv2
import numpy as np
def format_image(image_path):
    image_content = cv2.imread(image_path)
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)
    image_content = cv2.resize(image_content, (32, 32))
    image_content = image_content / 255.0  # Normalize to [0, 1]
    return image_content

# def brain_segment(image):
#     # Load the model
#     model = models.load_model("src/brainhealth_web/app/models/brain_segmentation_model.h5")
#     # Make predictions
#     predictions = model.predict(image)
#     predicted_class = tf.argmax(predictions, axis=1).numpy()
#     label_map = {0: 'Brain', 1: 'Not Brain'}
#     predicted_label = label_map.get(int(predicted_class[0]), "Unknown")
#     return predicted_label

# def contain_brain(image):
#     # Load the model
#     model = models.load_model("src/brainhealth_web/app/models/brain_model.h5")
#     # Make predictions
#     predictions = model.predict(image)
#     predicted_class = tf.argmax(predictions, axis=1).numpy()
#     label_map = {0: 'Brain', 1: 'Not Brain'}
#     predicted_label = label_map.get(int(predicted_class[0]), "Unknown")
#     return predicted_label