import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_labels(label_file):
    """
    Load class names from the specified labels file.
    
    Args:
        label_file (str): Path to the file containing class labels.
        
    Returns:
        list: A list of class names.
    """
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def load_and_preprocess_image(image_path):
    """
    Load an image from the specified path and preprocess it for model prediction.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: The preprocessed image ready for model prediction.
    """
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to read image.")
    
    # Resize to match the input size of the model (224, 224) and normalize
    img = cv2.resize(img, (224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def display_image(image_path):
    """
    Display the image using matplotlib.
    
    Args:
        image_path (str): Path to the image file.
    """
    # Load the image with OpenCV in RGB mode
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def predict_class(image, model, class_names):
    """
    Predict the class of the given image using the loaded model.
    
    Args:
        image (numpy.ndarray): The preprocessed image for prediction.
        model (tensorflow.keras.Model): The loaded model.
        class_names (list): A list of class names.
        
    Returns:
        tuple: The predicted class label and the confidence score.
    """
    # Predict the class
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)
    confidence_score = np.max(predictions)  # Get the confidence score (probability)
    prediction_label = class_names[class_idx]
    return prediction_label, confidence_score

def main(image_path):
    """
    Main function to handle image prediction and display.
    
    Args:
        image_path (str): Path to the image file.
    """
    # Load the trained model
    model = tf.keras.models.load_model('Model/keras_model.h5')

    # Load the class names from the labels.txt file
    class_names = load_labels('Model/labels.txt')

    # Preprocess the image
    image = load_and_preprocess_image(image_path)

    # Predict the class and confidence score
    prediction_label, confidence_score = predict_class(image, model, class_names)
    
    # Display the prediction result
    print(f'Class: {prediction_label}, Confidence: {confidence_score:.2f}')
    
    # Display the image
    display_image(image_path)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Rock Paper Scissors Image Classification")
    parser.add_argument('--image-path', type=str, required=True, help="Path to the image file testImage.jpg")
    args = parser.parse_args()

    # Call the main function with the provided image path
    main(args.image_path)


    