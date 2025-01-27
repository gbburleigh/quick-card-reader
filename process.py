import cv2
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # Use KNN for classification
from sklearn.metrics import accuracy_score

import numpy as np

def extract_features(image):
    """Extracts simple features from the image (example: average pixel intensity).

    Args:
        image: The image to extract features from (grayscale).

    Returns:
        A 1D NumPy array of features.
    """
    if image is None:
        return None

    # Example: Calculate average pixel intensity
    average_intensity = np.mean(image)
    return np.array([average_intensity])  # Must return a NumPy array


def load_data(data_dir):
    """Loads image data and labels from a directory structure.

    Args:
        data_dir: The root directory containing subdirectories for each class.

    Returns:
        A tuple containing:
        - X: A list of feature vectors.
        - y: A list of labels (class names).
    """
    X = []
    y = []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):  # Make sure it's a directory
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                print('Loading...', image_path)
                image = read_image(image_path)
                if image is not None:
                    gray_image = process_image(image) # Process to grayscale
                    if gray_image is not None:
                        features = extract_features(gray_image)
                        if features is not None:
                            X.append(features)
                            y.append(class_name)
    return X, y


def train_model(X_train, y_train):
    """Trains a model (example: K-Nearest Neighbors).

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        A trained model.
    """
    model = KNeighborsClassifier(n_neighbors=3)  # Example: KNN with k=3
    model.fit(X_train, y_train)
    return model
    
def read_image(image_path):
    """Reads an image from the given path using OpenCV.

    Args:
        image_path: The path to the image file.

    Returns:
        The image as a NumPy array (OpenCV format), or None if the image 
        could not be read. Prints an error message to the console if the file 
        does not exist or can't be opened.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open or read image at {image_path}. Check file format.")
            return None
        return image
    except Exception as e: # Catching a broader exception type to handle various potential issues
        print(f"An error occurred while reading the image: {e}")
        return None

def process_image(image):
    """Processes the given image.

    Args:
        image: The image to process (NumPy array).

    Returns:
        The processed image (NumPy array), or None if there was an error.
    """
    if image is None:
        print("Error: No image to process.")
        return None

    try:
        # Example processing: Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        return None


def display_image(window_name, image):
    """Displays the given image in a window.

        Args:
            window_name: The name of the window.
            image: The image to display.
    """
    if image is None:
        print("No image to display.")
        return

    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()


def main():
    """Main function to demonstrate image reading and processing."""
    args = sys.argv
    image_path = args[1]

    X, y = load_data(image_path)
    X = np.array(X).reshape(-1, 1)  # Reshape for sklearn if using single feature.
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()