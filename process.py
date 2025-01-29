import cv2
import os
import sys
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # Use KNN for classification
from sklearn.metrics import accuracy_score
import pytesseract
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

def resize_image(image, target_size=(300, 200)):
    return cv2.resize(image, target_size) 

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
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                if not image_file.endswith('.png'):
                    continue
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
        The image as a NumPy array
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

def extract_text(card_roi):
    try:
        text = pytesseract.image_to_string(card_roi, config='--psm 6')
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

def process_image(image):
    """Processes the given image.

    Args:
        image: The image to process (NumPy array).

    Returns:
        None (Just displays the image for now)
    """
    if image is None:
        print("Error: No image to process.")
        return None

    try:
        # Example processing: Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = resize_image(gray_image)
        resized_image = cv2.GaussianBlur(image, (5, 5), 0)
        # thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        edges = cv2.Canny(resized_image, 200, 250)
        text = extract_text(edges)
        cv2.imshow("Grayscale Image", gray_image)
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        return None

def main():
    """Main function to demonstrate image reading and processing."""
    args = sys.argv
    image_path = args[1]

    X, y = load_data(image_path)
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()