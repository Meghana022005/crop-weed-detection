import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define paths for local environment
ROOT_DIR = r"C:\machine learning project\Project5_Ag_Crop and weed detection\Project5_Ag_Crop and weed detection\Project5_Ag_Crop and weed detection\agri_data\data"
ANNOTATION_DIR = r"C:\machine learning project\Project5_Ag_Crop and weed detection\Project5_Ag_Crop and weed detection\Project5_Ag_Crop and weed detection\agri_data\data"

def load_images_and_annotations(image_dir, annotation_dir):
    """
    Load image files and their corresponding annotation files.
    Returns lists of image paths and annotation paths.
    """
    try:
        # Check if directories exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(annotation_dir):
            raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

        # Get image files (.jpeg or .jpg)
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpeg", ".jpg"))]
        if not image_files:
            raise ValueError(f"No .jpeg or .jpg files found in {image_dir}")

        # Get annotation files (.txt)
        annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith(".txt")]
        if not annotation_files:
            raise ValueError(f"No .txt annotation files found in {annotation_dir}")

        # Match images with annotations based on filenames (without extensions)
        image_paths = []
        annotation_paths = []
        for img in image_files:
            img_name = os.path.splitext(img)[0]
            ann_file = f"{img_name}.txt"
            if ann_file in annotation_files:
                image_paths.append(os.path.join(image_dir, img))
                annotation_paths.append(os.path.join(annotation_dir, ann_file))
            else:
                print(f"Warning: No annotation found for image {img}")

        print(f"Found {len(image_paths)} images with matching annotations")
        return image_paths, annotation_paths

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the dataset is downloaded and extracted to 'C:\\machine learning project\\dataset\\agri_data'.")
        return [], []
    except ValueError as e:
        print(f"Error: {e}")
        return [], []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return [], []

def parse_yolo_annotation(annotation_path, img_width, img_height):
    """
    Parse YOLO annotation file and convert to pixel coordinates.
    Returns list of [class_id, x_min, y_min, x_max, y_max].
    """
    bboxes = []
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Invalid annotation format in {annotation_path}")
                    continue
                class_id, x_center, y_center, width, height = map(float, parts)
                # Convert YOLO format (normalized) to pixel coordinates
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height
                bboxes.append([int(class_id), int(x_min), int(y_min), int(x_max), int(y_max)])
        return bboxes
    except Exception as e:
        print(f"Error parsing annotation {annotation_path}: {e}")
        return []

def visualize_image_with_bboxes(image_path, bboxes):
    """
    Display an image with bounding boxes.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

        # Draw bounding boxes
        for bbox in bboxes:
            class_id, x_min, y_min, x_max, y_max = bbox
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # Green for crop, red for weed
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            label = "Crop" if class_id == 0 else "Weed"
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display image
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(image_path))
        plt.show()

    except Exception as e:
        print(f"Error visualizing image {image_path}: {e}")

def main():
    # Load images and annotations
    image_paths, annotation_paths = load_images_and_annotations(ROOT_DIR, ANNOTATION_DIR)
    if not image_paths:
        print("No images or annotations to process. Exiting.")
        return

    # Process and visualize the first image as an example
    sample_image = image_paths[0]
    sample_annotation = annotation_paths[0]
    
    # Read image to get dimensions
    img = cv2.imread(sample_image)
    if img is None:
        print(f"Failed to load sample image: {sample_image}")
        return
    img_height, img_width = img.shape[:2]

    # Parse annotations
    bboxes = parse_yolo_annotation(sample_annotation, img_width, img_height)
    if not bboxes:
        print(f"No valid annotations for {sample_image}")
        return

    # Visualize
    print(f"Visualizing sample image: {sample_image}")
    visualize_image_with_bboxes(sample_image, bboxes)

if __name__ == "__main__":
    main()
