import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ultralytics import YOLO
import shutil
import yaml


class ProbeDetectorPipeline:
    """
    A class to handle the end-to-end pipeline for detecting probes in images using a YOLO model.
    This includes dataset preparation, model training, and evaluation.
    """

    def __init__(self, data_dir, annotations_file, output_dir, img_size=640):
        """
        Initializes the class with paths and configurations.

        Args:
            data_dir (str): Path to the directory containing the images.
            annotations_file (str): Path to the JSON file with annotations.
            output_dir (str): Path to the directory where processed data will be stored.
            img_size (int): Image size for training and inference (e.g., 640x640).
        """
        self.data_dir = Path(data_dir)  # Directory containing raw images
        self.annotations_file = Path(annotations_file)  # JSON file with bounding box annotations
        self.output_dir = Path(output_dir)  # Directory to store processed data and results
        self.img_size = img_size  # Image size used for YOLO training and inference

        # Define paths for train, validation, and test subsets
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"

    def parse_annotations(self):
        """
        Parses the annotations JSON file to extract bounding boxes and match them to images.

        Returns:
            list: A list of dictionaries containing image paths and bounding boxes.
        """
        # Load the annotations JSON file
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)

        # Create a dictionary mapping image IDs to file names
        image_dict = {img['id']: img['file_name'] for img in data['images']}
        annotations = []

        # Match annotations to their corresponding images
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            bbox = annotation['bbox']  # Bounding box [x, y, width, height]

            # If the image ID exists, append the annotation to the list
            if image_id in image_dict:
                annotations.append({
                    "image_path": self.data_dir / image_dict[image_id],  # Full path to the image
                    "bbox": bbox  # Bounding box
                })
        return annotations

    def split_dataset(self, annotations, test_split=0.2, val_split=0.1):
        """
        Splits the dataset into train, validation, and test sets and creates directories for each subset.

        Args:
            annotations (list): A list of annotations containing image paths and bounding boxes.
            test_split (float): Proportion of the dataset to use as the test set.
            val_split (float): Proportion of the remaining data to use as the validation set.
        """
        # Split annotations into training+validation and test subsets
        train_val, test = train_test_split(annotations, test_size=test_split, random_state=42)
        # Further split training+validation into training and validation subsets
        train, val = train_test_split(train_val, test_size=val_split / (1 - test_split), random_state=42)

        # Prepare directories and copy images for each subset
        for subset, data in zip(["train", "val", "test"], [train, val, test]):
            subset_dir = getattr(self, f"{subset}_dir") / "images"  # Image directory for the subset
            labels_dir = getattr(self, f"{subset}_dir") / "labels"  # Label directory for the subset
            subset_dir.mkdir(parents=True, exist_ok=True)  # Create directories
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Copy images and create YOLO-compatible labels
            for entry in tqdm(data, desc=f"Preparing {subset} dataset"):
                image_path = entry['image_path']  # Path to the image
                shutil.copy(image_path, subset_dir / Path(image_path).name)  # Copy image to subset directory

                # Write YOLO label files
                label_file = labels_dir / (Path(image_path).stem + ".txt")
                with open(label_file, 'w') as f:
                    x, y, width, height = entry['bbox']
                    # Convert bounding box to YOLO format
                    x_center = (x + width / 2) / self.img_size
                    y_center = (y + height / 2) / self.img_size
                    norm_width = width / self.img_size
                    norm_height = height / self.img_size
                    f.write(f"0 {x_center} {y_center} {norm_width} {norm_height}\n")

    def create_config(self):
        """
        Creates a YOLO configuration file specifying paths and class names.
        """
        config = {
            "train": str((self.train_dir / "images").resolve()),  # Path to training images
            "val": str((self.val_dir / "images").resolve()),  # Path to validation images
            "nc": 1,  # Number of classes (1 for "probe")
            "names": ["probe"]  # Class name
        }
        # Write configuration to a YAML file
        with open(self.output_dir / "data.yaml", 'w') as f:
            yaml.dump(config, f)

    def train_model(self, model_arch="yolov8n", epochs=50, batch_size=16):
        """
        Trains the YOLO model on the prepared dataset.

        Args:
            model_arch (str): YOLO model architecture to use (e.g., yolov8n, yolov8s).
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
        """
        # Load a YOLO model with pretrained weights
        model = YOLO(f"{model_arch}.pt")
        # Train the model using the configuration file
        model.train(
            data=str((self.output_dir / "data.yaml").resolve()),  # Path to the configuration file
            epochs=epochs,
            imgsz=self.img_size,  # Image size
            batch=batch_size  # Batch size
        )
        print(f"Training complete. Weights saved in {self.output_dir / 'weights'}")

    def evaluate_model(self, model_path, subset="test"):
        """
        Evaluates the YOLO model on the specified subset (test or validation).

        Args:
            model_path (str): Path to the trained YOLO model weights.
            subset (str): Subset to evaluate on ("test" or "val").
        """
        # Load the trained YOLO model
        model = YOLO(model_path)
        subset_dir = getattr(self, f"{subset}_dir") / "images"  # Path to subset images
        # Run inference on the subset
        results = model.predict(source=str(subset_dir.resolve()), imgsz=self.img_size, save=True)
        print(f"Evaluation complete. Check predictions in {results}")


if __name__ == "__main__":
    # Initialize the pipeline with paths and configurations
    pipeline = ProbeDetectionPipeline(
        data_dir="./probe_dataset/probe_images",  # Directory containing images
        annotations_file="./probe_dataset/probe_labels.json",  # JSON file with annotations
        output_dir="processed_data",  # Directory to store processed data and results
        img_size=640  # Image size for training and inference
    )

    # Step 1: Parse annotations
    print("Parsing annotations...")
    annotations = pipeline.parse_annotations()

    # Step 2: Split and prepare dataset
    print("Preparing dataset...")
    pipeline.split_dataset(annotations)

    # Step 3: Create YOLO configuration
    print("Creating YOLO configuration...")
    pipeline.create_config()

    # Step 4: Train the YOLO model (comment this line to skip training)
    # print("Starting model training...")
    # pipeline.train_model(model_arch="yolov8n", epochs=50, batch_size=16)

    # Step 5: Evaluate the model
    print("Evaluating the model...")
    pipeline.evaluate_model(model_path="runs/detect/train/weights/best.pt", subset="test")
