import os
import helpers
import numpy as np
import rf_model as rf
from ultralytics import YOLO


model = YOLO("yolo11n.pt")
rf_model = rf.RFModelWrapper("labeled_data.txt")

def process_directory(input_dir, results_path):
    # Define supported image extensions.
    image_extensions = ('.jpeg', '.tiff')
    # Get all image files in the input directory and sort them.
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)])
    
    with open(results_path, 'w') as out_file:
        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            image = helpers.thermal_image_to_array(img_path)
            if image is None:
                print(f"Warning: Couldn't load image {img_path}")
                continue
            
            desired_classes = [0, 1, 2]
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            elif image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            image = image.astype('uint8')

            # Run prediction.
            results = model.predict(source=image, classes=desired_classes, conf=0.10)
            # Access the boxes property which contains bounding boxes, class labels, and confidences.
            boxes = results[0].boxes
            boundaries_arr = boxes.xyxy  # Bounding box coordinates
            rf_model_predictions = [rf_model.predict(*boundary) for boundary in boundaries_arr]

            # Iterate over each detected object.
            for i in range(len(boundaries_arr)):
                # Convert bounding box coordinates to list if needed.
                boundary = boundaries_arr[i].tolist() if hasattr(boundaries_arr[i], 'tolist') else boundaries_arr[i]
                class_id, confidence = rf_model_predictions[i]
                out_file.write(f"{img_file} {class_id} {confidence} {int(boundary[0])} {int(boundary[1])} {(int(boundary[2]))} {int(boundary[3])}\n")

if __name__ == '__main__':
    process_directory("test_images_16_bit", "results.txt")