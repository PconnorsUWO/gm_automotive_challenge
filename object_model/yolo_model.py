from ultralytics import YOLO, settings

class YOLOModel:
    def __init__(self, model_path='yolo11n.pt'):
        """
        Initializes the YOLO model.
        
        Parameters:
            model_path (str): Path to the model weights file.
        """

        self.model = YOLO("yolo11n.pt")
    
    def predict(self, image):
        """
        Runs inference on the given image.
        
        Parameters:
            image: Input image. It can be a file path, PIL image, numpy array, etc., 
                   as supported by the YOLOv5 model.
        
        Returns:
            The model's prediction results.
        """
        results = self.model(image)
        # You can further process results (e.g., extract bounding boxes, classes, etc.)
        return results
    
    def show_settings(self):
        """
        Prints the current settings of the YOLO model.
        """
        return settings