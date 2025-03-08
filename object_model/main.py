import helpers as h
import yolo_model as ym

if __name__ == "__main__":
    # Load a thermal image
    model = ym.YOLOModel("yolo11n.pt")
    file_path = "test_images_16_bit/image_2.tiff"
    image = h.thermal_image_to_array(file_path)
    model.predict(image)

