import helpers as h
import yolo_model as ym
import numpy as np

if __name__ == "__main__":
    # Load a thermal image
    model = ym.YOLOModel("yolo11n.pt")
    file_path = "test_images_16_bit/image_2.tiff"
    image = h.thermal_image_to_array(file_path)

    # Convert thermal image (single channel) to a 3-channel image if needed.
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # Ensure image is of type uint8
    image = image.astype('uint8')

    # Now run prediction
    predictions = model.predict(image)
    predictions[0].show()
