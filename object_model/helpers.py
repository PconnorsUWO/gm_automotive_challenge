import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import ultralytics as u

def thermal_image_to_array(image_path):
    with Image.open(image_path) as img:
        img_array = np.array(img)
    return img_array


def apply_fft_thermal_image(image: np.ndarray) -> tuple:
    """
    Applies a 2D Fast Fourier Transform (FFT) on a thermal image and returns both the shifted FFT
    and its magnitude spectrum.

    Parameters
    ----------
    image : numpy.ndarray
        The input thermal image as a 2D array (grayscale).
    Returns
    -------
    fshift : numpy.ndarray
        The FFT of the image with zero frequency shifted to the center.
    magnitude_spectrum : numpy.ndarray
        The logarithmic magnitude spectrum of the FFT, which helps visualize the frequency content.

    Example
    -------
    >>> import cv2
    >>> image = cv2.imread('thermal_image.png', cv2.IMREAD_GRAYSCALE)
    >>> fshift, magnitude_spectrum = apply_fft_thermal_image(image)
    >>> plt.imshow(magnitude_spectrum, cmap='gray')
    >>> plt.title('Magnitude Spectrum')
    >>> plt.show()
    """
    # Compute the 2D FFT of the image
    f = np.fft.fft2(image)
    
    # Shift the zero frequency component to the center
    fshift = np.fft.fftshift(f)
    
    # Compute the magnitude spectrum (using a logarithmic scale for better visualization)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    
    return fshift, magnitude_spectrum

def display_fft_image(image):
    """
    Compute and display the 2D FFT of a thermal image.

    Parameters:
    image (np.array): 2D numpy array representing the thermal image.
    """
    # Compute the 2D FFT of the image
    fft2d = np.fft.fft2(image)
    # Shift the zero frequency component to the center
    fft2d_shifted = np.fft.fftshift(fft2d)
    # Compute the magnitude spectrum with logarithmic scaling
    magnitude_spectrum = np.log(1 + np.abs(fft2d_shifted))
    
    # Plot the magnitude spectrum
    plt.figure(figsize=(8, 8))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('2D FFT Magnitude Spectrum')
    plt.axis('off')
    plt.show()

def extract_fft_features(image):
    """
    Compute the 2D FFT of an image and extract a feature vector from the magnitude spectrum.
    
    Parameters:
    image (np.array): Grayscale image as a 2D array.
    
    Returns:
    features (np.array): A feature vector representing the image's frequency domain.
    """
    # Compute the 2D FFT
    fft2d = np.fft.fft2(image)
    # Shift zero-frequency component to the center
    fft2d_shifted = np.fft.fftshift(fft2d)
    # Compute the magnitude spectrum and apply logarithmic scaling
    magnitude_spectrum = np.log(1 + np.abs(fft2d_shifted))
    
    # Option 1: Flatten the entire magnitude spectrum (may be high-dimensional)
    features = magnitude_spectrum.flatten()
    
    # Option 2: Alternatively, you can compute summary statistics or downsample the spectrum
    # For example, by taking a central crop or resizing:
    # features = cv2.resize(magnitude_spectrum, (32, 32)).flatten()
    
    return features
def display_thermal_data(image: np.ndarray, auto_scale: bool = True) -> None:
    """
    Displays thermal image data using a thermal colormap.
    
    Parameters
    ----------
    image : np.ndarray
        A 2D NumPy array representing the thermal image data.
    auto_scale : bool, optional
        If True, the display scales to the image's min and max values.
        If False, it uses the full 16-bit range (0 to 65535). Default is True.
    """
    plt.figure(figsize=(8, 6))
    
    # Determine display range: use auto scaling or assume full 16-bit range.
    if auto_scale:
        vmin, vmax = image.min(), image.max()
    else:
        vmin, vmax = 0, 65535
    
    plt.imshow(image, cmap='inferno', vmin=vmin, vmax=vmax)
    plt.title("Thermal Image Data")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.colorbar(label="Intensity")
    plt.axis("off")
    plt.show()

def get_exif_metadata(file_path: str) -> dict:
    """
    Extracts EXIF metadata from an image file.

    Parameters
    ----------
    file_path : str
        Path to the image file.

    Returns
    -------
    dict
        A dictionary of EXIF metadata with human-readable tag names.
    """
    metadata = {}
    with Image.open(file_path) as img:
        exif_data = img.getexif()
        if exif_data:
            metadata = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    return metadata

# Example usage:
if __name__ == "__main__":
    # Load a thermal image
    file_path = "test_images_16_bit/image_2.tiff"
    thermal_image = thermal_image_to_array(file_path)
    display_thermal_data(thermal_image)