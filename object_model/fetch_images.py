import requests
import os
import time
from datetime import datetime

IMAGE_URL = 'https://drive.google.com/file/d/1YyBsaV7HOfpaeqa6Af-6IvUv-DLp_8zr/view?usp=drive_link'

SAVE_DIR = '/home/trinhee/GM_automotive_challenge/object_model/test_images_16_bit'

if not os.path.exists(SAVE_DIR):
	os.makedirs(SAVE_DIR)

def download_image():
	try:
		response = requests.get(IMAGE_URL, stream=True)
		if response.status_code == 200:
			filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".tiff"
			filepath = os.path.join(SAVE_DIR, filename)

			with open(filepath, "wb") as file:
				for chunk in response.iter_content(1024):
					file.write(chunk)

			print(f"Image saved: {filepath}")
		else:
			print(f"Failed to download image, status code: {response.status_code}")

	except Exception as e:
		print(f"Error downloading image: {e}")

while True:
	download_image()
	time.sleep(5)
