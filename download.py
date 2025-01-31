import requests
import zipfile
from tqdm import tqdm

# URL of the zip file
zip_file_url = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"


# Function to download the file with progress bar
def download_file_with_progress(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(dest_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


# Function to unzip the file with progress bar
def unzip_file_with_progress(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        total_files = len(zip_ref.infolist())
        progress_bar = tqdm(total=total_files, unit="file")

        for file_info in zip_ref.infolist():
            zip_ref.extract(file_info, extract_to)
            progress_bar.update(1)
        progress_bar.close()


# Paths
download_path = "FullIJCNN2013.zip"
extract_path = ""

# Download and unzip the file
print(f"Downloading GTSDB dataset from {zip_file_url}")
download_file_with_progress(zip_file_url, download_path)
print("Unpacking the dataset")
unzip_file_with_progress(download_path, extract_path)
