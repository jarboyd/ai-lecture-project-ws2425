import requests
import zipfile
from tqdm import tqdm

# URLs of the zip files
zip_file_urls = [
    'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip',
    'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip',
    'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip'
]

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
download_path_1 = "GTSRB_Final_Training_Images.zip"
download_path_2 = "GTSRB_Final_Test_GT.zip"
extract_path = "./Datasets"

# Download and unzip the files sequentially
for idx, url in enumerate(zip_file_urls):
    print(f"Downloading file {idx+1} from {url}")
    download_file_with_progress(url, download_path_1 if idx == 0 else download_path_2)
    print(f"Unpacking file {idx+1}")
    unzip_file_with_progress(download_path_1 if idx == 0 else download_path_2, extract_path)
    print(f"File {idx+1} downloaded and unpacked successfully!\n")