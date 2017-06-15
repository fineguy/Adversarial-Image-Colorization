# Creating the dataset

Please use `python <script_name> -h` to see the exact parameters for each script.

# Step 1. Download images from Flickr

- `python download.py --key <API_key> --secrete <API_secret>`


This will download all images with the specified tag from https://www.flickr.com and randomly split them into training, validation and testing folders.


# Step 2. Build HDF5 images dataset

`python make_dataset.py <dir_name>`

This will create an HDF5 images dataset from the images in your `<dir_name>` folder. It must contain `train`, `valid` and `test` subfolders.

# Step 3. Upload dataset to floydhub from Google Drive

`python upload.py --id <google_drive_id>`
