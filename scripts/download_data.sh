#!/bin/bash

# This script downloads and extracts the WebNLG dataset.

# Create the data directory if it doesn't exist
mkdir -p data
cd data

# URL for the WebNLG 3.0 dataset
URL="https://gitlab.com/shimorina/webnlg-dataset/-/archive/master/webnlg-dataset-master.zip?path=webnlg-dataset-v3.0/en"

# Download the zip file
echo "Downloading WebNLG dataset..."
wget -O webnlg.zip "$URL"

# Unzip the file
echo "Unzipping dataset..."
unzip webnlg.zip

# Clean up the directory structure
# The files are nested in a weird way, let's move them to a cleaner path
mv webnlg-dataset-master-webnlg-dataset-v3.0-en/webnlg-dataset-v3.0 webnlg
rm -rf webnlg-dataset-master-webnlg-dataset-v3.0-en
rm webnlg.zip

echo "Dataset downloaded and extracted to data/webnlg/"
