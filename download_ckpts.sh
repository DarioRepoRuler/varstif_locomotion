#!/bin/bash

# Check if a download folder name is provided as an argument
if [ $# -eq 0 ]; then
  echo "Please provide a download folder name as an argument."
  exit 1
fi

download_folder_name="$1"

# Create the local download directory if it doesn't exist
local_dir="outputs/$download_folder_name"
mkdir -p "$local_dir"

# SFTP to the remote server and download the specified folder
sftp dspoljaric@eda01 << EOF
cd TAvic/outputs/$download_folder_name
lcd "$local_dir"
get -r .
quit
EOF