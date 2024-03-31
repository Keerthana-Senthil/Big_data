#!/bin/bash

# Define function
check_archive_validity() {
    archive_path="$1"
    destination_path="$2"
    
    # Print the name of the archive file
    echo "Processing archive: $archive_path"
    
    # Check if the file exists
    if [ -f "$archive_path" ]; then
        # Check if the file is a valid ZIP archive
        if /usr/bin/unzip -t "$archive_path" ; then
            # Unzip the contents into individual CSV files
            echo "Archive is valid"
            /usr/bin/unzip "$archive_path" -d "$destination_path"
            
            echo "Archive is valid and contents have been extracted."
            # Exit with success
            exit 0
        else
            echo "Error: The file is not a valid ZIP archive."
            exit 1
        fi
    else
        echo "Error: File not found."
        exit 1
    fi
}

# Call function with arguments passed from Python
check_archive_validity "$1" "$2"
