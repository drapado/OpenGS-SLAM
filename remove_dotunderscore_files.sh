#!/bin/bash

# Script to remove all files starting with ._ (like macOS metadata files) from a folder and its subfolders
# Usage: ./remove_dotunderscore_files.sh [FOLDER_PATH]

# Function to display usage
usage() {
    echo "Usage: $0 [FOLDER_PATH]"
    echo "  FOLDER_PATH: Path to the folder to clean (default: current directory)"
    echo ""
    echo "This script will recursively remove all files starting with ._ from the specified folder."
    echo ""
    echo "Examples:"
    echo "  $0                    # Clean current directory"
    echo "  $0 /path/to/folder    # Clean specified folder"
    echo "  $0 ./test             # Clean test folder"
}

# Function to remove ._ files
remove_dotunderscore_files() {
    local target_dir="$1"
    
    # Check if directory exists
    if [ ! -d "$target_dir" ]; then
        echo "Error: Directory '$target_dir' does not exist!"
        exit 1
    fi
    
    echo "Searching for files starting with ._ in: $target_dir"
    echo ""
    
    # Find and count files starting with ._
    local files_found=$(find "$target_dir" -name "._*" -type f 2>/dev/null | wc -l)
    
    if [ "$files_found" -eq 0 ]; then
        echo "No files starting with ._ found in $target_dir"
        exit 0
    fi
    
    echo "Found $files_found files starting with ._"
    echo ""
    
    # Ask for confirmation
    read -p "Do you want to remove all these files? (y/N): " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        echo ""
        echo "Removing files..."
        
        # Find and remove files starting with ._
        find "$target_dir" -name "._*" -type f -print0 | while IFS= read -r -d '' file; do
            echo "Removing: $file"
            rm "$file"
        done
        
        echo ""
        echo "✓ All files starting with ._ have been removed from $target_dir"
    else
        echo "Operation cancelled."
    fi
}

# Main script logic
main() {
    # Show help if requested
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi
    
    # Set target directory (default to current directory if not provided)
    local target_dir="${1:-.}"
    
    # Convert to absolute path
    target_dir=$(realpath "$target_dir" 2>/dev/null || echo "$target_dir")
    
    # Call the removal function
    remove_dotunderscore_files "$target_dir"
}

# Run the main function with all arguments
main "$@"
