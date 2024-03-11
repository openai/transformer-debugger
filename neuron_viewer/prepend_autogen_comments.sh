#!/bin/bash

start_dir="./src/client"
prepend_string="// Auto-generated code. Do not edit! See neuron_explainer/activation_server/README.md to learn how to regenerate it."

# Find all files
find "$start_dir" -type f | while read file; do
    # Create a temporary file
    temp_file=$(mktemp)
    
    # Write the string to the temporary file
    echo "$prepend_string\n" > "$temp_file"

    # Concatenate the original file to the temporary file
    cat "$file" >> "$temp_file"
    
    # Replace the original file with the temporary file
    mv "$temp_file" "$file"
done
