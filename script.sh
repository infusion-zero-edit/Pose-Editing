#!/bin/bash

# List of class names and corresponding input images
class_names=("office chair" "table, flower, book" "lamp" "chair" "chair_1" "sofa" "flower vase, flower" "laptop")
input_images=("./inputs/office_chair.jpg" "./inputs/table.jpg" "./inputs/lamp.jpg" "./inputs/chair.jpg" "./inputs/chair_1.jpg" "./inputs/sofa.jpg" "./inputs/flower_vase.jpg" "./inputs/laptop.jpg")

# Iterate over class names and input images
for ((i=0; i<${#class_names[@]}; i++)); do
    class_name="${class_names[i]}"
    input_image="${input_images[i]}"
    output_file="generated_${class_name// /_}.png"  # Replace spaces with underscores in class name

    # Run the Python script with the specified parameters
    python task1.py --image "$input_image" --class_name "$class_name" --output "$output_file"

    echo "Processed: $class_name -> $output_file"
done