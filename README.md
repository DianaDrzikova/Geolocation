# Geolocation

The goal of visual geolocation is to estimate the GPS coordinates of a photograph using only visual data. This project aims to evaluate the performance of modern vision models, such as those trained on the Osv5M dataset, on images of natural scenes from the CrossLocate dataset. Selected models will be fine-tuned using the CrossLocate training split. The project will explore the characteristics and accuracy of GPS prediction through regression and classification.  

## File Structure

    Geolocation/
    ├── data/                  
    ├── requirements.txt        
    └── src/
        ├── generate_predictions.py  # Main script to run predictions
        ├── models/                  # Osv5M model implementation
        └── utils/                   # Osv5M utilities

Note: The `src/models/` and `src/utils/` directories are directly adapted from the official OSV5M Hugging Face implementation.

## Run

Example usage:

    python3.10 generate_predictions.py --tar_path ../data/query_photos.tar.gz --output_csv ../data/geolocations_uniform.csv

Arguments:

- --tar_path       Path to the input .tar.gz file containing images from CrossLocate Uniform dataset
- --output_csv     Path where the output CSV with geolocation predictions will be stored

## Output CSV Format

The output file looks like:

    image,latitude_radians,longitude_radians
    image1.png,0.87654,1.23456
    image2.jpg,0.65432,1.98765
    ...

## Requirements

Install dependencies with:

    pip install -r requirements.txt

## Important information

**OpenStreetView-5M** https://github.com/gastruc/osv5m

**CrossLocate** https://github.com/JanTomesek/CrossLocate

The CrossLocate Uniform dataset can be found on: 
https://cphoto.fit.vutbr.cz/crosslocate/dataset/datasets/uniform_dataset/

This project uses the Osv5M baseline model from Hugging Face:
https://huggingface.co/osv5m/baseline


