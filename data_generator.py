import googlemaps
import csv
from PIL import Image
import torch
from torchvision import models, transforms
import os

# Initialize Google Maps client
gmaps = googlemaps.Client(key="need to get free API key")

# Define the feature extraction model
model = models.resnet50(pretrained=True)
model.eval()

# Function to extract image features
def extract_features(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().tolist()

# Fetch geolocation data and images
def fetch_data(location_query, output_dir="data"):
    geocode_result = gmaps.geocode(location_query)
    if not geocode_result:
        print(f"No results for {location_query}")
        return None, None

    lat_lng = geocode_result[0]["geometry"]["location"]
    lat, lng = lat_lng["lat"], lat_lng["lng"]

    # Download a Google Street View image (use a dummy local image for simplicity here)
    # Example: Replace the below path with a valid API call or local file for testing
    image_path = f"{output_dir}/{location_query.replace(' ', '_')}.jpg"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    Image.new("RGB", (256, 256), color="blue").save(image_path)  # Replace with actual image fetching logic

    return extract_features(image_path), (lat, lng)

# Generate dataset
def generate_dataset(queries, feature_file="features.csv", label_file="labels.csv"):
    with open(feature_file, "w", newline="") as feat_f, open(label_file, "w", newline="") as label_f:
        feature_writer = csv.writer(feat_f)
        label_writer = csv.writer(label_f)

        for query in queries:
            features, lat_lng = fetch_data(query)
            if features and lat_lng:
                feature_writer.writerow(features)
                label_writer.writerow(lat_lng)

if __name__ == "__main__":
    locations = ["New York, USA", "Paris, France", "Tokyo, Japan", "Sydney, Australia"]
    generate_dataset(locations)
