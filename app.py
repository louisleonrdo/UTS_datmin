from PIL import Image
import streamlit as st
import cv2
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

# Preprocessing function for uploaded image (no resizing)
def preprocess_image(uploaded_image): 
    image = Image.open(uploaded_image)
    
    # Convert the PIL image to a NumPy array (OpenCV format)
    img = np.array(image)

    # Convert the image to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    return img, img_hsv

# Feature extraction functions
def extract_color_features(image_hsv):
    h_channel = image_hsv[:, :, 0].flatten()
    s_channel = image_hsv[:, :, 1].flatten()
    v_channel = image_hsv[:, :, 2].flatten()
    return np.stack([h_channel, s_channel, v_channel], axis=1)

def extract_texture_features_opencv(image_gray):
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    mean_lap = np.mean(laplacian)
    var_lap = np.var(laplacian)
    return np.array([mean_lap, var_lap], dtype=np.float32)

def extract_spatial_features(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(image)
    cx = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0
    cy = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 0
    return np.array([cx, cy], dtype=np.float32)

# KMeans clustering functions
def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2, dtype=np.float32))
    return np.argmin(distances, axis=0)

def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            new_centroids.append(X[np.random.choice(X.shape[0])])
    return np.array(new_centroids, dtype=np.float32)

def kmeans_manual(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Normalize the features
def normalize_features(features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

# Calculate silhouette score on subset to save memory
def calculate_silhouette_on_subset(X, labels, subset_size=10000):  # Use a smaller subset
    if X.shape[0] > subset_size:
        X, labels = resample(X, labels, n_samples=subset_size, random_state=42)
    return silhouette_score(X, labels)

# Function to plot clusters with mean colors from input image without numbering
def plot_clustered_image_without_numbers(image, labels, k):
    h, w, _ = image.shape
    image_copy = np.zeros_like(image)

    # Reshape labels to match the 2D image shape (h, w)
    labels_reshaped = labels.reshape(h, w)

    # Calculate the mean color of each cluster from the original image
    mean_colors = []
    for i in range(k):
        cluster_pixels = image[labels_reshaped == i]
        if len(cluster_pixels) > 0:
            mean_color = cluster_pixels.mean(axis=0)
            mean_colors.append(mean_color)
        else:
            mean_color = np.random.randint(0, 255, 3)
            mean_colors.append(mean_color)

    # Apply mean colors to the image
    for i in range(h):
        for j in range(w):
            cluster_id = labels_reshaped[i, j]
            image_copy[i, j] = mean_colors[cluster_id]

    return image_copy, mean_colors

# Helper function to convert RGB to HEX
def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))

# Process and plot function with normalization
def processPlotting(image, num_clusters):
    # Preprocess the image for feature extraction
    img_resized, img_hsv = preprocess_image(image)
    h, w, _ = img_resized.shape
    img_flattened = img_resized.reshape(-1, 3).astype(np.float32)

    # Extract features for clustering
    color_features = extract_color_features(img_hsv)
    texture_features = extract_texture_features_opencv(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY))
    spatial_features = extract_spatial_features(img_resized)

    # Combine features
    features = np.concatenate([color_features, np.repeat([texture_features], len(color_features), axis=0), 
                               np.repeat([spatial_features], len(color_features), axis=0)], axis=1)
    
    # Normalize the features before clustering
    normalized_features = normalize_features(features)

    # Perform KMeans clustering
    labels_kmeans, _ = kmeans_manual(normalized_features, num_clusters)

    # Reshape labels to the original image shape
    labels_kmeans = labels_kmeans.reshape(h, w)

    # Check if clustering produced more than one label
    if len(np.unique(labels_kmeans)) > 1:
        silhouette_kmeans = calculate_silhouette_on_subset(normalized_features, labels_kmeans.flatten())
        st.write(f"KMeans - {num_clusters} Clusters: Silhouette: {silhouette_kmeans:.4f}")

        # Plot clustered image using mean colors without numbering the clusters
        clustered_image, mean_colors = plot_clustered_image_without_numbers(img_resized, labels_kmeans, num_clusters)
        st.image(clustered_image, caption=f"KMeans - {num_clusters} Clusters", use_column_width=True)

        # Display cluster colors with actual color swatches
        st.subheader("Cluster Colors:")
        for i, color in enumerate(mean_colors):
            hex_color = rgb_to_hex(color)
            st.color_picker(f"Cluster {i + 1} Color:", hex_color, disabled=True)
    else:
        st.write(f"KMeans - {num_clusters} Clusters: Only one cluster found, skipping silhouette calculation.")

# Streamlit app setup
st.title("Image Upload and Clustering with KMeans")

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose image(s)...", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Allow the user to select the number of clusters
num_clusters = st.slider('Select the number of clusters', min_value=2, max_value=10, value=3)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader("Uploaded Image")
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Button to trigger clustering
        if st.button(f"Run Clustering on {uploaded_file.name}"):
            processPlotting(uploaded_file, num_clusters)
