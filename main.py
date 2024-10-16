from PIL import Image
import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

# Preprocessing function for uploaded image
def preprocess_image(uploaded_image, size=(64, 64)):  # Set size to 64x64
    image = Image.open(uploaded_image)
    
    # Convert the PIL image to a NumPy array (OpenCV format)
    img = np.array(image)

    # Resize the image using OpenCV
    img_resized = cv2.resize(img, size)

    # Convert the image to HSV color space
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)

    return img_resized, img_hsv

# Feature extraction functions
def extract_color_features(image_hsv):
    h_channel = image_hsv[:, :, 0].flatten()
    s_channel = image_hsv[:, :, 1].flatten()
    v_channel = image_hsv[:, :, 2].flatten()
    return np.concatenate([h_channel, s_channel, v_channel])

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

# Calculate silhouette score on subset to save memory
def calculate_silhouette_on_subset(X, labels, subset_size=10000):  # Use a smaller subset
    if X.shape[0] > subset_size:
        X, labels = resample(X, labels, n_samples=subset_size, random_state=42)
    return silhouette_score(X, labels)

# Plotting function for clustered images using mean colors from the input image
def plot_clustered_image_with_mean_colors(image, labels, k):
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
        else:
            mean_color = np.random.randint(0, 255, 3)  # Fallback color for empty clusters
        mean_colors.append(mean_color)

    # Apply the mean colors to each cluster in the output image
    for i in range(h):
        for j in range(w):
            cluster_id = labels_reshaped[i, j]
            image_copy[i, j] = mean_colors[cluster_id]

    return image_copy

# Process and plot function
def processPlotting(image, k):
    # Preprocess the image for feature extraction
    img_resized, img_hsv = preprocess_image(image, size=(64, 64))
    img_flattened = img_resized.reshape(-1, 3).astype(np.float32)  # Flatten image to (h * w, 3)

    # Extract features for clustering
    color_features = extract_color_features(img_hsv)
    texture_features = extract_texture_features_opencv(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY))
    spatial_features = extract_spatial_features(img_resized)

    # Combine features
    features = np.concatenate([color_features, texture_features, spatial_features])

    # Perform KMeans clustering for the user-selected number of clusters
    labels_kmeans, _ = kmeans_manual(img_flattened, k)

    # Check if the clustering produced more than one label
    if len(np.unique(labels_kmeans)) > 1:
        # Calculate silhouette score on a smaller subset to save memory
        silhouette_kmeans = calculate_silhouette_on_subset(img_flattened, labels_kmeans)

        # Plot clustered image using mean colors from the original image
        clustered_image = plot_clustered_image_with_mean_colors(img_resized, labels_kmeans, k)
        return clustered_image, silhouette_kmeans
    else:
        return None, None

# Streamlit app setup
st.title("Image Upload and Clustering with KMeans")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Add a number input for the number of clusters
num_clusters = st.number_input("Select the number of clusters", min_value=2, max_value=10, value=3)

# Layout for the images
col1, col2 = st.columns(2)


with col1:
    st.subheader("Uploaded Image")
    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    else:
        st.write("Please upload an image.")

with col2:
    st.subheader("Clustering Plot")
    if uploaded_file:
        if st.button("Run Clustering"):
            with st.spinner("Clustering in progress..."):
                clustered_image, silhouette_score = processPlotting(uploaded_file, num_clusters)
            
            if clustered_image is not None:
                st.image(clustered_image, caption=f"KMeans - {num_clusters} Clusters", use_column_width=True)
                st.success(f"Clustering completed! Silhouette Score: {silhouette_score:.4f}")
            else:
                st.write(f"KMeans - {num_clusters} Clusters: Only one cluster found, skipping silhouette calculation.")
    else:
        st.write("Please upload an image first.")
