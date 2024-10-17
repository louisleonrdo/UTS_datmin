from PIL import Image
import streamlit as st
import cv2
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

# Fungsi preprocessing untuk gambar yang diunggah (tanpa resizing)
def preprocess_image(uploaded_image): 
    image = Image.open(uploaded_image)
    
    # Mengonversi gambar PIL menjadi array NumPy (format OpenCV)
    img = np.array(image)

    # Mengonversi gambar ke ruang warna HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    return img, img_hsv

# Fungsi ekstraksi fitur
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

# Fungsi KMeans clustering
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

# Normalisasi fitur
def normalize_features(features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

# Menghitung skor silhouette pada subset untuk menghemat memori
def calculate_silhouette_on_subset(X, labels, subset_size=10000):  # Menggunakan subset yang lebih kecil
    if X.shape[0] > subset_size:
        X, labels = resample(X, labels, n_samples=subset_size, random_state=42)
    return silhouette_score(X, labels)

# Fungsi untuk memplot hasil cluster dengan warna rata-rata dari gambar asli tanpa nomor cluster
def plot_clustered_image_without_numbers(image, labels, k):
    h, w, _ = image.shape
    image_copy = np.zeros_like(image)

    # Mengubah bentuk label sesuai dengan bentuk gambar 2D (h, w)
    labels_reshaped = labels.reshape(h, w)

    # Menghitung warna rata-rata dari setiap cluster dari gambar asli
    mean_colors = []
    for i in range(k):
        cluster_pixels = image[labels_reshaped == i]
        if len(cluster_pixels) > 0:
            mean_color = cluster_pixels.mean(axis=0)
            mean_colors.append(mean_color)
        else:
            mean_color = np.random.randint(0, 255, 3)
            mean_colors.append(mean_color)

    # Menerapkan warna rata-rata ke gambar
    for i in range(h):
        for j in range(w):
            cluster_id = labels_reshaped[i, j]
            image_copy[i, j] = mean_colors[cluster_id]

    return image_copy, mean_colors

# Fungsi bantu untuk mengonversi RGB ke HEX
def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))

# Fungsi untuk proses dan plot dengan normalisasi
def processPlotting(image, num_clusters):
    # Preprocessing gambar untuk ekstraksi fitur
    img_resized, img_hsv = preprocess_image(image)
    h, w, _ = img_resized.shape
    img_flattened = img_resized.reshape(-1, 3).astype(np.float32)

    # Ekstraksi fitur untuk clustering
    color_features = extract_color_features(img_hsv)
    texture_features = extract_texture_features_opencv(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY))
    spatial_features = extract_spatial_features(img_resized)

    # Menggabungkan fitur
    features = np.concatenate([color_features, np.repeat([texture_features], len(color_features), axis=0), 
                               np.repeat([spatial_features], len(color_features), axis=0)], axis=1)
    
    # Normalisasi fitur sebelum clustering
    normalized_features = normalize_features(features)

    # Melakukan KMeans clustering
    labels_kmeans, _ = kmeans_manual(normalized_features, num_clusters)

    # Mengubah bentuk label sesuai dengan bentuk gambar asli
    labels_kmeans = labels_kmeans.reshape(h, w)

    # Memeriksa apakah clustering menghasilkan lebih dari satu label
    if len(np.unique(labels_kmeans)) > 1:
        silhouette_kmeans = calculate_silhouette_on_subset(normalized_features, labels_kmeans.flatten())
        st.write(f"KMeans - {num_clusters} Clusters: Silhouette: {silhouette_kmeans:.4f}")

        # Memplot gambar hasil clustering dengan warna rata-rata tanpa nomor cluster
        clustered_image, mean_colors = plot_clustered_image_without_numbers(img_resized, labels_kmeans, num_clusters)
        st.image(clustered_image, caption=f"KMeans - {num_clusters} Clusters", use_column_width=True)

        # Menampilkan warna cluster dengan swatch warna yang sesuai
        st.subheader("Warna Cluster:")
        for i, color in enumerate(mean_colors):
            hex_color = rgb_to_hex(color)
            st.color_picker(f"Warna Cluster {i + 1}:", hex_color, disabled=True)
    else:
        st.write(f"KMeans - {num_clusters} Clusters: Hanya ditemukan satu cluster, melewati perhitungan silhouette.")

# Pengaturan aplikasi Streamlit
st.title("Upload Gambar dan Clustering dengan KMeans")

# Pengunggah file untuk beberapa gambar
uploaded_files = st.file_uploader("Pilih gambar...", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Memungkinkan pengguna untuk memilih jumlah cluster
num_clusters = st.slider('Pilih jumlah cluster', min_value=2, max_value=10, value=3)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader("Gambar yang Diunggah")
        # Menampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

        # Tombol untuk memulai clustering
        if st.button(f"Jalankan Clustering pada {uploaded_file.name}"):
            processPlotting(uploaded_file, num_clusters)
