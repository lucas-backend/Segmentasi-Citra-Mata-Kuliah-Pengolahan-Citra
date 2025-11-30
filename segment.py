import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. Fungsi Konvolusi Manual (Algoritma Inti) ---
def manual_convolution(image, kernel):
    # Mendapatkan dimensi
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    # Menghitung padding (titik pusat kernel)
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # Padding citra dengan nol
    # Khusus Roberts (2x2), paddingnya sedikit berbeda agar konsisten
    if k_h == 2:
        padded_img = np.pad(image, ((0, 1), (0, 1)), mode='constant')
    else:
        padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    output = np.zeros_like(image, dtype=float)
    
    # Proses Sliding Window
    for i in range(img_h):
        for j in range(img_w):
            # Mengambil region of interest (ROI)
            if k_h == 2: # Roberts
                roi = padded_img[i:i+2, j:j+2]
            else: # 3x3 kernels
                roi = padded_img[i:i+k_h, j:j+k_w]
            
            # Operasi konvolusi: Sum(ROI * Kernel)
            if roi.shape == kernel.shape:
                output[i, j] = np.sum(roi * kernel)
                
    return output

# --- 2. Fungsi Wrapper Edge Detection ---
def apply_edge_detection(image, method):
    # Pastikan input float untuk presisi perhitungan
    img_float = image.astype(float)
    
    # Definisi Kernel
    if method == 'Roberts':
        kx = np.array([[1, 0], [0, -1]])
        ky = np.array([[0, 1], [-1, 0]])
    elif method == 'Prewitt':
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    elif method == 'Sobel':
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif method == 'Frei-Chen':
        s2 = math.sqrt(2)
        kx = np.array([[-1, 0, 1], [-s2, 0, s2], [-1, 0, 1]])
        ky = np.array([[-1, -s2, -1], [0, 0, 0], [1, s2, 1]])
        
    # Konvolusi X dan Y
    grad_x = manual_convolution(img_float, kx)
    grad_y = manual_convolution(img_float, ky)
    
    # Magnitude Gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalisasi ke 0-255
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255
        
    return magnitude.astype(np.uint8)

# --- 3. Main Program & Layout Visualisasi ---

# Setup Data
files_portrait = ["portrait.jpg", "portrait-grey.jpg", "portrait-gaussian.jpg", "portrait-snp.jpg"]
files_landscape = ["landscape.jpg", "landscape-grey.jpg", "landscape-gaussian.jpg", "landscape-snp.jpg"]

# Dictionary untuk grouping agar loop lebih rapi
image_groups = {
    "Kelompok Portrait": files_portrait,
    "Kelompok Landscape": files_landscape
}

methods = ['Roberts', 'Prewitt', 'Sobel', 'Frei-Chen']

# Loop per Kelompok (Akan menghasilkan 2 Window terpisah)
for group_name, file_list in image_groups.items():
    
    # Membuat Figure grid 4x4 (4 Baris Metode x 4 Kolom Gambar)
    fig, axes = plt.subplots(4, 4, figsize=(14, 16))
    fig.suptitle(f'Hasil Segmentasi - {group_name}', fontsize=20, y=0.98)
    
    # Loop Kolom (Gambar)
    for col_idx, filename in enumerate(file_list):
        
        # Baca & Preprocessing Gambar
        img = cv2.imread(filename)
        
        # Handling jika file tidak ada (Dummy image)
        if img is None:
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.putText(img, "N/A", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        else:
            # Resize proporsional (width 300)
            h, w = img.shape[:2]
            aspect = h/w
            img = cv2.resize(img, (300, int(300*aspect)))
        
        # Convert ke Grayscale untuk diproses
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # Loop Baris (Metode)
        for row_idx, method in enumerate(methods):
            
            # Lakukan Deteksi Tepi Manual
            result_img = apply_edge_detection(img_gray, method)
            
            # Tampilkan pada subplot yang sesuai [baris, kolom]
            ax = axes[row_idx, col_idx]
            ax.imshow(result_img, cmap='gray')
            
            # --- Styling Label agar mirip referensi ---
            # Judul hanya di baris paling atas (nama file)
            if row_idx == 0:
                ax.set_title(filename, fontsize=10, pad=10)
            
            # Label Metode hanya di kolom paling kiri
            if col_idx == 0:
                ax.set_ylabel(method, fontsize=12, fontweight='bold', labelpad=10)
            
            # Hilangkan ticks angka agar bersih
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Beri ruang untuk judul utama
    
    # Tampilkan window saat ini (tutup window ini untuk memunculkan window berikutnya)
    plt.show()