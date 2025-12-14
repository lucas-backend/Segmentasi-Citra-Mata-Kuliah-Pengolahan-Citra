import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. Fungsi Konvolusi Manual (Algoritma Inti) ---
def manual_convolution(image, kernel):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    if k_h == 2:
        padded_img = np.pad(image, ((0, 1), (0, 1)), mode='constant')
    else:
        padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    output = np.zeros_like(image, dtype=float)
    
    for i in range(img_h):
        for j in range(img_w):
            if k_h == 2: # Roberts
                roi = padded_img[i:i+2, j:j+2]
            else: # 3x3 kernels
                roi = padded_img[i:i+k_h, j:j+k_w]
            
            if roi.shape == kernel.shape:
                output[i, j] = np.sum(roi * kernel)
                
    return output

# --- 2. Fungsi Wrapper Edge Detection ---
def apply_edge_detection(image, method):
    img_float = image.astype(float)
    
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
        
    grad_x = manual_convolution(img_float, kx)
    grad_y = manual_convolution(img_float, ky)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255
        
    return magnitude.astype(np.uint8)

# --- 3. Fungsi Visualisasi Grid ---
def process_and_display_batch(title, file_list):
    methods = ['Roberts', 'Prewitt', 'Sobel', 'Frei-Chen']
    
    # Membuat Figure grid 4x4 (4 baris metode, 4 kolom citra)
    fig, axes = plt.subplots(4, 4, figsize=(14, 16))
    fig.suptitle(f'Hasil Segmentasi - {title}', fontsize=20, y=0.98)
    
    # Dictionary untuk menyimpan hasil edge detection guna perhitungan MSE nanti
    results_cache = {m: [] for m in methods}
    
    # Loop Kolom (Gambar)
    for col_idx, filename in enumerate(file_list):
        
        # Baca & Preprocessing Gambar
        img = cv2.imread(filename)
        
        if img is None:
            # Placeholder jika gambar tidak ditemukan
            img_gray = np.zeros((100, 100), dtype=np.uint8)
        else:
            h, w = img.shape[:2]
            aspect = h/w
            img = cv2.resize(img, (300, int(300*aspect)))
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img

        # Loop Baris (Metode)
        for row_idx, method in enumerate(methods):
            
            # Algoritma Manual
            result_img = apply_edge_detection(img_gray, method)
            
            # Simpan hasil ke cache untuk MSE
            results_cache[method].append(result_img)
            
            # Plotting
            ax = axes[row_idx, col_idx]
            ax.imshow(result_img, cmap='gray')
            
            if row_idx == 0:
                ax.set_title(filename, fontsize=10, pad=10)
            
            if col_idx == 0:
                ax.set_ylabel(method, fontsize=12, fontweight='bold', labelpad=10)
            
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    return results_cache

# --- 4. Fungsi untuk Perhitungan MSE ---
def calculate_mse(imageA, imageB):
    # Pastikan ukuran sama sebelum hitung
    if imageA.shape != imageB.shape:
        return -1 
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# --- 5. Main Execution ---

# Daftar file spesifik Landscape
files_landscape = [
    "landscape.jpg",          # Index 0
    "landscape-grey.jpg",     # Index 1 (Anggap ini referensi Ground Truth)
    "landscape-gaussian.jpg", # Index 2
    "landscape-snp.jpg"       # Index 3
]

print("Memproses Kelompok Landscape...")
# Menjalankan visualisasi dan mengambil hasil olahan gambar
processed_results = process_and_display_batch("Kelompok Landscape", files_landscape)

print("\n--- Analisis Nilai MSE ---")
print("Perbandingan dilakukan terhadap hasil segmentasi 'landscape-grey.jpg' sebagai referensi.\n")

methods = ['Roberts', 'Prewitt', 'Sobel', 'Frei-Chen']

# Header Tabel
print(f"{'Metode':<12} | {'MSE Gaussian':<15} | {'MSE Salt & Pepper':<15}")
print("-" * 50)

for method in methods:
    # Mengambil hasil segmentasi dari cache
    # Index 1 adalah landscape-grey (referensi)
    ref_img = processed_results[method][1] 
    
    # Index 2 adalah landscape-gaussian
    gauss_img = processed_results[method][2]
    
    # Index 3 adalah landscape-snp
    snp_img = processed_results[method][3]
    
    # Hitung MSE
    mse_gauss = calculate_mse(ref_img, gauss_img)
    mse_snp = calculate_mse(ref_img, snp_img)
    
    print(f"{method:<12} | {mse_gauss:<15.2f} | {mse_snp:<15.2f}")