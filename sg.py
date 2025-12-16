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

# --- 5. Fungsi Visualisasi Grafik MSE ---
def plot_mse_comparison(methods, mse_gauss, mse_snp):
    x = np.arange(len(methods))  # Lokasi label
    width = 0.35  # Lebar bar

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, mse_gauss, width, label='Gaussian Noise', color='skyblue')
    rects2 = ax.bar(x + width/2, mse_snp, width, label='Salt & Pepper Noise', color='salmon')

    # Menambahkan teks label, judul, dan konfigurasi sumbu
    ax.set_ylabel('Nilai Mean Squared Error (MSE)')
    ax.set_title('Perbandingan MSE Berdasarkan Metode Deteksi Tepi')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    # Menambahkan label nilai di atas bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

# --- 6. Main Execution ---

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

# List untuk menyimpan data guna plotting grafik
mse_gauss_list = []
mse_snp_list = []

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
    
    # Simpan ke list
    mse_gauss_list.append(mse_gauss)
    mse_snp_list.append(mse_snp)
    
    print(f"{method:<12} | {mse_gauss:<15.2f} | {mse_snp:<15.2f}")

# Panggil fungsi plotting grafik MSE
print("\nMenampilkan Grafik Perbandingan MSE...")
plot_mse_comparison(methods, mse_gauss_list, mse_snp_list)


# Tambahan

# --- 7. Fungsi Filter Manual ---

def manual_mean_filter(image):
    # Filter Mean pada dasarnya adalah konvolusi dengan kernel rata-rata
    # Kita menggunakan kembali fungsi manual_convolution yang sudah dibuat
    kernel = np.ones((3, 3), dtype=float) / 9.0
    return manual_convolution(image, kernel).astype(np.uint8)

def manual_median_filter(image, kernel_size=3):
    h, w = image.shape
    pad = kernel_size // 2
    
    # Padding citra agar ukuran tetap sama
    padded_img = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')
    output = np.zeros_like(image)
    
    # Sliding window manual
    for i in range(h):
        for j in range(w):
            # Ambil area tetangga (window)
            window = padded_img[i:i+kernel_size, j:j+kernel_size]
            
            # Ubah array 2D menjadi 1D list
            flat_window = window.flatten()
            
            # Urutkan nilai pixel (sorting manual bisa dilakukan, tapi np.sort diperbolehkan)
            sorted_window = np.sort(flat_window)
            
            # Ambil nilai tengah
            median_val = sorted_window[len(sorted_window) // 2]
            output[i, j] = median_val
            
    return output

# --- 8. Fungsi Helper Resize (Agar konsisten) ---
def load_and_resize(filename, width=300):
    img = cv2.imread(filename)
    if img is None:
        return None
    h, w = img.shape[:2]
    aspect = h/w
    img = cv2.resize(img, (width, int(width*aspect)))
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# --- 9. Eksekusi Utama (Perhitungan MSE & Visualisasi Grafik) ---

# Load semua gambar dan resize di awal agar ukuran seragam
img_ref_gray = load_and_resize("landscape-grey.jpg")
img_gauss = load_and_resize("landscape-gaussian.jpg")
img_snp = load_and_resize("landscape-snp.jpg")

# Validasi File
if (img_ref_gray is None) or (img_gauss is None) or (img_snp is None):
    print("Error Salah satu file gambar tidak ditemukan.")
else:
    # A. Pra-pemrosesan Filter (Denoising)
    print("Sedang melakukan filtering manual (Mean & Median)...")
    
    # Terapkan Filter Median & Mean pada Salt & Pepper
    img_snp_median = manual_median_filter(img_snp)
    img_snp_mean = manual_mean_filter(img_snp)

    methods = ['Roberts', 'Prewitt', 'Sobel', 'Frei-Chen']
    
    # Siapkan dictionary untuk menampung data plot
    plot_data = {
        'S&P Raw': [],
        'Gaussian Raw': [],
        'S&P Median': [],
        'S&P Mean': []
    }
    
    print("\n--- Hasil Perbandingan MSE (4 Skenario) ---")
    
    # Header Tabel
    header = f"{'Metode':<10} | {'1. S&P Raw':<12} | {'2. Gauss Raw':<12} | {'3. S&P Median':<12} | {'4. S&P Mean':<12}"
    print(header)
    print("-" * 70)

    for method in methods:
        # 1. Segmentasi Citra Referensi (Ground Truth)
        seg_ref = apply_edge_detection(img_ref_gray, method)
        
        # 2. Segmentasi Citra Salt & Pepper (Tanpa Filter)
        seg_snp = apply_edge_detection(img_snp, method)
        
        # 3. Segmentasi Citra Gaussian (Tanpa Filter)
        seg_gauss = apply_edge_detection(img_gauss, method)
        
        # 4. Segmentasi Citra S&P + Filter Median
        seg_snp_median = apply_edge_detection(img_snp_median, method)
        
        # 5. Segmentasi Citra S&P + Filter Mean
        seg_snp_mean = apply_edge_detection(img_snp_mean, method)
        
        # Hitung MSE (FIX: Gunakan float untuk menghindari overflow uint8)
        mse_1 = np.mean((seg_ref.astype(float) - seg_snp.astype(float)) ** 2)
        mse_2 = np.mean((seg_ref.astype(float) - seg_gauss.astype(float)) ** 2)
        mse_3 = np.mean((seg_ref.astype(float) - seg_snp_median.astype(float)) ** 2)
        mse_4 = np.mean((seg_ref.astype(float) - seg_snp_mean.astype(float)) ** 2)
        
        # Simpan data ke dictionary untuk plotting
        plot_data['S&P Raw'].append(mse_1)
        plot_data['Gaussian Raw'].append(mse_2)
        plot_data['S&P Median'].append(mse_3)
        plot_data['S&P Mean'].append(mse_4)
        
        # Print baris tabel
        row_str = f"{method:<10} | {mse_1:<12.2f} | {mse_2:<12.2f} | {mse_3:<12.2f} | {mse_4:<12.2f}"
        print(row_str)

    # --- B. Visualisasi Grafik Perbandingan ---
    print("\nMenampilkan Grafik Perbandingan...")
    
    x = np.arange(len(methods))  # Lokasi label sumbu X
    width = 0.2  # Lebar setiap bar
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Membuat 4 bar berdampingan
    # Posisi diatur dengan offset dari x (misal: x - 1.5*width)
    rects1 = ax.bar(x - 1.5*width, plot_data['S&P Raw'], width, label='1. S&P Raw', color='#ff9999')
    rects2 = ax.bar(x - 0.5*width, plot_data['Gaussian Raw'], width, label='2. Gaussian Raw', color='#66b3ff')
    rects3 = ax.bar(x + 0.5*width, plot_data['S&P Median'], width, label='3. S&P Median', color='#99ff99')
    rects4 = ax.bar(x + 1.5*width, plot_data['S&P Mean'], width, label='4. S&P Mean', color='#ffcc99')

    # Label dan Judul
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('Perbandingan Efektivitas Filter dan Noise pada Deteksi Tepi')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.show()