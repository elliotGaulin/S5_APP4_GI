import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane


def remove_aberations(img):
    zeros = [0, -0.99, -0.99, 0.8]
    poles = [0.9 * np.e**(1j*np.pi/2), 0.9 * np.e**(-1j*np.pi/2), 0.95 * np.e**(1j*np.pi/8), 0.95 * np.e**(-1j*np.pi/8)]
    
    num = np.poly(zeros)
    denum = np.poly(poles)
    plt.figure()
    zplane(num, denum)
    plt.title("Pôles et zéros du filtre de correction d'aberrations")

    filtered_img = signal.lfilter(num, denum, img)
    return filtered_img

def rotate_90_clockwise(img):
    mat_rotation = [[0, 1], [-1, 0]]
    img_rotated = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pos = [i, j]
            new_pos = np.dot(mat_rotation, pos)
            pixel = img[i][j]
            
            img_rotated[np.int32(new_pos[0])][np.int32(new_pos[1])] = pixel
    
    return img_rotated

def low_pass_filter(img):
    fs = 1600
    niquist = fs / 2
    wp = 500 / niquist 
    ws = 750 / niquist
    gpass = 0.2
    gstop = 60
    (ord_butter, _) = signal.buttord(wp, ws, gpass=gpass, gstop=gstop)
    (ord_cheby1, _) = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
    (ord_cheby2, _) = signal.cheb2ord(wp, ws, gpass=gpass, gstop=gstop)
    (ord_ellip, _) = signal.ellipord(wp, ws, gpass=gpass, gstop=gstop)
    
    print("Order of Butterworth filter:", ord_butter)
    print("Order of Chebyshev type 1 filter:", ord_cheby1)
    print("Order of Chebyshev type 2 filter:", ord_cheby2)
    print("Order of Elliptic filter:", ord_ellip)
    
    # elliptic filter has the lowest order
    
    b, a = signal.ellip(ord_ellip, gpass, gstop, wp)
    filtered_img = signal.lfilter(b, a, img)
    plt.figure()
    zplane(b, a)
    plt.title("Pôles et zéros du filtre elliptique")
    w, h = signal.freqz(b, a)
    if True:
        plt.figure()
        plt.plot((w * niquist / np.pi), 20 * np.log10(abs(h)))
        plt.axhline(y=-3, color='green', linestyle='--')
        plt.axvline(x=500, color='red', linestyle='--')
        plt.title('Réponse en fréquence du filtre elliptique')
        plt.xlabel('Frequence [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.grid()
    
    return filtered_img

def low_pass_filter_bilineaire(img):
    fe = 1600
    fc = 500
    wa = 2*fe*np.tan((fc*2*np.pi/fe)/2)
    
    num_butt = [1, 2, 1]
    denum_butt = [((4 * fe**2 / wa**2) + (np.sqrt(2) * 2 * fe / wa) + 1), (-(8 * fe**2 / wa**2) +2), ((4 * fe**2 / wa**2)-(np.sqrt(2) * 2 * fe / wa) + 1)]
    
    num_butt = num_butt / denum_butt[0]
    denum_butt = denum_butt / denum_butt[0]
    
    filtered_img = signal.lfilter(num_butt, denum_butt, img)
    
    print("Coefficients du filtre Butterworth d'ordre 2:")
    print("Numérateur :", num_butt)
    print("Dénominateur :", denum_butt)
    
    plt.figure()
    zplane(num_butt, denum_butt)
    plt.title("Pôles et zéros du filtre Butterworth d'ordre 2")
    
    plt.figure()
    w, h = signal.freqz(num_butt, denum_butt)
    plt.plot((w * fe / (2 * np.pi)), 20 * np.log10(abs(h)))
    plt.axhline(y=-3, color='green', linestyle='--')
    plt.axvline(x=fc, color='red', linestyle='--')
    plt.title('Réponse en fréquence du filtre Butterworth d\'ordre 2')
    
    return filtered_img

def compress(img):
    cov = np.cov(img)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    inv_sorted_eigenvectors = np.linalg.inv(sorted_eigenvectors)
    img_newbase = np.dot(img, sorted_eigenvectors)
    
    img_compressed_50 = img_newbase.copy()
    img_compressed_50[:, img.shape[1]//2:] = 0
    img_decompressed_50 = np.dot(img_compressed_50, inv_sorted_eigenvectors)
    
    img_compressed_70 = img_newbase.copy()
    img_compressed_70[:, int(img.shape[1]*0.3):] = 0
    img_decompressed_70 = np.dot(img_compressed_70, inv_sorted_eigenvectors)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Image originale")
    plt.subplot(1,3,2)
    plt.title("Image compressée et décompressée (50% de compression)")
    plt.imshow(img_decompressed_50, cmap='gray')
    plt.subplot(1,3,3)
    plt.title("Image compressée et décompressée (70% de compression)")
    plt.imshow(img_decompressed_70, cmap='gray')
    
def main():   
    # return
    
    img_abberations = np.load('./images/goldhill_aberrations.npy')
    filtered_img = remove_aberations(img_abberations)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_abberations, cmap='gray')
    plt.title("Image avec aberrations")
    plt.subplot(1,2,2)
    plt.imshow(filtered_img, cmap='gray')
    plt.title("Image sans aberrations")
    
    goldhill_rotate = mpimg.imread('./images/goldhill_rotate.png')
    goldhill_rotated = rotate_90_clockwise(goldhill_rotate)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(goldhill_rotate, cmap='gray')
    plt.title("Image originale")
    plt.subplot(1,2,2)
    plt.imshow(goldhill_rotated, cmap='gray')
    plt.title("Image tournée de 90°")    
    
    goldhill_noise = np.load('./images/goldhill_bruit.npy')
    goldhill_noise_removed = low_pass_filter(goldhill_noise)
    goldhill_noise_removed_bilineaire = low_pass_filter_bilineaire(goldhill_noise)
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(goldhill_noise, cmap='gray')
    plt.title("Image avec bruit")
    plt.subplot(1,3,2)
    plt.imshow(goldhill_noise_removed, cmap='gray')
    plt.title("Image filtrée (passe-bas elliptique d'ordre 3)")
    plt.subplot(1,3,3)
    plt.imshow(goldhill_noise_removed_bilineaire, cmap='gray')
    plt.title("Image filtrée (passe-bas Butterworth d'ordre 2)")
    
    compress(mpimg.imread('./images/goldhill.png'))

    
    img_complete = np.load('./images/image_complete.npy')
    print("Image complete shape : ", img_complete.shape)
    
    img_complete_aberations = remove_aberations(img_complete)
    img_complete_rotated = rotate_90_clockwise(img_complete_aberations)
    img_complete_noise_removed = low_pass_filter(img_complete_rotated)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(img_complete, cmap='gray')
    plt.title("Image originale")
    plt.subplot(2,2,2)
    plt.imshow(img_complete_aberations, cmap='gray')
    plt.title("Image avec aberrations retirées")
    plt.subplot(2,2,3)
    plt.imshow(img_complete_rotated, cmap='gray')
    plt.title("Image avec aberrations retirées et tournée de 90°")
    plt.subplot(2,2,4)
    plt.imshow(img_complete_noise_removed, cmap='gray')
    plt.title("Image avec bruit supprimé")
    
    compress(img_complete_noise_removed)            
        
    # for fig_num in plt.get_fignums():
    #     # Get the figure object by its number
    #     fig = plt.figure(fig_num)
    #     # Define a filename for each figure, e.g., using its number
    #     filename = f"figure_{fig_num}.png"
    #     # Save the figure
    #     fig.savefig(filename, bbox_inches='tight', dpi=100) #
    #     print(f"Saved {filename}")        
    plt.show()
    print("Hello, World!")
    
    
if __name__ == "__main__":    main()