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
    # zplane(num, denum)

    filtered_img = signal.lfilter(num, denum, img)
    return filtered_img

def rotate_90_clockwise(img):
    mat_rotation = [[0, 1], [-1, 0]]
    img_rotated = np.zeros(img.shape) # on crée une image vide pour stocker le résultat de la transformation linéaire
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pos = [i, j]
            new_pos = np.dot(mat_rotation, pos)
            pixel = img[i][j]
            

            img_rotated[np.int32(new_pos[0])][np.int32(new_pos[1])] = pixel
    
    return img_rotated



def main():
    if False:
        img_abberations = np.load('./images/goldhill_aberrations.npy')
        filtered_img = remove_aberations(img_abberations)
        
        plt.subplot(1,2,1)
        plt.imshow(img_abberations, cmap='gray')
        plt.title("Image avec aberrations")
        plt.subplot(1,2,2)
        plt.imshow(filtered_img, cmap='gray')
        plt.title("Image filtrée")
        
        goldhill_rotate = mpimg.imread('./images/goldhill_rotate.png')
        goldhill_rotated = rotate_90_clockwise(goldhill_rotate)
            
        plt.subplot(1,2,1)
        plt.imshow(goldhill_rotate, cmap='gray')
        plt.title("Image originale")
        plt.subplot(1,2,2)
        plt.imshow(goldhill_rotated, cmap='gray')
        plt.title("Image tournée de 90°")    
    
    img_complete = np.load('./images/image_complete.npy')
    img_complete_aberations = remove_aberations(img_complete)
    img_complete_rotated = rotate_90_clockwise(img_complete_aberations)
    
    plt.subplot(1,3,1)
    plt.imshow(img_complete, cmap='gray')
    plt.title("Image originale")
    plt.subplot(1,3,2)
    plt.imshow(img_complete_aberations, cmap='gray')
    plt.title("Image avec aberrations retirées")
    plt.subplot(1,3,3)
    plt.imshow(img_complete_rotated, cmap='gray')
    plt.title("Image avec aberrations retirées et tournée de 90°")
        
        
    plt.show()
    print("Hello, World!")
    
    
if __name__ == "__main__":    main()