import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

T = [[0.5, 0],
    [0, 2]]
# a
# T = [[2, 0],
#     [0, 0.5]]

# b
plt.gray() # défini la colormap apr défaut en niveau de gris
img_couleur = mpimg.imread('./images/goldhill.png') # charge l'image

# img_gris = np.mean(img_couleur, -1)
img_gris = img_couleur

print("img_couleur.shape =", img_couleur.shape)

img_scaled = np.zeros((img_gris.shape[0] // 2, img_gris.shape[1] * 2)) # on crée une image vide pour stocker le résultat de la transformation linéaire
img_scaled_2 = img_scaled.copy()
print(img_gris[0][0])
for i in range(img_gris.shape[0]):
    for j in range(img_gris.shape[1]):
        pos = [i, j]
        new_pos = np.dot(T, pos)
        pixel = img_gris[i][j]
        
        img_scaled[i // 2][j * 2] = pixel
        img_scaled[i // 2][j * 2 + 1] = pixel

        img_scaled_2[np.int32(new_pos[0])][np.int32(new_pos[1])] = pixel
        img_scaled_2[np.int32(new_pos[0])][np.int32(new_pos[1])+1] = pixel
        
plt.subplot(1, 2, 1)
plt.title("Image originale")
plt.imshow(img_gris, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Image transformée 1")
plt.imshow(img_scaled, cmap='gray')


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Image originale")
plt.imshow(img_gris, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Image transformée 2")
plt.imshow(img_scaled_2, cmap='gray')

plt.show()

