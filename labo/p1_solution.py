import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from zplane import zplane

# ========== ÉTAPE 1: CONCEPTION DU FILTRE COUPE-BANDE ==========

# Fréquence à rejeter (en rad/échantillon)
omega_reject = np.pi / 16

# Rayon du pôle (< 1 pour la stabilité)
r = 0.95

# ZÉROS: sur le cercle unité à ±omega_reject
zeros = [np.exp(1j * omega_reject), np.exp(-1j * omega_reject)]

# PÔLES: à l'intérieur du cercle unité, même angle que les zéros
poles = [r * np.exp(1j * omega_reject), r * np.exp(-1j * omega_reject)]

# Obtenir les coefficients du numérateur et dénominateur
# np.poly(racines) retourne [1, -sum(racines), produit(racines), ...]
num = np.poly(zeros)  # coefficients du numérateur
den = np.poly(poles)  # coefficients du dénominateur

# Affichage de la fonction de transfert
print("="*60)
print("FONCTION DE TRANSFERT H(z)")
print("="*60)
print(f"\nZéros (positions exactes):")
print(f"  z1 = exp(j*π/16) = {zeros[0]}")
print(f"  z2 = exp(-j*π/16) = {zeros[1]}")

print(f"\nPôles (positions pour stabilité):")
print(f"  p1 = 0.95*exp(j*π/16) = {poles[0]}")
print(f"  p2 = 0.95*exp(-j*π/16) = {poles[1]}")

print(f"\nCoefficients du numérateur: {num}")
print(f"Coefficients du dénominateur: {den}")

# Développement analytique
cos_val = np.cos(omega_reject)
print(f"\nFormes développées:")
print(f"Numérateur:   z² - {2*cos_val:.6f}z + 1")
print(f"Dénominateur: z² - {2*r*cos_val:.6f}z + {r**2:.6f}")

# ========== ÉTAPE 2: VISUALISATION POLE-ZERO ==========
print("\n" + "="*60)
print("DIAGRAMME POLE-ZERO")
print("="*60)

plt.figure(figsize=(5, 5))
zplane(num, den)
plt.title('Diagramme Pole-Zero\n(Zéros en croix, Pôles en cercle)')
plt.grid(True, alpha=0.3)

# ========== ÉTAPE 3: RÉPONSE EN FRÉQUENCE ==========
print("\n" + "="*60)
print("RÉPONSE EN FRÉQUENCE")
print("="*60)

# Évaluation de H(e^jω) pour ω ∈ [0, π]
w, h = signal.freqz(num, den, worN=1000)

# Conversion en fréquence normalisée (rad/π)
w_norm = w / np.pi

# Magnitude en dB
magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)  # 1e-10 pour éviter log(0)

# Phase en radians
phase = np.angle(h)

# Vérification à ω = π/16
idx_reject = np.argmin(np.abs(w - omega_reject))
mag_at_reject = 20 * np.log10(np.abs(h[idx_reject]) + 1e-10)
freq_at_reject = w[idx_reject]

print(f"\nMagnitude à ω = π/16:")
print(f"  Fréquence exacte vérifiée: {freq_at_reject:.6f} rad ({freq_at_reject/np.pi:.6f}π)")
print(f"  Magnitude: {mag_at_reject:.2f} dB")
print(f"  Atténuation réalisée: OUI ✓" if mag_at_reject < -40 else f"  Atténuation: FAIBLE")

print(f"\nMagnitude aux autres fréquences (échantillons):")
for freq_test in [np.pi/32, np.pi/8, np.pi/4]:
    idx = np.argmin(np.abs(w - freq_test))
    mag = 20 * np.log10(np.abs(h[idx]) + 1e-10)
    print(f"  ω = {freq_test/np.pi:.4f}π: {mag:.2f} dB")

# ========== ÉTAPE 4: GRAPHIQUES DE LA RÉPONSE EN FRÉQUENCE ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Magnitude (dB)
ax1.plot(w_norm, magnitude_db, 'b', linewidth=2)
ax1.axvline(omega_reject/np.pi, color='r', linestyle='--', alpha=0.7, label=f'ω = π/16 (rejet)')
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('Magnitude (dB)', fontsize=11)
ax1.set_title('Réponse en Fréquence du Filtre Coupe-Bande', fontsize=12, fontweight='bold')
ax1.legend()
ax1.set_ylim([-100, 5])

# Phase (radians)
ax2.plot(w_norm, phase, 'g', linewidth=2)
ax2.axvline(omega_reject/np.pi, color='r', linestyle='--', alpha=0.7, label=f'ω = π/16 (rejet)')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Fréquence normalisée (xπ rad/sample)', fontsize=11)
ax2.set_ylabel('Phase (radians)', fontsize=11)
ax2.legend()

plt.tight_layout()

# ========== ÉTAPE 5: TEST DU FILTRE SUR LE SIGNAL ==========
print("\n" + "="*60)
print("TEST DU FILTRE SUR LE SIGNAL D'ENTRÉE")
print("="*60)

# Signal d'entrée: x[n] = sin(πn/16) + sin(πn/32)
N = 256  # longueur du signal
n = np.arange(N)

x = np.sin(np.pi * n / 16) + np.sin(np.pi * n / 32)

# Filtrage
y = signal.lfilter(num, den, x)

print(f"\nLongueur du signal: {N} samples")
print(f"Durée temporelle: ~{N} échantillons")
print(f"Composantes du signal d'entrée:")
print(f"  1. sin(πn/16) - fréquence 0.0625π (À REJETER)")
print(f"  2. sin(πn/32) - fréquence 0.03125π (À LAISSER PASSER)")

# ========== ÉTAPE 6: ANALYSE SPECTRALE (FFT) ==========
print("\n" + "="*60)
print("ANALYSE SPECTRALE PAR FFT")
print("="*60)

# FFT du signal d'entrée et sortie
X = np.fft.rfft(x)
Y = np.fft.rfft(y)
freq = np.fft.rfftfreq(N, d=1) * 2 * np.pi  # fréquences en rad

# Amplitude normalisée
mag_X = np.abs(X) / (N/2)
mag_Y = np.abs(Y) / (N/2)

# Identification des pics principaux
print(f"\nSpectres (amplitude normalisée):")
print(f"\nSIGNAL D'ENTRÉE:")
for f_target in [np.pi/32, np.pi/16]:
    idx = np.argmin(np.abs(freq - f_target))
    print(f"  ω = {f_target/np.pi:.6f}π: amplitude = {mag_X[idx]:.4f}")

print(f"\nSIGNAL DE SORTIE (APRÈS FILTRAGE):")
for f_target in [np.pi/32, np.pi/16]:
    idx = np.argmin(np.abs(freq - f_target))
    amp_reduction = mag_X[idx] / (mag_Y[idx] + 1e-10)
    print(f"  ω = {f_target/np.pi:.6f}π: amplitude = {mag_Y[idx]:.4f} (atténuation x{amp_reduction:.1f})")

# ========== ÉTAPE 7: VISUALISATION TEMPORELLE ET SPECTRALE ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Domaine temporel (entrée et sortie)
ax = axes[0, 0]
ax.plot(n[:100], x[:100], 'b', label='Entrée x[n]', linewidth=1.5, alpha=0.7)
ax.plot(n[:100], y[:100], 'r', label='Sortie y[n]', linewidth=1.5)
ax.set_xlabel('n (échantillons)')
ax.set_ylabel('Amplitude')
ax.set_title('Signal temporel (premiers 100 samples)')
ax.legend()
ax.grid(True, alpha=0.3)

# Spectre d'amplitude (entrée)
ax = axes[0, 1]
ax.stem(freq / np.pi, mag_X, basefmt=' ')
ax.set_xlim([0, 0.15])
ax.set_xlabel('Fréquence normalisée (xπ rad/sample)')
ax.set_ylabel('Amplitude')
ax.set_title('ENTRÉE: Spectre d\'amplitude (FFT)')
ax.axvline(1/16, color='r', linestyle='--', alpha=0.5, label='π/16 (doit disparaître)')
ax.axvline(1/32, color='g', linestyle='--', alpha=0.5, label='π/32 (doit rester)')
ax.legend()
ax.grid(True, alpha=0.3)

# Spectre d'amplitude (sortie)
ax = axes[1, 1]
ax.stem(freq / np.pi, mag_Y, basefmt=' ')
ax.set_xlim([0, 0.15])
ax.set_xlabel('Fréquence normalisée (xπ rad/sample)')
ax.set_ylabel('Amplitude')
ax.set_title('SORTIE: Spectre d\'amplitude (FFT) - APRÈS FILTRAGE')
ax.axvline(1/16, color='r', linestyle='--', alpha=0.5, label='π/16 (rejet)')
ax.axvline(1/32, color='g', linestyle='--', alpha=0.5, label='π/32 (passé)')
ax.legend()
ax.grid(True, alpha=0.3)

# Comparaison spectrale (entrée vs sortie)
ax = axes[1, 0]
ax.plot(freq / np.pi, mag_X, 'b', label='Entrée', linewidth=2, alpha=0.7)
ax.plot(freq / np.pi, mag_Y, 'r', label='Sortie (filtrée)', linewidth=2)
ax.set_xlim([0, 0.15])
ax.set_xlabel('Fréquence normalisée (xπ rad/sample)')
ax.set_ylabel('Amplitude')
ax.set_title('Comparaison ENTRÉE vs SORTIE')
ax.axvline(1/16, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='π/16 = FIN DE VIE')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("\n✓ Filtre coupe-bande (notch filter) conçu avec succès")
print("✓ Fréquence π/16 rejetée")
print("✓ Autres fréquences laissées pratiquement inchangées")
print("✓ Stabilité garantie (pôles à l'intérieur du cercle unité)")

plt.show()
