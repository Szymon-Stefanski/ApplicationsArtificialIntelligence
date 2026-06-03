# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:20:00 2026

@author: Tomek
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt


class MNISTCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)

        x = self.classifier(x)

        return x


model = MNISTCNN()

model.load_state_dict(
    torch.load(
        "mnist_cnn_weights.pt",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

print("Model loaded.")

transform = torchvision.transforms.ToTensor()

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


def predict(img):
    with torch.no_grad():
        output = model(img.unsqueeze(0))

        probs = torch.softmax(output, dim=1)

        pred = torch.argmax(probs).item()

        conf = torch.max(probs).item()

    return pred, conf, probs


idx = np.random.randint(len(test_dataset))

# idx = 16
img, label = test_dataset[idx]

pred, conf, probs = predict(img)

print("True label:", label)
print("Prediction:", pred)
print("Confidence:", conf)

print("All probabilities:")
print(probs.squeeze().numpy().round(4).tolist())

plt.imshow(img.squeeze(), cmap="gray")
plt.title(
    f"True={label}, Pred={pred}, Conf={conf:.3f}"
)
plt.show()

# =============================================================================
# TUTAJ ZACZYNA SIĘ TWÓJ DODATKOWY KOD DO EXPERYMENTÓW (ZADANIA 3-6)
# =============================================================================

# --- ZADANIE 3: EKSPERYMENT Z SZUMEM ---
print("\n--- URUCHAMIAM ZADANIE 3: SZUM ---")


def add_noise(img, sigma):
    noise = torch.randn_like(img) * sigma
    return torch.clamp(img + noise, 0, 1)


# Poziomy szumu do przetestowania
sigmas = np.arange(0.0, 0.5, 0.02)

for sigma in sigmas:
    noisy_img = add_noise(img, sigma)
    pred_noise, conf_noise, _ = predict(noisy_img)

    # Szukamy momentu, w którym model się pomyli
    if pred_noise != label:
        print(f"[ZAD3] Model oszukany przy sigma = {sigma:.2f}")
        print(f"[ZAD3] Prawdziwa: {label}, Predykcja z szumem: {pred_noise}, Pewność: {conf_noise:.3f}")

        # Wyświetlamy zaszumiony obrazek
        plt.figure()
        plt.imshow(noisy_img.squeeze(), cmap="gray")
        plt.title(f"Szum sigma={sigma:.2f} -> Pred={pred_noise}")
        plt.show()
        break

# --- ZADANIE 4: EKSPERYMENT Z PRZESUNIĘCIEM ---
print("\n--- URUCHAMIAM ZADANIE 4: PRZESUWANIE ---")


def shift_right(img, pixels):
    return torch.roll(img, pixels, dims=2)


shifts = range(0, 12)

for shift in shifts:
    shifted_img = shift_right(img, shift)
    pred_shift, conf_shift, _ = predict(shifted_img)

    # Szukamy momentu, w którym przesunięcie popsuje wynik
    if pred_shift != label:
        print(f"[ZAD4] Model oszukany przy przesunięciu o {shift} pikseli!")
        print(f"[ZAD4] Prawdziwa: {label}, Predykcja po przesunięciu: {pred_shift}, Pewność: {conf_shift:.3f}")

        # Wyświetlamy przesunięty obrazek
        plt.figure()
        plt.imshow(shifted_img.squeeze(), cmap="gray")
        plt.title(f"Przesunięcie o {shift} px -> Pred={pred_shift}")
        plt.show()
        break

# --- ZADANIE 5: EKSPERYMENT Z ZASŁANIANIEM ---
print("\n--- URUCHAMIAM ZADANIE 5: ZASŁANIANIE ---")


def erase_patch(img, x, y, size=8):
    img2 = img.clone()
    img2[:, y:y + size, x:x + size] = 0
    return img2


zasloniony = False
for x in range(0, 20, 4):
    for y in range(0, 20, 4):
        test_img = erase_patch(img, x, y)
        pred_patch, conf_patch, _ = predict(test_img)

        if pred_patch != label:
            print(
                f"[ZAD5] Zasłonięcie kwadratu na pozycji X={x}, Y={y} zmienia wynik na: {pred_patch} (Pewność: {conf_patch:.3f})")

            plt.figure()
            plt.imshow(test_img.squeeze(), cmap="gray")
            plt.title(f"Zasłonięte X={x}, Y={y} -> Pred={pred_patch}")
            plt.show()
            zasloniony = True
            break
    if zasloniony:
        break

# --- ZADANIE 6: SZUKANIE DUŻEGO BŁĘDU (>90%) ---
print("\n--- URUCHAMIAM ZADANIE 6: SZUKANIE PEWNEGO BŁĘDU ---")

znalazlem_blad = False
for i in range(len(test_dataset)):
    test_img, test_label = test_dataset[i]
    pred_err, conf_err, _ = predict(test_img)

    # Warunek z zadania: model się myli ORAZ ma pewność wyższą niż 90%
    if pred_err != test_label and conf_err > 0.90:
        print(f"[ZAD6] SUKCES! Znalazłem spektakularny błąd sieci:")
        print(f"[ZAD6] Indeks obrazka (idx): {i}")
        print(f"[ZAD6] Prawdziwa cyfra: {test_label}")
        print(f"[ZAD6] Co przewidział model: {pred_err}")
        print(f"[ZAD6] Pewność modelu: {conf_err * 100:.2f}%")

        # Wyświetlamy ten konkretny trefny obrazek
        plt.figure()
        plt.imshow(test_img.squeeze(), cmap="gray")
        plt.title(f"Idx: {i} | True: {test_label} | Pred: {pred_err} | Conf: {conf_err:.2f}")
        plt.show()

        znalazlem_blad = True
        break

if not znalazlem_blad:
    print("[ZAD6] O kurde, nie znalazłem takiego błędu w całym zbiorze testowym.")
