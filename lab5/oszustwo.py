# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:00:54 2026

@author: Tomek
"""

def add_noise(img, sigma):

    noise = torch.randn_like(img) * sigma

    return torch.clamp(
        img + noise,
        0,
        1
    )


# losowy obrazek z MNIST
#idx = np.random.randint(len(test_dataset))

img, true_label = test_dataset[idx]


# predykcja bazowa
pred, conf, probs = predict(img)


print("True label:", true_label)
print("Initial prediction:", pred)
print("Confidence:", round(conf, 3))


# rozkład prawdopodobieństw
print("\nClass probabilities:")
print(probs.squeeze().numpy().round(3))


# obraz wejściowy
plt.figure(figsize=(4,4))

plt.imshow(
    img.squeeze(),
    cmap="gray"
)

plt.title(
    f"Original: {true_label}"
)

plt.show()


# poziomy szumu
sigmas = np.arange(
    0.0,
    0.5,
    0.02
)


first_error_sigma = None

conf_history = []


for sigma in sigmas:

    noisy_img = add_noise(
        img,
        sigma
    )

    pred, conf, probs = predict(
        noisy_img
    )

    conf_history.append(
        conf
    )


    if pred != true_label:

        first_error_sigma = sigma

        print(
            f"\nModel fooled at sigma = {sigma:.2f}"
        )

        print(
            f"Prediction: {pred}"
        )

        print(
            f"Confidence: {conf:.3f}"
        )


        plt.figure(figsize=(4,4))

        plt.imshow(
            noisy_img.squeeze(),
            cmap="gray"
        )

        plt.title(
            f"Fooled! sigma={sigma:.2f}"
        )

        plt.show()

        break


if first_error_sigma is None:

    print(
        "\nModel was NOT fooled in tested range."
    )


# wykres confidence
plt.figure(figsize=(6,4))

plt.plot(
    sigmas[:len(conf_history)],
    conf_history,
    marker='o'
)

plt.xlabel(
    "Noise sigma"
)

plt.ylabel(
    "Confidence"
)

plt.title(
    "Model confidence vs noise"
)

plt.grid()

plt.show()
