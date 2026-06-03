# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:04:32 2026

@author: Tomek
"""

#przesuwanie
def shift_right(img, pixels):

    return torch.roll(
        img,
        pixels,
        dims=2
    )


shifts = range(0, 12)

for shift in shifts:

    shifted_img = shift_right(
        img,
        shift
    )

    pred, conf, probs = predict(
        shifted_img
    )

    print(
        f"shift={shift}, pred={pred}, conf={conf:.3f}"
    )

    if pred != true_label:

        plt.imshow(
            shifted_img.squeeze(),
            cmap="gray"
        )

        plt.title(
            f"shift={shift}, conf={conf:.3f}"
        )

        plt.show()

        break
