# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:14:13 2026

@author: Tomek
"""

#zaslanianie fragmentu
def erase_patch(img, x, y, size=8):

    img2 = img.clone()

    img2[:, y:y+size, x:x+size] = 0

    return img2



for x in range(0, 20, 4):

    for y in range(0, 20, 4):

        test_img = erase_patch(
            img,
            x,
            y
        )

        pred, conf, probs = predict(
            test_img
        )

        if pred != true_label:

            print(
                x,
                y,
                pred,
                conf
            )

            plt.imshow(
                test_img.squeeze(),
                cmap="gray"
            )

            plt.show()

            break


