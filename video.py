import cv2
import numpy as np
import torchvision
from mss import mss

invert = None


def main():
    global invert

    mnist = torchvision.datasets.mnist.MNIST(".")
    digit = mnist.data[2].numpy()
    label = mnist.classes[2]
    print(label)
    cv2.imshow("digit", digit)

    vid = cv2.VideoCapture(0, cv2.CAP_V4L2)
    while True:
        # for i in range(0):
        ret, frame = vid.read()
        h, w, c = frame.shape
        crop = frame[:, (w - h) // 2 : (w + h) // 2, :]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_LINEAR)
        small[small > 100] = 255
        invert = 255 - small
        cv2.imshow("frame", invert)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vid.release()

    bbox = {"top": 300, "left": 2100, "width": 200, "height": 200}
    sct = mss()
    # for i in range(200):
    # while True:
    #     screen = np.array(sct.grab(bbox))
    #     gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    #     small = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_LINEAR)
    #     small[small > 100] = 255
    #     invert = 255 - small
    #     cv2.imshow("window", invert)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    # print(f"{digit.shape=}")
    # print(f"{invert.shape=}")
    cv2.destroyAllWindows()


main()
