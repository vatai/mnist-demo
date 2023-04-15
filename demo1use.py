import cv2
import numpy as np
import torch
from mss import mss
from torchvision import transforms

from demo1 import Net


def main():
    device = "cpu"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    model = Net()
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    sct = mss()
    bbox = {"top": 300, "left": 300, "width": 200, "height": 200}
    while True:
        screen = np.array(sct.grab(bbox))
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_LINEAR)
        # small[small > 100] = 255
        invert = 255 - small
        cv2.imshow("window", invert)
        invert = invert.reshape(1, 28, 28)
        tensor = transform(invert).reshape(1, 1, 28, 28)
        model.eval()
        pred = model(tensor.to(device))
        print(torch.argmax(pred))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # print(f"{digit.shape=}"
    # print(f"{invert.shape=}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
