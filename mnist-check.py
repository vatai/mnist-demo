#!/usr/bin/env python

import matplotlib.pyplot as plt
import torch
import torchvision


def main():
    mnist = torchvision.datasets.mnist.MNIST(
        ".", train=False, transform=torchvision.transforms.ToTensor()
    )
    N, M = 4, 4
    fig, axs = plt.subplots(N, M)
    dl = torch.utils.data.DataLoader(mnist)
    dl_list = list(dl)
    print(dl_list[0])
    for i in range(N):
        for j in range(M):
            idx = i * M + j
            data = mnist.data[idx]
            target = mnist.targets[idx].item()
            # data = mnist[idx]
            axs[i][j].imshow(data)
            axs[i][j].set_title(f"tgt: {target}")
    plt.show()

    # print(mnist.data[0])
    print("done")


main()
