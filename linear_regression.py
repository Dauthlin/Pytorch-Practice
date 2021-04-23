import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cuda = torch.device('cuda')
    torch.cuda.init()
    x_train = np.array([[4.7], [2.4], [7.5], [7.1], [4.3],
                        [7.8], [8.9], [5.2], [4.59], [2.1],
                        [8], [5], [7.5], [5], [4],
                        [8], [5.2], [4.9], [3], [4.7],
                        [4], [4.8], [3.5], [2.1], [4.1]])
    y_train = np.array([[2.6], [1.6], [3.09], [2.4], [2.4],
                        [3.3], [2.6], [1.96], [3.13], [1.76],
                        [3.2], [2.1], [1.6], [2.5], [2.2],
                        [2.75], [2.4], [1.8], [1], [2],
                        [1.6], [2.4], [2.6], [1.5], [3.1]])

    plt.scatter(x_train, y_train)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    input_size = 1
    hidden_size = 1
    output_size = 1
    learning_rate = 0.001
    w1 = torch.rand(input_size, hidden_size, requires_grad=True)
    b1 = torch.rand(input_size, hidden_size, requires_grad=True)
    # print(w1, b1)

    for i in range(1, 4000):
        y_pred = x_train.mm(w1.double()).clamp(min=0) + b1
        # creating the guess, clamping at 0 is the relu activation function
        loss = ((y_pred - y_train) ** 2).sum() + 0.01 * w1.norm()
        # loss squared
        loss.backward()
        # get gradient of weights

        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            b1 -= learning_rate * b1.grad
            # go in direction of gradients
            w1.grad.zero_()
            b1.grad.zero_()
            # reset gradients
    predicted_y = x_train.mm(w1.double()).clamp(min=0) + b1

    plt.plot(x_train.detach(), predicted_y.detach(), c='g')
    plt.show()
