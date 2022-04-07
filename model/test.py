import numpy as np
from DataLoader import load_images, load_labels
import pickle





def relu(x): return np.maximum(0, x)


def reluDerivative(x):
    return np.array([reluDerivativeSingleElement(xi) for xi in x])


def reluDerivativeSingleElement(xi):
    if xi > 0:
        return 1
    elif xi <= 0:
        return 0


def matmul(matrix1: np.ndarray, matrix2: np.ndarray):
    return np.matmul(matrix1, matrix2)


def mse(mat1, mat2):
    difference = np.subtract(mat1, mat2)
    return matmul(np.transpose(difference), difference)


def l2_regularization(lamb, coefficients: list):
    sum0 = 0
    for i in coefficients:
        sum0 += np.linalg.norm(i, ord='f')
    return lamb * sum0



def test():
    test_images = load_images('dataset/t10k-images-idx3-ubyte.gz')
    test_labels = load_labels('dataset/t10k-labels-idx1-ubyte.gz')
    num_test = len(test_images)
    f = open('parameters/隐藏层100, 正则化0.010000, 学习率0.000100.txt', 'rb')
    w1,b1,w2,b2 = pickle.load(f)
    num_right = 0
    sum_test_loss = 0
    for i in range(num_test):
        a0 = test_images[i]
        y = test_labels[i]

        z1 = matmul(w1, a0) + b1
        a1 = relu(z1)
        z2 = matmul(w2, a1) + b2
        a2 = relu(z2)

        if np.argmax(a2) == np.argmax(y):
            num_right += 1
        loss = mse(a2, y)
        sum_test_loss += loss

    accuracy_new = num_right/num_test
    print('accuracy:', accuracy_new)

if __name__  == '__main__':
    test()