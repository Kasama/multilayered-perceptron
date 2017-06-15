import cv2 as opencv
import numpy as np


def main():
    train = np.loadtxt('train.csv', delimiter=',', skiprows=1)
    print('ready!')
    n = int(input())
    while n <= 256:
        image = train[n, 1:len(train[n])]
        number = train[n][0]
        print('number is: ', number)
        image = image.reshape(28, 28)
        opencv.imshow('ibagem', image)
        opencv.waitKey(0)
        n = int(input())


if __name__ == '__main__':
    main()
