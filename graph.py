import numpy as np
from matplotlib import pyplot as plt


def main():
    train_loss = []
    train_epoch = []
    valid_loss = []
    valid_epoch = []
    with open('./log/log.txt') as file:
        for line in file.readlines():
            line = line.strip().split()
            epoch, mode, loss = line
            epoch = int(epoch)
            loss = float(loss) / 2
            if mode == 'train':
                train_epoch.append(epoch)
                train_loss.append(loss)
            elif mode == 'val':
                valid_epoch.append(epoch)
                valid_loss.append(loss)
    train_epoch = np.array(train_epoch)
    train_loss = np.array(train_loss)
    valid_epoch = np.array(valid_epoch)
    valid_loss = np.array(valid_loss)
    plt.plot(train_epoch, train_loss)
    plt.plot(valid_epoch, valid_loss)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title('NanoGPT Train & Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('graph.png')


if __name__ == '__main__':
    main()
