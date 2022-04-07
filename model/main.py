from time import sleep
import numpy as np
from sklearn.utils import shuffle
from sympy import Order
from DataLoader import load_images, load_labels
import matplotlib
import matplotlib.pyplot as plt
import pickle

from PIL import Image

def numpy2pil(np_array: np.ndarray) -> Image:
    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img


def numpy2pil_bw(np_array: np.ndarray) -> Image:
    # assert_msg = 'Input shall be a HxWx1 ndarray'
    # assert isinstance(np_array, np.ndarray), assert_msg
    # assert len(np_array.shape) == 3, assert_msg
    # assert np_array.shape[2] == 1, assert_msg
    img = Image.fromarray(np_array)
    return img

def zeroMean(dataMat):      
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData=dataMat-meanVal
    return newData,meanVal
 
def pca(dataMat, n):  
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #减去均值
    stded = meanRemoved / np.std(dataMat) #用标准差归一化
    covMat = np.cov(stded, rowvar=0) #求协方差方阵
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) #求特征值和特征向量
    k = n   #get the topNfeat
    eigValInd = np.argsort(eigVals)  #对特征值进行排序
    eigValInd = eigValInd[:-(k + 1):-1]  #get topNfeat
    redEigVects = eigVects[:, eigValInd]       # 除去不需要的特征向量
    lowDDataMat = stded * redEigVects    #求新的数据矩阵
    #reconMat = (lowDDataMat * redEigVects.T) * std(dataMat) + meanVals  #对矩阵还原
    return lowDDataMat



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


def train():
    # 数据读取
    print('开始读取数据')
    train_images = load_images('dataset/train-images-idx3-ubyte.gz')
    train_labels = load_labels('dataset/train-labels-idx1-ubyte.gz')
    test_images = load_images('dataset/t10k-images-idx3-ubyte.gz')
    test_labels = load_labels('dataset/t10k-labels-idx1-ubyte.gz')
    print('结束读取数据')

    num_train = len(train_images)
    num_test = len(test_images)
    dimen_input = len(train_images[0])
    print(dimen_input)

    # 超参数设置

    num_cells_list = [5, 10, 30, 100]
    lamb_list = [1e-2, 1e-3, 1e-4, 1e-5]
    learning_rate_list = [1e-3, 1e-4, 1e-5, 1e-6]
    accuracy = 0
    max_accuaracy = 0

    for num_cells in num_cells_list:
        for lamb in lamb_list:
            for learning_rate in learning_rate_list:
                lr = learning_rate
                print('隐藏层大小', num_cells, '正则化系数', lamb, '学习率', learning_rate)
                # 参数初始化
                w1 = np.random.randn(num_cells, dimen_input)/1000
                b1 = np.random.randn(num_cells)/1000
                w2 = np.random.randn(10, num_cells)/1000
                b2 = np.random.randn(10)/1000
                loss_list = []
                test_loss_list = []
                accuracy_list = []
                patience = 0

                while 1:
                    train_index = shuffle([i for i in range(num_train)])
                    train_images = train_images[train_index]
                    train_labels = train_labels[train_index]
                    sum_train_loss = 0
                    for i in range(num_train):
                        a0 = train_images[i]
                        y = train_labels[i]

                        z1 = matmul(w1, a0) + b1
                        a1 = relu(z1)
                        z2 = matmul(w2, a1) + b2
                        a2 = (z2)

                        loss = mse(a2, y) + l2_regularization(lamb, [w1, w2])
                        sum_train_loss += loss
                        sigma2 = 2*np.subtract(a2, y)
                        sigma1 = np.multiply(reluDerivative(z1), matmul(np.transpose(w2), sigma2))

                        w2 = np.subtract(w2, learning_rate *
                                         (np.outer(sigma2, a1) + lamb * w2))
                        w1 = np.subtract(w1, learning_rate *
                                         (np.outer(sigma1, a0) + lamb * w1))
                        b2 = np.subtract(b2, learning_rate * sigma2)
                        b1 = np.subtract(b1, learning_rate * sigma1)
                        f = open('./parameters/隐藏层%d, 正则化%f, 学习率%f.txt'%(num_cells, lamb, lr), 'wb')
                        pickle.dump([w1,b1,w2, b2], f)
                    loss_list.append(sum_train_loss/num_train)

                    # 验证测试集上的accuracy
                    f = open('./parameters/隐藏层%d, 正则化%f, 学习率%f.txt'%(num_cells, lamb, lr), 'rb')
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
                    test_loss_list.append(sum_test_loss/num_test)

                    accuracy_new = num_right/num_test
                    max_accuaracy = max(max_accuaracy, accuracy_new)
                    print('accuracy:', accuracy_new)
                    accuracy_list.append(accuracy_new)
                    learning_rate = 0.8 * learning_rate
                    if accuracy_new <= accuracy:
                        patience += 1
                        if patience >= 5:
                            loss = loss_list
                            test_loss = test_loss_list
                            plt.plot(loss, label='loss')
                            plt.plot(test_loss, label='test_loss')
                            plt.plot(accuracy_list, label='accuracy')
                            plt.title('model loss and accuarcy')
                            plt.ylabel('loss')
                            plt.xlabel('epoch')
                            plt.legend(['train', 'test','accuracy'], loc='upper left')
                            plt.savefig('./images/隐藏层%d, 正则化%f, 学习率%f.png'%(num_cells, lamb, lr))
                            plt.close()
                            lowDmatrix_w1 = pca(w1, 3)
                            lowDmatrix_w1 = np.resize(lowDmatrix_w1, [len(w1),1,3])
                            img_w1 = numpy2pil(lowDmatrix_w1)
                            img_w1.save('./parameters_visual/w1 隐藏层%d, 正则化%f, 学习率%f.png'%(num_cells, lamb, lr))
                            lowDmatrix_w2 = pca(w2, 3)
                            lowDmatrix_w2 = np.resize(lowDmatrix_w2, [len(w2),1,3])
                            img_w2 = numpy2pil(lowDmatrix_w2)
                            img_w2.save('./parameters_visual/w2 隐藏层%d, 正则化%f, 学习率%f.png'%(num_cells, lamb, lr))
                            img_b1 = numpy2pil_bw(np.reshape(b1, [len(b1), 1]))
                            img_b1.convert('L').save('./parameters_visual/b1 隐藏层%d, 正则化%f, 学习率%f.png'%(num_cells, lamb, lr))
                            img_b2 = numpy2pil_bw(np.reshape(b2, [len(b2), 1]))
                            img_b2.convert('L').save('./parameters_visual/b2 隐藏层%d, 正则化%f, 学习率%f.png'%(num_cells, lamb, lr))




                            break
                    
                    accuracy = accuracy_new

                    #隐藏层大小 100 正则化系数 0.01 学习率 0.0001
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


if __name__ == '__main__':

    # train()
    test()

'''(base) russell_a@daixiaohaodeMacBook-Air 两层神经网络分类器 % conda run -n base --no-capture-output --live-stream
 python /Users/russell_a/Documents/学习材料/计算机视觉/两层神经网络分类器/model/main.py
开始读取数据
结束读取数据
784
隐藏层大小 5 正则化系数 0.01 学习率 0.001
accuracy: 0.0974
accuracy: 0.1032
accuracy: 0.1135
accuracy: 0.0958
accuracy: 0.101
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 5 正则化系数 0.01 学习率 0.0001
accuracy: 0.5717
accuracy: 0.5901
accuracy: 0.581
accuracy: 0.5822
accuracy: 0.5819
accuracy: 0.5815
accuracy: 0.5849
accuracy: 0.5892
accuracy: 0.5855
accuracy: 0.5838
隐藏层大小 5 正则化系数 0.01 学习率 1e-05
accuracy: 0.5845
accuracy: 0.585
accuracy: 0.5869
accuracy: 0.5857
accuracy: 0.5863
accuracy: 0.5872
accuracy: 0.5869
accuracy: 0.5871
accuracy: 0.5873
accuracy: 0.5864
accuracy: 0.5869
accuracy: 0.5864
隐藏层大小 5 正则化系数 0.01 学习率 1e-06
accuracy: 0.5718
accuracy: 0.5576
accuracy: 0.5315
accuracy: 0.5156
accuracy: 0.503
隐藏层大小 5 正则化系数 0.001 学习率 0.001
accuracy: 0.1135
accuracy: 0.101
accuracy: 0.1009
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 5 正则化系数 0.001 学习率 0.0001
accuracy: 0.4837
accuracy: 0.553
accuracy: 0.5565
accuracy: 0.5554
accuracy: 0.5629
accuracy: 0.5653
accuracy: 0.5654
accuracy: 0.562
accuracy: 0.5641
accuracy: 0.5635
accuracy: 0.5671
accuracy: 0.5631
accuracy: 0.5644
accuracy: 0.5645
accuracy: 0.5646
accuracy: 0.5639
隐藏层大小 5 正则化系数 0.001 学习率 1e-05
accuracy: 0.5199
accuracy: 0.5804
accuracy: 0.5807
accuracy: 0.5821
accuracy: 0.5812
accuracy: 0.5806
accuracy: 0.5815
accuracy: 0.58
accuracy: 0.5821
accuracy: 0.5806
隐藏层大小 5 正则化系数 0.001 学习率 1e-06
accuracy: 0.4435
accuracy: 0.5379
accuracy: 0.5441
accuracy: 0.5426
accuracy: 0.5393
accuracy: 0.5508
accuracy: 0.5263
accuracy: 0.5386
accuracy: 0.5161
隐藏层大小 5 正则化系数 0.0001 学习率 0.001
accuracy: 0.0974
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 5 正则化系数 0.0001 学习率 0.0001
accuracy: 0.5778
accuracy: 0.5746
accuracy: 0.5823
accuracy: 0.5902
accuracy: 0.5855
accuracy: 0.5835
accuracy: 0.5844
accuracy: 0.5859
accuracy: 0.584
accuracy: 0.5855
accuracy: 0.5856
accuracy: 0.5848
隐藏层大小 5 正则化系数 0.0001 学习率 1e-05
accuracy: 0.5405
accuracy: 0.5795
accuracy: 0.5815
accuracy: 0.5837
accuracy: 0.5831
accuracy: 0.5839
accuracy: 0.5831
accuracy: 0.5836
accuracy: 0.5829
accuracy: 0.5839
accuracy: 0.5823
隐藏层大小 5 正则化系数 0.0001 学习率 1e-06
accuracy: 0.6073
accuracy: 0.5398
accuracy: 0.5063
accuracy: 0.4937
accuracy: 0.4919
accuracy: 0.492
accuracy: 0.4927
accuracy: 0.4928
accuracy: 0.4928
隐藏层大小 5 正则化系数 1e-05 学习率 0.001
accuracy: 0.0974
accuracy: 0.0974
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.101
隐藏层大小 5 正则化系数 1e-05 学习率 0.0001
accuracy: 0.4895
accuracy: 0.4925
accuracy: 0.4864
accuracy: 0.5826
accuracy: 0.5831
accuracy: 0.5934
accuracy: 0.5815
accuracy: 0.583
accuracy: 0.5831
accuracy: 0.5823
accuracy: 0.5813
accuracy: 0.5824
accuracy: 0.5822
隐藏层大小 5 正则化系数 1e-05 学习率 1e-05
accuracy: 0.5827
accuracy: 0.585
accuracy: 0.5868
accuracy: 0.588
accuracy: 0.5867
accuracy: 0.5873
accuracy: 0.5864
accuracy: 0.5874
accuracy: 0.5877
accuracy: 0.588
accuracy: 0.5873
accuracy: 0.5874
accuracy: 0.5883
accuracy: 0.5881
accuracy: 0.5877
隐藏层大小 5 正则化系数 1e-05 学习率 1e-06
accuracy: 0.6001
accuracy: 0.5995
accuracy: 0.6098
accuracy: 0.5871
accuracy: 0.5967
accuracy: 0.579
accuracy: 0.5709
accuracy: 0.5691
隐藏层大小 10 正则化系数 0.01 学习率 0.001
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 10 正则化系数 0.01 学习率 0.0001
accuracy: 0.7196
accuracy: 0.818
accuracy: 0.8159
accuracy: 0.8201
accuracy: 0.8185
accuracy: 0.8261
accuracy: 0.8316
accuracy: 0.8314
accuracy: 0.8319
accuracy: 0.8321
accuracy: 0.8316
accuracy: 0.8346
accuracy: 0.8321
隐藏层大小 10 正则化系数 0.01 学习率 1e-05
accuracy: 0.9143
accuracy: 0.9106
accuracy: 0.9155
accuracy: 0.9197
accuracy: 0.9195
accuracy: 0.9204
accuracy: 0.9203
accuracy: 0.9224
accuracy: 0.9218
accuracy: 0.923
accuracy: 0.9206
隐藏层大小 10 正则化系数 0.01 学习率 1e-06
accuracy: 0.835
accuracy: 0.8612
accuracy: 0.8852
accuracy: 0.8931
accuracy: 0.8971
accuracy: 0.8993
accuracy: 0.8997
accuracy: 0.9034
accuracy: 0.9034
accuracy: 0.9035
accuracy: 0.9051
accuracy: 0.9048
accuracy: 0.9057
accuracy: 0.9067
accuracy: 0.9069
accuracy: 0.907
accuracy: 0.907
accuracy: 0.9075
accuracy: 0.9073
隐藏层大小 10 正则化系数 0.001 学习率 0.001
accuracy: 0.098
accuracy: 0.1135
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 10 正则化系数 0.001 学习率 0.0001
accuracy: 0.8613
accuracy: 0.8864
accuracy: 0.891
accuracy: 0.8954
accuracy: 0.9067
accuracy: 0.9058
accuracy: 0.8993
accuracy: 0.9088
accuracy: 0.9063
accuracy: 0.9107
accuracy: 0.9075
accuracy: 0.9096
accuracy: 0.9087
隐藏层大小 10 正则化系数 0.001 学习率 1e-05
accuracy: 0.9066
accuracy: 0.9135
accuracy: 0.9172
accuracy: 0.9189
accuracy: 0.9186
accuracy: 0.9201
accuracy: 0.9194
accuracy: 0.9218
accuracy: 0.9204
accuracy: 0.9207
accuracy: 0.9212
accuracy: 0.9215
accuracy: 0.9199
隐藏层大小 10 正则化系数 0.001 学习率 1e-06
accuracy: 0.7787
accuracy: 0.817
accuracy: 0.8308
accuracy: 0.8373
accuracy: 0.8388
accuracy: 0.8407
accuracy: 0.8414
accuracy: 0.8426
accuracy: 0.844
accuracy: 0.8448
accuracy: 0.8451
accuracy: 0.8469
accuracy: 0.8468
accuracy: 0.8482
accuracy: 0.8465
accuracy: 0.8476
accuracy: 0.8472
accuracy: 0.8472
隐藏层大小 10 正则化系数 0.0001 学习率 0.001
accuracy: 0.0958
accuracy: 0.1009
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 10 正则化系数 0.0001 学习率 0.0001
accuracy: 0.8948
accuracy: 0.9016
accuracy: 0.9112
accuracy: 0.9071
accuracy: 0.9152
accuracy: 0.9161
accuracy: 0.9211
accuracy: 0.9145
accuracy: 0.9197
accuracy: 0.919
accuracy: 0.9215
accuracy: 0.9184
accuracy: 0.9204
accuracy: 0.9206
accuracy: 0.9209
accuracy: 0.9204
隐藏层大小 10 正则化系数 0.0001 学习率 1e-05
accuracy: 0.8305
accuracy: 0.836
accuracy: 0.9174
accuracy: 0.9179
accuracy: 0.9212
accuracy: 0.9221
accuracy: 0.9225
accuracy: 0.922
accuracy: 0.9231
accuracy: 0.922
accuracy: 0.9208
accuracy: 0.9235
accuracy: 0.9226
隐藏层大小 10 正则化系数 0.0001 学习率 1e-06
accuracy: 0.8016
accuracy: 0.8652
accuracy: 0.8784
accuracy: 0.8828
accuracy: 0.887
accuracy: 0.8892
accuracy: 0.8907
accuracy: 0.893
accuracy: 0.8954
accuracy: 0.8957
accuracy: 0.8972
accuracy: 0.8966
accuracy: 0.8972
accuracy: 0.8982
accuracy: 0.8975
accuracy: 0.8981
accuracy: 0.8976
accuracy: 0.8984
accuracy: 0.8981
隐藏层大小 10 正则化系数 1e-05 学习率 0.001
accuracy: 0.103
accuracy: 0.1135
accuracy: 0.1009
accuracy: 0.1135
accuracy: 0.101
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 10 正则化系数 1e-05 学习率 0.0001
accuracy: 0.882
accuracy: 0.8939
accuracy: 0.8755
accuracy: 0.8951
accuracy: 0.8917
accuracy: 0.9017
accuracy: 0.901
accuracy: 0.8985
accuracy: 0.897
隐藏层大小 10 正则化系数 1e-05 学习率 1e-05
accuracy: 0.905
accuracy: 0.915
accuracy: 0.9172
accuracy: 0.9193
accuracy: 0.9223
accuracy: 0.9222
accuracy: 0.9218
accuracy: 0.9204
accuracy: 0.9221
accuracy: 0.9225
accuracy: 0.9211
accuracy: 0.9223
accuracy: 0.9235
accuracy: 0.9225
隐藏层大小 10 正则化系数 1e-05 学习率 1e-06
accuracy: 0.8303
accuracy: 0.8408
accuracy: 0.8444
accuracy: 0.8517
accuracy: 0.8517
accuracy: 0.852
accuracy: 0.8514
accuracy: 0.8484
accuracy: 0.848
隐藏层大小 30 正则化系数 0.01 学习率 0.001
accuracy: 0.0974
accuracy: 0.0974
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.1028
隐藏层大小 30 正则化系数 0.01 学习率 0.0001
accuracy: 0.9291
accuracy: 0.943
accuracy: 0.9427
accuracy: 0.9467
accuracy: 0.947
accuracy: 0.9519
accuracy: 0.9485
accuracy: 0.9515
accuracy: 0.9534
accuracy: 0.9523
accuracy: 0.9524
accuracy: 0.953
accuracy: 0.9545
accuracy: 0.9541
accuracy: 0.9531
隐藏层大小 30 正则化系数 0.01 学习率 1e-05
accuracy: 0.9345
accuracy: 0.939
accuracy: 0.9438
accuracy: 0.9439
accuracy: 0.9459
accuracy: 0.9476
accuracy: 0.949
accuracy: 0.9483
accuracy: 0.948
accuracy: 0.9499
accuracy: 0.9487
accuracy: 0.9498
accuracy: 0.9497
隐藏层大小 30 正则化系数 0.01 学习率 1e-06
accuracy: 0.8769
accuracy: 0.9074
accuracy: 0.9163
accuracy: 0.9196
accuracy: 0.9215
accuracy: 0.923
accuracy: 0.9253
accuracy: 0.9258
accuracy: 0.9274
accuracy: 0.9262
accuracy: 0.9282
accuracy: 0.9281
accuracy: 0.9284
accuracy: 0.9278
accuracy: 0.9278
隐藏层大小 30 正则化系数 0.001 学习率 0.001
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 30 正则化系数 0.001 学习率 0.0001
accuracy: 0.931
accuracy: 0.9352
accuracy: 0.9423
accuracy: 0.9483
accuracy: 0.9492
accuracy: 0.9526
accuracy: 0.9534
accuracy: 0.9554
accuracy: 0.953
accuracy: 0.9539
accuracy: 0.9549
accuracy: 0.9551
accuracy: 0.9539
accuracy: 0.9552
accuracy: 0.9554
accuracy: 0.9549
accuracy: 0.9559
accuracy: 0.9555
accuracy: 0.9553
隐藏层大小 30 正则化系数 0.001 学习率 1e-05
accuracy: 0.9279
accuracy: 0.9341
accuracy: 0.9404
accuracy: 0.9409
accuracy: 0.9409
accuracy: 0.9435
accuracy: 0.944
accuracy: 0.9447
accuracy: 0.9455
accuracy: 0.9453
accuracy: 0.9457
accuracy: 0.947
accuracy: 0.946
accuracy: 0.9464
accuracy: 0.9465
accuracy: 0.9468
accuracy: 0.9467
隐藏层大小 30 正则化系数 0.001 学习率 1e-06
accuracy: 0.8847
accuracy: 0.9026
accuracy: 0.9136
accuracy: 0.9177
accuracy: 0.9195
accuracy: 0.9213
accuracy: 0.9223
accuracy: 0.9247
accuracy: 0.9258
accuracy: 0.9258
accuracy: 0.9273
accuracy: 0.9274
accuracy: 0.9267
accuracy: 0.9274
accuracy: 0.9278
accuracy: 0.928
accuracy: 0.9277
accuracy: 0.9279
accuracy: 0.928
accuracy: 0.9279
隐藏层大小 30 正则化系数 0.0001 学习率 0.001
accuracy: 0.0958
accuracy: 0.1135
accuracy: 0.101
accuracy: 0.098
accuracy: 0.1135
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.101
隐藏层大小 30 正则化系数 0.0001 学习率 0.0001
accuracy: 0.9254
accuracy: 0.9238
accuracy: 0.9403
accuracy: 0.9463
accuracy: 0.9464
accuracy: 0.9492
accuracy: 0.9521
accuracy: 0.9522
accuracy: 0.9511
accuracy: 0.9519
accuracy: 0.9526
accuracy: 0.9528
accuracy: 0.9522
accuracy: 0.9523
accuracy: 0.9519
accuracy: 0.9524
accuracy: 0.9511
隐藏层大小 30 正则化系数 0.0001 学习率 1e-05
accuracy: 0.9346
accuracy: 0.9429
accuracy: 0.9448
accuracy: 0.9443
accuracy: 0.9466
accuracy: 0.9492
accuracy: 0.9493
accuracy: 0.9505
accuracy: 0.9521
accuracy: 0.9512
accuracy: 0.952
accuracy: 0.9523
accuracy: 0.9523
accuracy: 0.9524
accuracy: 0.9524
隐藏层大小 30 正则化系数 0.0001 学习率 1e-06
accuracy: 0.8767
accuracy: 0.9015
accuracy: 0.9118
accuracy: 0.9157
accuracy: 0.9194
accuracy: 0.9207
accuracy: 0.9222
accuracy: 0.9246
accuracy: 0.9241
accuracy: 0.9256
accuracy: 0.9256
accuracy: 0.9261
accuracy: 0.9271
accuracy: 0.927
accuracy: 0.9272
accuracy: 0.9277
accuracy: 0.9275
隐藏层大小 30 正则化系数 1e-05 学习率 0.001
accuracy: 0.1032
accuracy: 0.101
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.0958
accuracy: 0.1135
accuracy: 0.101
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 30 正则化系数 1e-05 学习率 0.0001
accuracy: 0.9265
accuracy: 0.9439
accuracy: 0.9483
accuracy: 0.9482
accuracy: 0.9463
accuracy: 0.9502
accuracy: 0.9534
accuracy: 0.9527
accuracy: 0.953
accuracy: 0.9524
accuracy: 0.9536
accuracy: 0.9527
隐藏层大小 30 正则化系数 1e-05 学习率 1e-05
accuracy: 0.934
accuracy: 0.9398
accuracy: 0.9455
accuracy: 0.9443
accuracy: 0.9468
accuracy: 0.9478
accuracy: 0.9505
accuracy: 0.9502
accuracy: 0.9502
accuracy: 0.9502
隐藏层大小 30 正则化系数 1e-05 学习率 1e-06
accuracy: 0.8812
accuracy: 0.9082
accuracy: 0.9151
accuracy: 0.9195
accuracy: 0.9224
accuracy: 0.9248
accuracy: 0.9262
accuracy: 0.9267
accuracy: 0.9281
accuracy: 0.9295
accuracy: 0.9299
accuracy: 0.9294
accuracy: 0.9294
accuracy: 0.9307
accuracy: 0.9303
accuracy: 0.9306
accuracy: 0.9307
accuracy: 0.9308
accuracy: 0.9308
隐藏层大小 100 正则化系数 0.01 学习率 0.001
accuracy: 0.0974
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1028
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 100 正则化系数 0.01 学习率 0.0001
accuracy: 0.9481
accuracy: 0.9605
accuracy: 0.9671
accuracy: 0.9727
accuracy: 0.9725
accuracy: 0.9728
accuracy: 0.9742
accuracy: 0.9748
accuracy: 0.9742
accuracy: 0.9756
accuracy: 0.9751
accuracy: 0.9752
accuracy: 0.9753
accuracy: 0.9757
accuracy: 0.9759
accuracy: 0.9751
accuracy: 0.9758
accuracy: 0.9759
accuracy: 0.9754
隐藏层大小 100 正则化系数 0.01 学习率 1e-05
accuracy: 0.9531
accuracy: 0.9619
accuracy: 0.9658
accuracy: 0.9655
accuracy: 0.971
accuracy: 0.9709
accuracy: 0.971
accuracy: 0.9711
accuracy: 0.9713
accuracy: 0.9716
accuracy: 0.9716
accuracy: 0.9723
accuracy: 0.9716
隐藏层大小 100 正则化系数 0.01 学习率 1e-06
accuracy: 0.8966
accuracy: 0.9163
accuracy: 0.9245
accuracy: 0.9314
accuracy: 0.9339
accuracy: 0.9371
accuracy: 0.9384
accuracy: 0.9385
accuracy: 0.9394
accuracy: 0.9415
accuracy: 0.9412
accuracy: 0.9419
accuracy: 0.9427
accuracy: 0.9427
accuracy: 0.9429
accuracy: 0.9428
accuracy: 0.9432
accuracy: 0.9436
accuracy: 0.9434
隐藏层大小 100 正则化系数 0.001 学习率 0.001
accuracy: 0.101
accuracy: 0.1009
accuracy: 0.1135
accuracy: 0.1028
accuracy: 0.1028
accuracy: 0.1028
隐藏层大小 100 正则化系数 0.001 学习率 0.0001
accuracy: 0.9495
accuracy: 0.9632
accuracy: 0.9697
accuracy: 0.9697
accuracy: 0.9694
accuracy: 0.9713
accuracy: 0.9739
accuracy: 0.9741
accuracy: 0.9738
accuracy: 0.9734
accuracy: 0.9733
隐藏层大小 100 正则化系数 0.001 学习率 1e-05
accuracy: 0.9509
accuracy: 0.9626
accuracy: 0.9646
accuracy: 0.9649
accuracy: 0.9676
accuracy: 0.9689
accuracy: 0.9697
accuracy: 0.97
accuracy: 0.9694
accuracy: 0.9696
accuracy: 0.9701
accuracy: 0.9692
accuracy: 0.9691
accuracy: 0.9697
accuracy: 0.9699
accuracy: 0.9702
accuracy: 0.97
隐藏层大小 100 正则化系数 0.001 学习率 1e-06
accuracy: 0.8872
accuracy: 0.9146
accuracy: 0.9236
accuracy: 0.9282
accuracy: 0.9324
accuracy: 0.9343
accuracy: 0.9367
accuracy: 0.9382
accuracy: 0.9399
accuracy: 0.9406
accuracy: 0.9409
accuracy: 0.9413
accuracy: 0.942
accuracy: 0.9423
accuracy: 0.9424
accuracy: 0.9423
accuracy: 0.9424
accuracy: 0.9424
accuracy: 0.9426
accuracy: 0.9427
accuracy: 0.9427
accuracy: 0.9427
隐藏层大小 100 正则化系数 0.0001 学习率 0.001
accuracy: 0.1135
accuracy: 0.1028
accuracy: 0.1028
accuracy: 0.098
accuracy: 0.1032
accuracy: 0.101
隐藏层大小 100 正则化系数 0.0001 学习率 0.0001
accuracy: 0.9503
accuracy: 0.9638
accuracy: 0.9666
accuracy: 0.9704
accuracy: 0.972
accuracy: 0.9713
accuracy: 0.9716
accuracy: 0.9734
accuracy: 0.9731
accuracy: 0.9733
accuracy: 0.9743
accuracy: 0.9745
accuracy: 0.9749
accuracy: 0.9742
accuracy: 0.9744
accuracy: 0.9749
accuracy: 0.9745
accuracy: 0.975
accuracy: 0.9743
隐藏层大小 100 正则化系数 0.0001 学习率 1e-05
accuracy: 0.9503
accuracy: 0.9631
accuracy: 0.9669
accuracy: 0.9681
accuracy: 0.9691
accuracy: 0.969
accuracy: 0.9698
accuracy: 0.9705
accuracy: 0.9703
accuracy: 0.9708
accuracy: 0.9706
accuracy: 0.9708
accuracy: 0.9713
accuracy: 0.9713
隐藏层大小 100 正则化系数 0.0001 学习率 1e-06
accuracy: 0.8906
accuracy: 0.9179
accuracy: 0.9249
accuracy: 0.932
accuracy: 0.935
accuracy: 0.9371
accuracy: 0.939
accuracy: 0.9399
accuracy: 0.9395
accuracy: 0.9409
accuracy: 0.9414
accuracy: 0.9408
accuracy: 0.9415
accuracy: 0.9417
accuracy: 0.9424
accuracy: 0.9415
accuracy: 0.9417
accuracy: 0.942
accuracy: 0.9418
隐藏层大小 100 正则化系数 1e-05 学习率 0.001
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
accuracy: 0.1135
隐藏层大小 100 正则化系数 1e-05 学习率 0.0001
accuracy: 0.954
accuracy: 0.9611
accuracy: 0.9667
accuracy: 0.9707
accuracy: 0.9705
accuracy: 0.9708
accuracy: 0.9733
accuracy: 0.9738
accuracy: 0.9733
accuracy: 0.9732
accuracy: 0.9733
accuracy: 0.973
accuracy: 0.9736
accuracy: 0.973
隐藏层大小 100 正则化系数 1e-05 学习率 1e-05
accuracy: 0.951
accuracy: 0.9598
accuracy: 0.9656
accuracy: 0.9677
accuracy: 0.9677
accuracy: 0.9688
accuracy: 0.9679
accuracy: 0.9697
accuracy: 0.9704
accuracy: 0.9695
accuracy: 0.9699
accuracy: 0.9702
accuracy: 0.9708
accuracy: 0.9709
accuracy: 0.971
accuracy: 0.9711
accuracy: 0.9705
隐藏层大小 100 正则化系数 1e-05 学习率 1e-06
accuracy: 0.8934
accuracy: 0.9171
accuracy: 0.9242
accuracy: 0.9312
accuracy: 0.9349
accuracy: 0.936
accuracy: 0.9379
accuracy: 0.9387
accuracy: 0.939
accuracy: 0.94
accuracy: 0.9403
accuracy: 0.941
accuracy: 0.9409
accuracy: 0.9412
accuracy: 0.9415
accuracy: 0.9418
accuracy: 0.9419
accuracy: 0.9419
accuracy: 0.9421
accuracy: 0.9422
accuracy: 0.942
accuracy: 0.9422
accuracy: 0.9422
'''