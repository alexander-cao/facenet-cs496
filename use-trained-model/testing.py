import pylab as plt
import numpy as np

z_train = np.load('train-embeddings.npy')
train_labels = np.load('train-labels.npy')

z_train_centroids = np.zeros((63, 128))
for i in range(63):
    find = np.where(train_labels == i)[0]
    z_temp = z_train[find]
    z_temp_mean = np.mean(z_temp, 0)
    z_train_centroids[i] = z_temp_mean

z_test = np.load('test-embeddings.npy')
test_labels = np.load('test-labels.npy')

find = np.where(test_labels <= 62)[0]
z_test_closed = z_test[find]
test_labels_closed = test_labels[find]

n_samples_closed = len(test_labels_closed)
guess = np.zeros(n_samples_closed)
for i in range(n_samples_closed):
    temp = np.sum(np.square(z_train_centroids - z_test_closed[i]), 1)
    guess[i] = np.argmin(temp)

error1_closed = np.sum(np.sign(np.abs(test_labels_closed-guess))) / n_samples_closed


guesses = np.zeros((n_samples_closed, 6))
for i in range(n_samples_closed):
    temp = np.sum(np.square(z_train_centroids - z_test_closed[i]), 1)
    guesses[i] = np.argpartition(temp, 6)[0:6] - test_labels_closed[i]

error6_closed = 1 - (np.count_nonzero(guesses==0) / n_samples_closed)

find = np.where(test_labels >= 63)[0]
np.random.shuffle(find)
find = find[0:1260]

z_test_unknown = z_test[find]
test_labels_unknown = test_labels[find]
test_labels_unknown = 0*test_labels_unknown - 99

z_test_open = np.concatenate((z_test_closed, z_test_unknown), axis = 0)
test_labels_open = np.concatenate((test_labels_closed, test_labels_unknown), axis = 0)

taus = np.arange(0, 1, 0.01)
error1_open = np.zeros(len(taus))
n_samples_open = len(test_labels_open)
for j in range(len(taus)):
    tau = taus[j]
    
    guess = np.zeros(n_samples_open)
    for i in range(n_samples_open):
        temp = np.sum(np.square(z_train_centroids - z_test_open[i]), 1)
        if np.amin(temp) > tau:
            guess[i] = -99
        else:
            guess[i] = np.argmin(temp)
            
    error1_open[j] = np.sum(np.sign(np.abs(test_labels_open-guess))) / n_samples_open
    
plt.figure(figsize = (8, 6))
plt.plot(taus, error1_open, 'bs')
plt.xlabel('threshold values', fontsize = 20)
plt.ylabel('error', fontsize = 20)
plt.grid()
plt.show()