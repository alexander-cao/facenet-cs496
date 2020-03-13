import pylab as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

z_train = np.load('train-embeddings.npy')
z_train_tsne = TSNE(n_components = 2, random_state = 815).fit_transform(z_train)

train_labels = np.load('train-labels.npy')
train_images = np.load('train-images.npy')

# a = np.random.randint(9295)
# b = a + 1
# c = np.random.randint(9295)
a = 1843
b = 1844
c = 9124

plt.figure(figsize = (8, 6))
plt.scatter(z_train_tsne[:, 0], z_train_tsne[:, 1], c = train_labels)
plt.grid()
plt.xlabel('tsne dim 1', fontsize = 20)
plt.ylabel('tsne dim 2', fontsize = 20)
plt.colorbar()
plt.scatter(z_train_tsne[a, 0], z_train_tsne[a, 1], s = 250, marker = '*', c = 'red', edgecolor = 'black')
plt.scatter(z_train_tsne[b, 0], z_train_tsne[b, 1], s = 250, marker = '*', c = 'red', edgecolor = 'black')
plt.scatter(z_train_tsne[c, 0], z_train_tsne[c, 1], s = 250, marker = '*', c = 'red', edgecolor = 'black')
plt.show()

plt.figure(figsize = (8, 6))
plt.subplot(1, 3, 1)
plt.imshow(train_images[a].astype(np.uint8))
plt.subplot(1, 3, 2)
plt.imshow(train_images[b].astype(np.uint8))
plt.subplot(1, 3, 3)
plt.imshow(train_images[c].astype(np.uint8))
plt.show()

inertias = np.zeros(40)
for i in range(40):
    kmeans = KMeans(n_clusters = 2*i+1, random_state = 815).fit(z_train)
    inertias[i] = kmeans.inertia_
    print(i)
    
plt.figure(figsize = (8, 6))
plt.plot(np.arange(1, 2*len(inertias)+1, 2), inertias, 'bs')
plt.xlabel('k', fontsize = 20)
plt.ylabel('inertia', fontsize = 20)
plt.grid()
plt.show()



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

z_test_closed_w_train_centroids = np.concatenate((z_train_centroids, z_test_closed), axis = 0)
z_test_closed_w_train_centroids_tsne = TSNE(n_components = 2, random_state = 815).fit_transform(z_test_closed_w_train_centroids)

plt.figure(figsize = (8, 6))
plt.scatter(z_test_closed_w_train_centroids_tsne[63:, 0], z_test_closed_w_train_centroids_tsne[63:, 1], c = test_labels_closed)
plt.grid()
plt.xlabel('tsne dim 1', fontsize = 20)
plt.ylabel('tsne dim 2', fontsize = 20)
plt.colorbar()
plt.scatter(z_test_closed_w_train_centroids_tsne[0:63, 0], z_test_closed_w_train_centroids_tsne[0:63, 1], s = 250, marker = '*', c = 'red', edgecolor = 'black')
plt.show()

find = np.where(test_labels >= 63)[0]
np.random.shuffle(find)
find = find[0:100]

z_test_open_w_train_centroids = np.concatenate((z_test_closed_w_train_centroids, z_test[find]), axis = 0)

z_test_open_w_train_centroids_tsne = TSNE(n_components = 2, random_state = 815).fit_transform(z_test_open_w_train_centroids)

plt.figure(figsize = (8, 6))
plt.scatter(z_test_open_w_train_centroids_tsne[63:-100, 0], z_test_open_w_train_centroids_tsne[63:-100, 1], c = test_labels_closed)
plt.grid()
plt.xlabel('tsne dim 1', fontsize = 20)
plt.ylabel('tsne dim 2', fontsize = 20)
plt.colorbar()
plt.scatter(z_test_open_w_train_centroids_tsne[0:63, 0], z_test_open_w_train_centroids_tsne[0:63, 1], s = 250, marker = '*', c = 'red', edgecolor = 'black')
plt.scatter(z_test_open_w_train_centroids_tsne[-100:, 0], z_test_open_w_train_centroids_tsne[-100:, 1], s = 100, marker = 'o', c = 'black', edgecolor = 'black')
plt.show()