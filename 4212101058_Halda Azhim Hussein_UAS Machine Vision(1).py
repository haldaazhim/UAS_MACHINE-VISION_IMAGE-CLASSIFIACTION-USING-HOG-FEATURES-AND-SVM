import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mlxtend.plotting import plot_confusion_matrix

# Muat dataset MNIST
mnist = datasets.fetch_openml('mnist_784', version=1)
X, y = mnist.data.astype('float32'), mnist.target.astype('int')

# Bagi dataset menjadi data latih dan uji
X_train, y_train = np.array(X[:1000]), np.array(y[:1000])
X_test, y_test = np.array(X[:10]), np.array(y[:10])


# Ekstraksi fitur HOG untuk data latih
hog_features_train = []
hog_images_train = []
for image in X_train:
    feature, hog_img = hog(image.reshape((28, 28)), 
                           orientations = 9, 
                           pixels_per_cell = (8,8), 
                           cells_per_block = (2,2), 
                           visualize = True, 
                           block_norm = 'L2')
    hog_features_train.append(feature)
    hog_images_train.append(hog_img)

hog_features_train_np = np.array(hog_features_train)
hog_images_train_np = np.array(hog_images_train)
    

# Ekstraksi fitur HOG untuk data uji
hog_features_test = []
hog_images_test = []
for image in X_test:
    feature, hog_img = hog(image.reshape((28, 28)), 
                           orientations = 9, 
                           pixels_per_cell = (8,8), 
                           cells_per_block = (2,2), 
                           visualize = True, 
                           block_norm = 'L2')
    hog_features_test.append(feature)
    hog_images_test.append(hog_img)

hog_features_test_np = np.array(hog_features_test)
hog_images_test_np = np.array(hog_images_test)    
 

# Normalisasi fitur HOG
scaler = StandardScaler()
hog_features_train_scaled = scaler.fit_transform(hog_features_train_np)
hog_features_test_scaled = scaler.transform(hog_features_test_np)

# Latih model SVM
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(hog_features_train_scaled, y_train)

# Lakukan prediksi pada data uji
predictions = svm_model.predict(hog_features_test_scaled)

# Evaluasi performa
conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')

# Tampilkan hasil evaluasi
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Class prediction:", predictions)
print("Class test:", y_test)

def plot_combined(X_test, hog_images_test):
    fig, axes = plt.subplots(2, 10, figsize=(10, 5))
    class_name = y_test
    # Plot untuk gambar dataset
    for i in range(min(len(X_test), 10)):  
        axes[0, i].imshow(X_test[i].reshape((28, 28)), cmap=cm.Greys_r)
        axes[0, i].axis('off')

    # Plot untuk gambar hasil HOG extraction
    for i in range(min(len(hog_images_test), 10)):  
        axes[1, i].imshow(hog_images_test[i].reshape((28, 28)), cmap=cm.Greys_r)
        axes[1, i].axis('off')

    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix)

    plt.show()

plot_combined(X_test, hog_images_test)









