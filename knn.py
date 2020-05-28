import numpy as np
from sklearn import model_selection
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from statistics import mean, median,variance

class knn_classifier:

    def __init__(self, k):
        self.k = k
        self._data = None
        self._labels = None

    def fit(self, data, labels):
        self._data = data
        self._labels = labels

    def predict(self, test_data):
        distance_matrix = distance.cdist(test_data, train_data)
        indexes = np.argsort(distance_matrix, axis=1)
        class_dict = {}
        for label in labels:
            class_dict[label] = 0
        class_dict_list = []
        for data_num, index in enumerate(indexes[:,:self.k]):
            class_dict_list.append(class_dict.copy())
            for i in index:
                class_dict_list[data_num][self._labels[i]] += 1
        predict_class = []
        for d in class_dict_list:
            max_class = max(d, key=d.get)
            predict_class.append(max_class)
        return predict_class


def data_sprit_half(data, labels):
    train_data, test_data, train_label, test_label = model_selection.train_test_split(data,
                                                                                  labels,
                                                                                  train_size=len(data) // 2,
                                                                                  test_size=len(labels) // 2)
    return train_data, test_data, train_label, test_label


if __name__ == "__main__":

    data = np.loadtxt('iris.data', delimiter=',', dtype='float64',usecols=[0, 1, 2, 3])
    labels = np.loadtxt('iris.data', delimiter=',', dtype='str',usecols=[4])
    print(data.shape)
    print(labels.shape)
    
    sprint_num = 50
    knn_accs = []

    for i in range(0, 20):
        print("training number {} ...".format(i+1))
        train_data_c0, test_data_c0, train_labels_c0, test_labels_c0 = data_sprit_half(data[:sprint_num], labels[:sprint_num])
        train_data_c1, test_data_c1, train_labels_c1, test_labels_c1 = data_sprit_half(data[1 * sprint_num:sprint_num * 2 + 1], labels[1 * sprint_num:sprint_num * 2 + 1])
        train_data_c2, test_data_c2, train_labels_c2, test_labels_c2 = data_sprit_half(data[2 * sprint_num:sprint_num * 3 + 1], labels[2 * sprint_num:sprint_num * 3 + 1])
        train_data = np.concatenate([train_data_c0, train_data_c1, train_data_c2])
        test_data = np.concatenate([test_data_c0, test_data_c1, test_data_c2])
        train_labels = np.concatenate([train_labels_c0, train_labels_c1, train_labels_c2])
        test_labels = np.concatenate([test_labels_c0, test_labels_c1, test_labels_c2])

        K = knn_classifier(k=2)
        K.fit(train_data, train_labels)
        predict_labels = K.predict(test_data)
        acc = accuracy_score(test_labels, predict_labels)
        knn_accs.append(acc)
        print("knn accuracy : {}".format(acc))
        
    print("=================================================")
    knn_mean = mean(knn_accs)
    print("knn accuracy mean : {}".format(knn_mean))
    knn_variance = variance(knn_accs)
    print("knn accuracy variance : {}".format(knn_variance))
