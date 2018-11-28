import csv
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score

def data_preprocess(root_path):
    data = {}
    for i in range(15):
        data_path = root_path +'Training/' + format(i,'05d') + '/'
        csv_name = 'GT-' + format(i,'05d') + '.csv'
        csv_file = open(data_path + csv_name,'r')
        reader = csv.reader(csv_file,delimiter = ';')
        next(reader)
        images = []
        for row in reader:
            img_name = row[0]
        #    box = (int(row[4]),int(row[3]),int(row[6]),int(row[5]))
            img = Image.open(data_path + img_name)
        #    img = img.crop(box)
            img = img.resize((32,32),Image.BICUBIC)
            img = np.array(img)
            images.append(img)
        data[i] = np.array(images)
        csv_file.close()
    f = open(root_path + 'data.txt','wb')
    pickle.dump(data,f)
    f.close()


if __name__ == '__main__':
    root_path ='TrafficSignData/'
    #data_preprocess(root_path)
    f = open(root_path + 'data.txt','rb')
    data = pickle.load(f)
    lin_clf = svm.SVC()
    X_training = []
    Y_training = []
    X_testing = []
    Y_testing = []
    plt.show()
    for i in range(15):
        images_num = (len(data[i]))
        train_size = images_num * 4 //5
        for j in range(train_size):
            X_training.append(data[i][j].flatten())
        for j in range(train_size,images_num):
            X_testing.append(data[i][j].flatten())
        Y_training = np.append(Y_training,np.ones(train_size) * i)
        Y_testing = np.append(Y_testing,np.ones(images_num-train_size)*i)
    X_training = np.array(X_training)
    #lin_clf.fit(X_training,Y_training)
    #with open('SVM.pickle','wb') as f:
    #    pickle.dump(lin_clf,f)
    #f.close()
    with open('fuck.pickle','rb') as f:
        lin_clf = pickle.load(f)
    f.close()
    print ('finish writing')
    Y_predict = lin_clf.predict(X_testing)
    print (accuracy_score(Y_predict,Y_testing) )
    