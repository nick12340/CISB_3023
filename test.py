import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv
for i in range(15):
    root_path ='TrafficSignData/'
    data_path = root_path +'Training/' + format(i,'05d') + '/'
    csv_name = 'GT-' + format(i,'05d') + '.csv'
    csv_file = open(data_path + csv_name,'r')
    reader = csv.reader(csv_file,delimiter = ';')
    next(reader)
    images = []
    for row in reader:
        img_name = row[0]
        box = (int(row[4]),int(row[3]),int(row[6]),int(row[5]))
        img = Image.open(data_path + img_name)
        img = img.crop(box)
        img = img.resize((32,32),Image.BICUBIC)
    plt.imshow(img)
    plt.show()
