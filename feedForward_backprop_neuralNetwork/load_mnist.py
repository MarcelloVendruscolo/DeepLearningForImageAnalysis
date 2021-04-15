import numpy as np
import imageio
import glob

def load_mnist(folder_path):
    
    NUM_LABELS = 10 # Number of different classes (digits ranging from 0 to 9)

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for label in range(NUM_LABELS):
        for image_path in glob.glob(str(folder_path) + "/Train/" + str(label) + "/*.png"):
            image = imageio.imread(image_path)
            train_images.append(image)
            letter = [0 for _ in range(0,NUM_LABELS)]    
            letter[label] = 1
            train_labels.append(letter)  
    
    for label in range(NUM_LABELS):
        for image_path in glob.glob(str(folder_path) + "/Test/" + str(label) + "/*.png"):
            image = imageio.imread(image_path)
            test_images.append(image)
            letter = [0 for _ in range(0,NUM_LABELS)]    
            letter[label] = 1
            test_labels.append(letter)                  
            
    X_train= np.array(train_images).reshape(-1,784)/255.0
    Y_train= np.array(train_labels)
    X_test= np.array(test_images).reshape(-1,784)/255.0
    Y_test= np.array(test_labels)
    
    return X_train, Y_train, X_test, Y_test
