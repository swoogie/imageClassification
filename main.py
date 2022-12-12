from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import ParameterGrid
from keras.datasets import cifar10
from keras import models, layers
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def displayDataset():
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])

    plt.show()


def createModel(activationFun):
    model = models.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))
    model.add(layers.Rescaling(scale=1.0 / 255))
    model.add(layers.Conv2D(
        32, (3, 3), activation=activationFun, input_shape=(32, 32, 3)))
    model.add((layers.MaxPooling2D(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activationFun))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activationFun))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=activationFun))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def trainModel(train_images, train_labels,
               epochs=10,
               activation='relu',
               batch_size=32,
               optimizer='adam',
               setNum=1):
    model = createModel(activation)
    global debounce
    if not debounce:
        model.summary()
        debounce = True

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_split=0.2, batch_size=batch_size)

    model.save(f'models/modelWSet{setNum}')

    return model, history


def evaluateModel(model, history, test_images, test_labels, **kwargs):
    if not history == None:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'model accuracy\n{kwargs.__str__()}')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'model loss\n{kwargs.__str__()}')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    loss, accuracy = model.evaluate(test_images, test_labels)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    return accuracy


if __name__ == '__main__':
    debounce = False
    # using cifar from https://www.cs.toronto.edu/~kriz/cifar.html
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    class_names = ['Plane', 'Car', 'Bird', 'Cat',
                   'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    displayDataset()

    grid = ParameterGrid({'activation': ['relu', 'sigmoid'], 'batch_size': [
                         32, 64, 128], 'optimizer': ['sgd', 'adam']})

    results = pd.DataFrame(list(grid))

    allAccuracies = []
    allModels = []
    allHistories = []

    for i in range(len(results)):
        if not Path(f'models/modelWSet{i}').exists():
            model, history = trainModel(
                train_images, train_labels, **grid[i], setNum=i)
            accuracy = evaluateModel(
                model, history, test_images, test_labels, **grid[i])
            allAccuracies.append(accuracy)
            allModels.append(model)
            allHistories.append(history)
            print(f'iteration {i}')

    for i in range(len(results)):
        model = models.load_model(f'models/modelWSet{i}')
        accuracy = evaluateModel(model, None, test_images, test_labels)
        allAccuracies.append(accuracy)
        allModels.append(models)

    bestIndex = allAccuracies.index(max(allAccuracies))

    print(f'most accurate set: modelWSet{bestIndex}')
    print('model results:')
    results['test_accuracies'] = allAccuracies
    print(results)

    bestModel = models.load_model(f'models/modelWSet{bestIndex}')

    yPred = bestModel.predict(test_images)
    yPred = [np.argmax(i) for i in yPred]
    confusion_matrix = confusion_matrix(test_labels, yPred)
    print(confusion_matrix)
    testAccuracy = accuracy_score(test_labels, yPred)
    print(f'Best model accuracy: {testAccuracy}')

    resultsdf = pd.DataFrame(classification_report(
        test_labels, yPred, output_dict=True))
    resultsdf.columns = np.concatenate(
        (class_names,  resultsdf.columns[10:].values))
    resultsdf

    print('prediction fragment:')

    label = []
    prediction = []
    correct = []

    indices = np.random.choice(
        np.where(np.array(yPred) == np.array(yPred))[0], 30)
    for i in indices:
        correct.append(yPred[i] == test_labels[i][0])
        prediction.append(class_names[yPred[i]])
        label.append(class_names[test_labels[i][0]])

    df = pd.DataFrame(
        {'Correct': correct, 'Label': label, 'Prediction': prediction})
    print(df)
