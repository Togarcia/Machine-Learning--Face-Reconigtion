import numpy as np
import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import os
from imutils import paths
import win32con
from sklearn.model_selection import train_test_split
from tkinter import messagebox

window = tk.Tk()  
window.wm_title("Bindi")
window.config(background="#FFFFFF")
window.config(width=250, height=200)
window.geometry("400x400")
img = ImageTk.PhotoImage(Image.open('images/train.gif'),format="gif -index 2")

sliderFrame = tk.Frame(window, width=150, height=100)
sliderFrame.pack(side="bottom")


label = Label(window, image = img)
label.pack()


def abrir_FExplo():
    import subprocess
    subprocess.Popen('explorer data"')



def entrenar():
    
    dataset="data"
    imagePaths = paths.list_images(dataset)
    data = []
    labels = []
    nombres = []

    for imagePath in imagePaths:
        image = Image.open(imagePath)
        image = np.array(image.resize((120, 120))) / 255.0
        label = imagePath.split(os.path.sep)[-2]
        data.append(image)
        if label not in nombres:
            nombres.append(label)

        labels.append(label)
    from sklearn.preprocessing import LabelBinarizer
    from keras.layers import Dropout
    from keras.models import Sequential
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.layers.core import Activation
    from keras.layers.core import Flatten
    from keras.layers.core import Dense
    from keras.optimizers import Adam
    from keras import optimizers
    nombreModelo = nombreTrain.get("1.0","end-1c")
    imagePaths = paths.list_images(dataset)

    print("Hola estoi apunto de entrar")

    print(len(nombres))
    print(len(labels))
    print(len(data))
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    (trainX, testX, trainY, testY) = train_test_split(np.array(data),
	    np.array(labels), test_size=0.3)
    
    model = Sequential()
    #Primera capa, de 8. aquesta es una capa d'input a on li entren els 32*32 pixels de l'imatge
    model.add(Conv2D(8, (3, 3), padding="same", input_shape=(120, 120, 3)))
    #funcio d'activació. determina com de "segur" ha d'estar per a tornar 1.
    model.add(Activation("relu"))
    #El max pooling serveix per que la xarxa no quedi inmensa i tardi anys
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #Segona capa
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))
    #Tercera capa
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #quarta
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #Afegeix una capa final 
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(len(nombres)))
    model.add(Activation("softmax"))

    opt = Adam(lr=1e-3, decay=1e-3 / 50)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=10, batch_size=32)
    test_loss, test_acc = model.evaluate(testX, testY)
    test_acc = round(test_acc,2)
    MsgBox = messagebox.askquestion ('Entrenaiento finalizado','Desea guardar este modelo con una precicion de: ' + str(test_acc),icon = 'warning')

    if len(nombreModelo) == 0:
        t = os.listdir("models") 
        nombreModelo = "Bidi" + str(len(t)+1)
    if MsgBox == "yes":
        model.save("models/"+nombreModelo)
    else:
        print("No")    










botonFileExplorer = Button(sliderFrame, text="Modificar datos entrenamiento", command=abrir_FExplo)
botonFileExplorer.pack()

botonTrain = Button(sliderFrame, text="Entrenar", command=entrenar)
botonTrain.pack()

nombreTrain = tk.Text(sliderFrame, height=1, width=25)
nombreTrain.pack()
window.mainloop() 