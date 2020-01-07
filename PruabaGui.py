import numpy as np
import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import os
from imutils import paths
from keras.models import model_from_json, load_model
import win32con
from sklearn.model_selection import train_test_split
global cargado 
cargado = False
dataset = "data"
global nombres
nombres = []

imagePaths = paths.list_images(dataset)
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    if label not in nombres:
        nombres.append(label)
face_dir = ('haars/haarcascade_frontalface_default.xml')
rostroCascade = cv2.CascadeClassifier(face_dir)

window = tk.Tk()  
window.wm_title("Bindi")
window.config(background="#FFFFFF")


imageFrame = tk.Frame(window, width=100, height=100)
imageFrame.grid(row=0, column=0, padx=10, pady=2)
global pedrin

camaraRecon = tk.Label(imageFrame)
lmain = tk.Label(imageFrame)
lmain.grid(row=1, column=0)
cap = cv2.VideoCapture(0)
pedrin= 0

def show_frame():
    lmain.grid(row=1, column=0)
    camaraRecon.grid_forget()
    global pedrin
    _, frame = cap.read()
    detectar = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = rostroCascade.detectMultiScale(
        detectar,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
        )
    for(x, y, w, h) in rostros:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    if pedrin==0:
        lmain.after(10, show_frame)
        #print("funcion 1")
    else:
        return

def exitall():
    iniciamos = True
    global pedrin
    if pedrin == 0:
        iniciamos = False
    print("Hola",iniciamos)
    pedrin = 0
    camaraRecon.grid_forget()   
    lmain.grid(row=1, column=0)
    if iniciamos == True:
        show_frame()

def entrar():
    iniciamos = True
    global pedrin
    
    if pedrin == 1:
        iniciamos = False
    pedrin = 1
    lmain.grid_forget()   
    camaraRecon.grid(row=1, column=0)
    print(iniciamos)
    if iniciamos == True:
        iniciar_recon()
global model        

def iniciar_recon():
    nombre = nombreModelo.get("1.0","end-1c")
    archivos = os.listdir("models")
    if nombre in archivos:
        print("entre")
        modelo = nombre
    else:
        modelo = "ModeloBidi.h5"
    print(type(modelo))
    lmain.grid_forget()
    global model
    global cargado
    if cargado == False:
        print("entro carga")
        model = load_model("models/"+ str(modelo))
        cargado = True
    global pedrin
    
    model.summary()
    camaraRecon.grid(row=1, column=0)
    _, frame = cap.read()
    detectar = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = rostroCascade.detectMultiScale(
        detectar,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
        )
    for(x, y, w, h) in rostros:
        face = frame[y:y + h, x:x + w]
        #Rescalamos imagen a 120x120
        face_resize = cv2.resize(face, (120, 120))
        #Pasamos la imagen a una numpy Array
        image = np.array(face_resize)
        #Añadimos una 4 dimension
        image = np.expand_dims(image, axis=0)
        #Dividimos por el numero de pixeles que queremos
        image = image / 255.0
        #predecimos la imagen
        prediction = model.predict(image)
        #almacenamos la predicion en una array
        array = prediction[0]
        #Obteemos el valor maximo con el objetivo se saber cuala es la precisión
        precision = np.max(array)
        #Sacamos el index de la array con el fin de identificar que sujeto tiene mas probabilidad
        max = np.nanargmax(array)
        #Dibujamos rectangulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)    
        if(precision > 0.80):
            indexReal = max
            cv2.putText(frame, nombres[indexReal], (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (128, 0, 255))
        else:
            cv2.putText(frame, "Desconocido", (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (128, 0, 255))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    camaraRecon.imgtk = imgtk
    camaraRecon.configure(image=imgtk)
    if pedrin == 1:
        camaraRecon.after(10, iniciar_recon)
    else:
        return


#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=150, height=100)
sliderFrame.grid(row = 1, column=0, padx=0, pady=0)

botonInciarRecon = Button(sliderFrame, text="Iniciar reconozimiento", command=entrar)
botonInciarRecon.pack()

nombreModelo = tk.Text(sliderFrame, height=1, width=25)
nombreModelo.pack()

botonGoBack = Button(sliderFrame, text="Atras", command=exitall)
botonGoBack.pack()



show_frame()
window.mainloop()  #Starts GUI
