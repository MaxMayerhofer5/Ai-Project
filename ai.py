import tkinter
import cv2
import tensorflow as tf
import numpy as numP
import os
import math 
import random
from tensorflow.keras import activations
from tensorflow.python.keras.layers import BatchNormalizationV2
from tensorflow.python.ops.control_flow_ops import switch_case
from tensorflow.python.ops.gen_array_ops import Size
from tensorflow.python.ops.gen_control_flow_ops import Switch, switch
import tensorflow_datasets as tfds
from tkinter import*
from tkinter import filedialog
from tensorflow import keras

#if 'filechosen' not in globals():
#    global bool; filechosen = FALSE
global bool; filechosen = FALSE
global mouse_pressed
mouse_pressed = False

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#      tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=2048),
#          tf.config.LogicalDeviceConfiguration(memory_limit=2048),
#          tf.config.LogicalDeviceConfiguration(memory_limit=2048)])

def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255., label  

drawing = numP.ndarray(shape=(28,28))
for i in range(0,28):
    for j in range(0,28):
        drawing[i,j] = 255




def browsePNGFiles():
    global filename
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("PNG files",
                                                        "*.png*"),
                                                       ("all files",
                                                        "*.*"))) 
    # getting image and getting scale 
    src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if (canvas.winfo_width() > canvas.winfo_height()):
       yscale = canvas.winfo_height()/int(src.shape[1])
       dsize = (int(int(src.shape[1])*yscale), int(int(src.shape[0])*yscale))

    if (canvas.winfo_width() < canvas.winfo_height()):
       xscale = canvas.winfo_width()/int(src.shape[0])
       dsize = (int(int(src.shape[1])*xscale), int(int(src.shape[0])*xscale))
    # writing new size img
    output = cv2.resize(src, dsize)
    cv2.imwrite(filename.strip(".png")+'_resized.png',output) 
    # placing image
    currentPNG = PhotoImage(file = filename.strip(".png")+"_resized.png")
    canvas.itemconfigure(img, image = currentPNG)
    print(canvas.nametowidget(img))


def buttoncallback1():
    global buttonClicked
    buttonClicked = False
    buttonClicked = not buttonClicked
    if (buttonClicked==True):
       browsePNGFiles()
    buttonClicked = not buttonClicked 

def Train():
    (em_train, em_test), em_info= tfds.load('emnist',split=['train', 'test'], shuffle_files=True,as_supervised=True,
    with_info=True)
    em_train = em_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    em_train = em_train.batch(256)
    em_test = em_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    em_test = em_test.batch(256)
    model = keras.Sequential()
    tf.keras.regularizers.l2(l2=0.00025)
    model.add(keras.layers.Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = "relu", input_shape = (28,28,1), kernel_regularizer='l2'))
    model.add(keras.layers.Conv2D(filters = 32, kernel_size = 5, strides = 1, use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = 2))
    model.add(tf.keras.layers.Dropout(.25))
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, kernel_regularizer='l2'))
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Dropout(.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2048, use_bias=False,  activation = "relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1024, use_bias=False,  activation = "relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, use_bias=False, activation = "relu"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(256, use_bias=False, activation = "relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(.25))
    model.add(keras.layers.Dense(128, use_bias=False, activation = "relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(.25))
    model.add(keras.layers.Dense(62, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics= ["accuracy"])
    model.fit(em_train, epochs = 40, validation_data=em_test)
    #model.save_weights(r'C:\Users\makot\Documents\AI project - Copy\model')
    model.save(r'C:\Users\makot\Documents\Final Ai thing')


def mouse_moving(event):
    x = canvas.winfo_pointerx() - canvas.winfo_rootx()
    y = canvas.winfo_pointery() - canvas.winfo_rooty()
    x = math.floor((x - (1000/2)+28*drawscale)/(2*drawscale)/2)
    y = math.floor((y - (600/2)+28*drawscale)/(2*drawscale)/2)
    #print (x,"and", y)
    if(x >= 0 and x < 14 and y >= 0 and y< 14): 
       drawing[y*2,x*2]=0
       drawing[y*2+1,x*2]=0
       drawing[y*2,x*2+1]=0
       drawing[y*2+1,x*2+1]=0
    drwng = cv2.resize(drawing,(28*2*drawscale, 28*2*drawscale))
    cv2.imwrite(r'C:\Users\makot\Documents\Final Ai thing\current_letter.png', drwng)
    print(drawing)
    imge = PhotoImage(file = r'C:\Users\makot\Documents\Final Ai thing\current_letter.png', gamma=0)
    displaydrwing = canvas.create_image(1000/2 - 28*drawscale, 600/2 -28*drawscale, anchor = NW, image = imge)
    canvas.tag_raise(displaydrwing)
    print(canvas.nametowidget(displaydrwing))
    
def Erase():
    for i in range(0,28):
        for j in range(0,28):
            drawing[i,j] = 255
            drwng = cv2.resize(drawing,(28*2*drawscale, 28*2*drawscale))
    cv2.imwrite(r'C:\Users\makot\Documents\Final Ai thing\current_letter.png', drwng)
    imge = PhotoImage(file = r'C:\Users\makot\Documents\Final Ai thing\current_letter.png')
    displaydrwing = canvas.create_image(1000/2 - 28*drawscale, 600/2 -28*drawscale, anchor = NW, image = imge)
    print(canvas.nametowidget(displaydrwing))


def OnMouseDown(event):
    mouse_pressed = True
    

def OnMouseUp(event):
    mouse_pressed = False

def Analyze():
    # model = keras.Sequential()
    # model.add(Flatten(input_shape = (28,28)))
    # model.add(Dense(128, activation= "relu"))
    # model.add(Dense(67, activation="softmax"))
    # model.load_weights(r'C:\Users\makot\Documents\AI project - Copy\model')
    # model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics= ["accuracy"])
    model = tf.keras.models.load_model(r'C:\Users\makot\Documents\Final Ai thing\my_h5_model.h5', compile = True)
    drawing2 = numP.ndarray(shape=(28,28))
    lowesti = 100
    lowestj = 100
    biggesti = -10
    biggestj = -10
    for i in range(0,28):
        for j in range(0,28):
            if drawing[i,j] == 255:
                drawing2[i,j] = 0
            else:
                drawing2[i,j] = 1
                
    #         if drawing2[i,j] == 1:
    #             if i<lowesti:
    #                 lowesti = i
    #             if j<lowestj:
    #                 lowestj = j
    #             if i>biggesti:
    #                 biggesti = i
    #             if j>biggestj:
    #                 biggestj = j

    # print(lowesti)
    # print(lowestj)
    # print(biggesti)
    # print(biggestj)

    # # scale = (int(28/(biggesti-lowesti)), int(28/(biggesti-lowesti)))
    # for i in range(0,28):
    #     for j in range(0,28):
    #         try: drawing2[i-lowesti,j-lowestj]
    #         except: continue
    #         drawing2[i-lowesti, j-lowestj] = drawing2[i,j] 
    
    # newdrawing2 = cv2.resize(drawing2, scale)
    print(drawing2)
    # print(newdrawing2)
    drwaingreshaped = drawing2.reshape(1,28,28,1)
    apredict = model(drwaingreshaped)
    value = None
    biggest = 0
    classes = numP.argmax(apredict, axis = 1)
    print(classes)
    # for i  in range(0,62):
    #      if apredict[0,i] > biggest:
    #         value = i
    #         biggest = apredict[0,i]
    # print(i)
    # print(apredict)
    result = numP.array(["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"])
    print(result[classes[0]])
    canvas.create_rectangle(480, 80 , 520,120, outline='#FFFFFF', fill='#FFFFFF', )
    canvas.create_text(500, 100, text = result[classes[0]], width=20)
# def left_click(event):
#     # x = canvas.winfo_pointerx() - canvas.winfo_rootx()
#     # y = canvas.winfo_pointery() - canvas.winfo_rooty()
#     # x = math.floor((x - 1000/2-28*drawscale))
#     # y = math.floor((y - 600/2-28*drawscale))
#     # print (x,"and", y)
    
def scalechange(num):
    if (num == 1):
        drawscale = 8
    else:
        drawscale = 4
    print(drawscale)

def buttoncallback3():
    global buttonClicked3
    buttonClicked3 = False
    buttonClicked3 = not buttonClicked3
    if (buttonClicked3==True):
       Analyze()
    buttonClicked3 = not buttonClicked3

def buttoncallback4():
    global buttonClicked4
    buttonClicked4 = False
    buttonClicked4 = not buttonClicked4
    if (buttonClicked4==True):
       Erase()
    buttonClicked4 = not buttonClicked4


def buttoncallback2():
    global buttonClicked2
    buttonClicked2 = False
    buttonClicked2 = not buttonClicked2
    if (buttonClicked2==True):
       Train()
    buttonClicked2 = not buttonClicked2

drawscale =4
canvas = Canvas(width = 1000, height = 600)  
canvas.pack(expand=YES, fill=BOTH)
buttonClicked  = False 
w = canvas.winfo_width()
h = canvas.winfo_height()
# b = Button(canvas, text="Press To Add Image", command=buttoncallback1,width=20 , height = 1).place(x=845,y=575) 
b2 = Button(canvas, text="Train", command=buttoncallback2, width=20, height=1).place(x=10, y = 575)
b3 = Button(canvas, text="Analyze", command=buttoncallback3, width=20, height=1).place(x=400, y = 575)
b3 = Button(canvas, text="Erase", command=buttoncallback4, width=20, height=1).place(x=600, y = 575)
canvas.bind("<ButtonPress-1>", OnMouseDown)
canvas.bind("<ButtonRelease-1>", OnMouseUp)
canvas.bind("<B1-Motion>", mouse_moving)
#slider1 = Scale(canvas, command= scalechange, from_=1, to=2, orient=HORIZONTAL).place(x=100,y=100)
img = canvas.create_image(500,300, anchor=CENTER, image = None)
img2 = canvas.create_rectangle(1000/2 - 28*drawscale, 600/2 - 28*drawscale,1000/2 + 28*drawscale,600/2 + 28*drawscale, outline='#000000', fill=None)

mainloop()




