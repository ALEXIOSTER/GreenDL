import sys
import tensorflow as tf
import Callback
from tensorflow.keras.layers import Convolution2D  # traite les image ( vu qu'il sont en 2d)
from tensorflow.keras.layers import MaxPooling2D  # pour la récupération d'information importante
from tensorflow.keras.layers import Flatten  # feature map pour les applatir
from tensorflow.keras.layers import Dense  # ajout de couche
from tensorflow.keras.layers import Dropout  # probabilité pour "endormir" les neurone
from tensorflow.keras.models import Sequential  # initialise le réseau de neuronne

from keras.preprocessing.image import ImageDataGenerator
import math
import os


callback = Callback.CustomCallback()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#enlève les print de tensorflow, ne garde que les messages d'error


def traitementDir(dir):
    """gère le probleme de /, sert a unifier pour facilité le traitement"""
    if dir[-1] == "/":
        tmp = dir[:-1]
    else:
        tmp = dir
    return tmp


dir = traitementDir(sys.argv[1])  # le directory des photos

path2write = sys.argv[2]

nbrHiddenLayer = 3
neuroneCoucheCacher = 128
tailleLot = 32
nbrPoolConv = 4
fctActivation = 'relu'
optimizer = 'adam'
imageSize = 150
rateDropOut = 0.3
shearRange = 0.2
zoomRange = 0.2
activationConvolution = 'relu'




if len(sys.argv) > 3:
    nbEpoque = int(sys.argv[3])  # nombre de fois ou on fait l'entrainement
else:
    nbEpoque = 15


if len(sys.argv) > 4:
    nbFiltre = int(sys.argv[4])
else:
    nbFiltre = 32


if len(sys.argv) > 5:
    kernelSize = int(sys.argv[5])
else:
    kernelSize = 3


if len(sys.argv) > 6:
    stridesConv = int(sys.argv[6])
else:
    stridesConv = 1


if len(sys.argv) > 7:
    stridesPool = int(sys.argv[7])
else:
    stridesPool = 2

if len(sys.argv) > 8:
    xPool = int(sys.argv[8])
else:
    xPool = 2

if len(sys.argv) > 9:
    yPool = int(sys.argv[9])
else:
    yPool = 2


allMetrics = ["acc"]
sizeMatricePool = (xPool, yPool)
folderTrain = dir + "/" + "training_set"
folderTest = dir + "/" + "test_set"

outUnit = 1
fctLoss = "binary_crossentropy"
classMode = 'binary'

def comptageTot(folderPathTrain, folderPathTest):
    """ compte le nombre total de photo """
    imageTest = 0
    imageTrain = 0
    foldersTest = os.listdir(folderPathTest)
    foldersTrain = os.listdir(folderPathTrain)
    for subFolders in foldersTrain:
        imageTrain += comptageImage(folderPathTrain + "/" + subFolders)
    for subFolders in foldersTest:
        imageTest += comptageImage(folderPathTest + "/" + subFolders)
    return (imageTrain, imageTest)


def comptageImage(pathFolder):
    folder = os.listdir(pathFolder)
    return len(folder)


nbImageTrain, nbImageTest = comptageTot(folderTrain, folderTest)


def buildModel():
    model = Sequential()

    # phase traitement image
    model.add(Convolution2D(filters=nbFiltre, kernel_size=kernelSize, strides=stridesConv, input_shape=(imageSize, imageSize, 3),
                            activation=activationConvolution))
    # Phase de convolution , on transforme l'image avec la fonction de convolution grace a des filtre pour avoir un set de featurs map
    model.add(MaxPooling2D(pool_size=sizeMatricePool, strides=stridesPool))
    # Phase de Pooling ( avec le max pooling ) , permet de réduire la taille en gardant l'information la plus importante ( pour les différente prise de vue)

    nbrLayer = 1
    multiplicateur = 1
    for i in range(nbrPoolConv - 1):
        model.add(Convolution2D(filters=nbFiltre * multiplicateur, kernel_size=kernelSize, strides=stridesConv,
                                activation=activationConvolution))
        model.add(MaxPooling2D(pool_size=sizeMatricePool, strides=stridesPool))
        nbrLayer += 1
        multiplicateur = math.floor((nbrLayer / 2) + 1)

    # phase feed forward
    model.add(Flatten())  # Couche d'entrée
    # Etape de flattening ( linéarise la matrice de pooling feature map pour le réseau de ANN)

    for j in range(nbrHiddenLayer):
        model.add(Dense(units=neuroneCoucheCacher, activation=fctActivation))

    ##Couche de sortie
    model.add(Dropout(rateDropOut))
    model.add(Dense(units=outUnit, activation="sigmoid"))

    model.compile(optimizer=optimizer, loss=fctLoss, metrics=allMetrics)

    return model


# pool_size = dimenssion de la matrice
# filtres = le nombre de featurs detector qu'on souhaite (on multiplie par 2 a chaque fois qu'on rajoute une couche)
# kernel size = la taille de la matrice des filtres
# strides = le déplacement utiliser lors de la fonction de convolution
# input_shape = formate ,pour avoir la meme dimension a chaque image ( H,L) ,dernier agrument 2 = noir et blanc , 3= RVB
# activation= pour la fonction d'activation ( relu permet d'enlever la linéarité)
# dense = rajoute une couche de neurone
# units => nombre de neuronne dans cette couche
# optimizer => algorithme du gradient
# loss => fonction de cout

CNN = buildModel()

###Configuration des  nouvelles images pour le train
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=shearRange,
    zoom_range=zoomRange,
    horizontal_flip=True,
)

# rescale => modifie les valeur des couleur nos pixelle entre 0 et 1
# shear_range => transvection (change l'angle de l'image)
# zoom_range => fait des zooms
# horizontal_flip => regarde dans un miroire

###Mise a l'échelle du train
test_datagen = ImageDataGenerator(rescale=1. / 255)

trainSet = train_datagen.flow_from_directory(
    folderTrain,
    target_size=(imageSize, imageSize),
    batch_size=tailleLot,
    class_mode=classMode)

# target_size => taille de nos image
# batch_size => le lot d'observation avant de mettre a jour les poids
# class_mode => couche de sortie binaire ou de catégorie


testSet = test_datagen.flow_from_directory(
    folderTest,
    target_size=(imageSize, imageSize),
    batch_size=tailleLot,
    class_mode=classMode)

###Entraine le modèle et évalue sa performance

saves = CNN.fit(
    trainSet,
    steps_per_epoch=math.ceil(nbImageTrain / tailleLot),
    epochs=nbEpoque,
    validation_data=testSet,
    validation_steps=math.ceil(nbImageTest / tailleLot),
    verbose=1,
    callbacks=[callback])

# steps_per_epoch => le nombre de réajustement des poids (plafond(nb image / tailleLot))
# epochs => nombre d'époque pour l'entrainement
# validation_steps => idem que steps_per_epoch mais pour le jeu de test (plafond(nb image / tailleLot))
# verbose = 0 => désactive le print de l'entrainement
# verbose = 2 => print la fin de l'époque




def writeIntoFile(lstValTest, lstValTrain, path2write):
    f = open(path2write, "a")
    toWrite = "Epoch,ACC_Test,ACC_Train,Diff_ACC,ellapsedTime \n"
    f.write(toWrite)
    for i in range(len(lstValTest)):
        precisionTest = lstValTest[i]
        precisionEntrainement = lstValTrain[i]
        toWrite = str(i+1) +","+str(round(precisionTest,4))+ ","+str(round(precisionEntrainement,4))+ "," + str(round(abs(precisionTest-precisionEntrainement),4)) + "," + str(round(callback.times[i][1],2)) + "s\n"
        f.write(toWrite)
    f.close()

writeIntoFile(saves.history['val_acc'],saves.history['acc'],path2write)