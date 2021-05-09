import os
import shutil
import numpy as np
import pandas as pd
from keras.preprocessing import image
from sklearn.model_selection import train_test_split


class folder:
    def __init__(self,folderImage = None ,folderTrain = None,folderTest = None ,slash = None,nameOne = None ,nameSecond = None):

        self.fileData = "training_images_file.xlsx"

        self.folderImage = folderImage
        self.folderTrain = folderTrain
        self.folderTest = folderTest
        self.slash = slash
        self.trainFolderFirst = folderTrain + slash + nameOne
        self.trainFolderSecond = folderTrain + slash + nameSecond
        self.testFolderFirst = folderTest + slash + nameOne
        self.testFolderSecond = folderTest + slash + nameSecond
        self.folderToPredict = "testing_set"
        self.extensionImage = ".jpg"

        self.recupExcel()

    def cleanFolder(self):
        try:
            # si les folder test et train existent on les effaces
            shutil.rmtree(self.folderTrain)
            shutil.rmtree(self.folderTest)
        except:
            pass
        finally:
            # création des folder test
            os.mkdir(self.folderTrain)
            os.mkdir(self.trainFolderFirst)
            os.mkdir(self.trainFolderSecond)
            os.mkdir(self.folderTest)
            os.mkdir(self.testFolderFirst)
            os.mkdir(self.testFolderSecond)


    def recupExcel(self):

        dataSet = pd.read_excel(self.fileData, encoding="utf-8-sig")
        x = dataSet.iloc[:, 0].values
        y = dataSet.iloc[:, 1].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25, random_state=0)

        # test_size => taille jeu de test
        # random_state => shuffle le jeu de donné avec seed a 0

        self.nbImageTrain = len(self.y_train)
        self.nbImageTest = len(self.y_test)


    def fillingFolderTrain(self):
        '''fonction qui permet de géré les erreus des photos qui manques et rempli les folders'''
        foldProb = []
        for trainImage in range(self.nbImageTrain):
            id = -1
            try:
                id = str(self.x_train[trainImage]) + self.extensionImage
                if self.y_train[trainImage]:
                    os.replace(self.folderImage + self.slash + id, self.trainFolderFirst + self.slash + id)
                else:
                    os.replace(self.folderImage + self.slash + id, self.trainFolderSecond + self.slash + id)
            except:
                foldProb.append(id)

    def fillingFolderTest(self):
        for testImage in range(self.nbImageTest):
            try:
                id = str(self.x_test[testImage]) + self.extensionImage
                if self.y_test[testImage]:
                    os.replace(self.folderImage + self.slash + id, self.testFolderFirst + self.slash + id)
                else:
                    os.replace(self.folderImage + self.slash + id, self.testFolderSecond + self.slash + id)
            except:
                pass


    def prediction(self,imageSize,classifier):
        dir = os.listdir(self.folderToPredict)
        listeElem = []
        for name in dir:
            testImage = image.load_img(self.folderToPredict + self.slash + name,
                                       target_size=(imageSize, imageSize))  # prend une image a prédire et la transforme
            testImage = image.img_to_array(testImage)  # transforme l'image pour qu'elle puisse rentrer dans notre ANN
            testImage = np.expand_dims(testImage,
                                       axis=0)  # rajoute une dimension ( car on a que une seule image) seule méthode magique que je ne connais pas
            y_pred = classifier.predict(testImage)
            ###attention selon le class indices il faut modifier y pred TODO

            listeElem.append([name[:-4], round(y_pred[0][0])])

        df_elem = pd.DataFrame(listeElem, columns=['image_id', 'prediction'])
        df_elem.to_csv("predictions.csv", index=False)














