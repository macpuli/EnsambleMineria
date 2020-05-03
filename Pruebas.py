import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import linecache as lc
from sklearn.decomposition import PCA

#Lectura del archivo
nombreArchivo=input("Introduce el nombre del archivo: ")
datos=pd.read_csv(nombreArchivo, skiprows=3, index_col=False, header=None)
print("\nDatos leídos:\n")
print(datos)

n_atributos=datos.shape[1]-1

dimensiones=int(input("Introduce el número de PCA (máximo {}): ".format(n_atributos)))
pca=PCA(n_components=dimensiones)
atributos=datos.loc[:,0:n_atributos].values

reduccion=pca.fit_transform(atributos)

columnas = ['comp'+str(x) for x in range(1, dimensiones+1)]
datos=datos.rename(columns={n_atributos: "Clase"})
componentesPrincipales=pd.concat([pd.DataFrame(data=reduccion,columns=columnas), datos[["Clase"]]], axis=1)

print(componentesPrincipales)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = np.array(componentesPrincipales.drop(['Clase'], 1))
y = np.array(componentesPrincipales['Clase'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
print(' {} datos para entrenamiento\n {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

print("Clases reales del conjunto de prueba: ")
print(y_test)
print()

vecinos=int(input("Introduce el número de vecinos más cercanos (se recomienda impar): "))
algoritmoKNN = KNeighborsClassifier(n_neighbors=5)
algoritmoKNN.fit(X_train, y_train)
Y_pred_KNN = algoritmoKNN.predict(X_test)
print("Predicciones con kNN:")
print(Y_pred_KNN)
precisionKNN=algoritmoKNN.score(X_train, y_train)
print('Precisión kNN (Vecinos más cercanos): {}'.format(precisionKNN))
print()

from sklearn import linear_model
algoritmoPerceptron = linear_model.Perceptron()
algoritmoPerceptron.fit(X_train, y_train)
Y_pred_PTRON = algoritmoPerceptron.predict(X_test)
print("Predicciones con Perceptron:")
print(Y_pred_PTRON)
precisionPTRON=algoritmoPerceptron.score(X_train, y_train)
print('Precisión Perceptron: {}'.format(precisionPTRON))
print()

from sklearn.naive_bayes import GaussianNB
algoritmoNaiveBayes = GaussianNB()
algoritmoNaiveBayes.fit(X_train, y_train)
Y_pred_BAYES = algoritmoNaiveBayes.predict(X_test)
print("Predicciones con NaiveBayes:")
print(Y_pred_BAYES)
precisionBAYES=algoritmoNaiveBayes.score(X_train, y_train)
print('Precisión NaiveBayes: {}'.format(precisionBAYES))
print()

from statistics import mode

pred_Ensamble=[]
for i in range(len(y_test)):
    pred1=Y_pred_KNN[i]
    pred2=Y_pred_PTRON[i]
    pred3=Y_pred_BAYES[i]
    predicciones = [pred1, pred2, pred3]
    pred_Ensamble.insert(i,mode(predicciones))
    '''
    if(pred1==pred2 & pred1==pred3):
        pred_Ensamble.insert(i,pred1)
    else:
        if((pred1==pred2)|(pred1==pred3)):
            pred_Ensamble.insert(i,pred1)
        else:
            if((pred2==pred3)):
                pred_Ensamble.insert(i,pred2)
            else:
                if(pred1!=pred2!=pred3):
                    masPreciso=max(precisionKNN,precisionPTRON,precisionBAYES)
                    if(masPreciso==precisionKNN):
                        pred_Ensamble.insert(i,pred1)
                    else:
                        if(masPreciso==precisionPTRON):
                            pred_Ensamble.insert(i, pred2)
                        else:
                            if(masPreciso==precisionBAYES):
                                pred_Ensamble.insert(i, pred3)    
    '''
    
prediccionesEnsamble = np.asarray(pred_Ensamble)    

print("Predicciones con el ensamble:")
print(prediccionesEnsamble)    

correctos=0
for e in range(len(y_test)):
    if(y_test[e]==prediccionesEnsamble[e]):
        correctos+=1

precisionEnsamble=correctos/len(y_test)

from sklearn.metrics import confusion_matrix
print("Matriz de confusión del ensamble:")
print(confusion_matrix(y_test, prediccionesEnsamble))
print('Precisión ensamble: {}'.format(precisionEnsamble))
print()
