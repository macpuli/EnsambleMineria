import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import linecache as lc
from sklearn.decomposition import PCA
import linecache
#Lectura del archivo
nombreArchivo=input("Introduce el nombre del archivo: ")
clases=0
try:
    clases=linecache.getline(nombreArchivo, 3)
    datos=pd.read_csv(nombreArchivo, skiprows=3, index_col=False, header=None)
except:
    print("No se encontró el archivo")
    exit()
nClases=(int(clases))
#print("\nDatos leídos:")
#print((datos))

n_atributos=datos.shape[1]-1

try:
    dimensiones=int(input("Introduce el número de PCA (máximo {}): ".format(min(n_atributos,len(datos)))))
except:
    print("Ingresa un número")
    exit()

if(dimensiones<1):
    dimensiones=1
else:
    if(dimensiones>(min(n_atributos,len(datos)))):
        dimensiones=(min(n_atributos,len(datos)))

print("Dimensiones: {}".format(dimensiones))
pca=PCA(n_components=dimensiones)
atributos=datos.loc[:,0:n_atributos].values

reduccion=pca.fit_transform(atributos)

columnas = ['comp'+str(x) for x in range(1, dimensiones+1)]
datos=datos.rename(columns={n_atributos: "Clase"})
componentesPrincipales=pd.concat([pd.DataFrame(data=reduccion,columns=columnas), datos[["Clase"]]], axis=1)

#print(componentesPrincipales)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = np.array(componentesPrincipales.drop(['Clase'], 1))
y = np.array(componentesPrincipales['Clase'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

try:
    vecinos=int(input("Introduce el número de vecinos más cercanos (se recomienda impar): "))
except:
    print("Ingrese un número")
    exit()

algoritmoKNN = KNeighborsClassifier(n_neighbors=vecinos)

from sklearn import linear_model
algoritmoPerceptron = linear_model.Perceptron()

from sklearn.naive_bayes import GaussianNB
algoritmoNaiveBayes = GaussianNB()

from sklearn.model_selection import StratifiedKFold

from statistics import mode

from sklearn.metrics import confusion_matrix
# creando pliegues
try:
    pliegues=int(input("Introduce el número de pliegues para la validación cruzada: "))
except:
    print("Ingresa un número")

if(pliegues<2):
    pligues=2
else:
    if(pliegues>10):
        dimensiones=10

skf = StratifiedKFold(n_splits=pliegues,random_state=None, shuffle=False)
# iterando entre los pliegues
precisionPromkNN = []
precisionPromPTRON = []
precisionPromBAYES = []
precisionPromEnsamble = []

column_names = ["Real", "Prediccion"]
tablaComparacion = pd.DataFrame(columns = column_names)

k=0
matrizEnsambleL=[[0] * nClases] * nClases 
matrizEnsamble=np.array(matrizEnsambleL)

import seaborn as sb
from matplotlib import pyplot as plt
import random

for train_index, test_index in skf.split(X, y):
    k+=1
    #Impresión de los índices por pliegue
    #print("TRAIN:", train_index, "TEST:", test_index); print()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #print("Clases reales del conjunto de prueba: ")
    #print(y_test)
    print("Pliegue {}:".format(k))

    algoritmoKNN.fit(X_train, y_train)
    Y_pred_KNN = algoritmoKNN.predict(X_test)
    #print("Predicciones con kNN:")
    #print(Y_pred_KNN)
    precisionKNN=algoritmoKNN.score(X_train, y_train)
    print('Precisión kNN (Vecinos más cercanos): {}'.format(precisionKNN))
    precisionPromkNN.append(precisionKNN)
    #print()

    algoritmoPerceptron.fit(X_train, y_train)
    Y_pred_PTRON = algoritmoPerceptron.predict(X_test)
    #print("Predicciones con Perceptron:")
    #print(Y_pred_PTRON)
    precisionPTRON=algoritmoPerceptron.score(X_train, y_train)
    print('Precisión Perceptron: {}'.format(precisionPTRON))
    precisionPromPTRON.append(precisionPTRON)
    #print()

    algoritmoNaiveBayes.fit(X_train, y_train)
    Y_pred_BAYES = algoritmoNaiveBayes.predict(X_test)
    #print("Predicciones con NaiveBayes:")
    #print(Y_pred_BAYES)
    precisionBAYES=algoritmoNaiveBayes.score(X_train, y_train)
    print('Precisión NaiveBayes: {}'.format(precisionBAYES))
    precisionPromBAYES.append(precisionBAYES)
    #print()
    
    pred_Ensamble=[]
    for i in range(len(y_test)):
        pred1=Y_pred_KNN[i]
        pred2=Y_pred_PTRON[i]
        pred3=Y_pred_BAYES[i]
        predicciones = [pred1, pred2, pred3]
        pred_Ensamble.insert(i,mode(predicciones))
    
    prediccionesEnsamble = np.asarray(pred_Ensamble)    

    #print("Predicciones con el ensamble:")
    #print(prediccionesEnsamble)    

    correctos=0
    for e in range(len(y_test)):
        rand1=random.uniform(-0.25, 0.25)
        tablaComparacion = tablaComparacion.append({'Real': y_test[e], 'Prediccion': prediccionesEnsamble[e]+rand1}, ignore_index=True)
        if(y_test[e]==prediccionesEnsamble[e]):
            correctos+=1

    precisionEnsamble=correctos/len(y_test)

    #print("Matriz de confusión del ensamble (en el pliegue):")
    matrizPliegue=confusion_matrix(y_test, prediccionesEnsamble)
    #print(matrizPliegue)
    for i in range(len(matrizEnsamble)):    
    # iterate through columns 
        for j in range(len(matrizEnsamble[0])): 
            matrizEnsamble[i][j] = matrizEnsamble[i][j] + matrizPliegue[i][j] 

    #print(matrizEnsamble)
    print('Precisión ensamble: {}'.format(precisionEnsamble))
    precisionPromEnsamble.append(precisionEnsamble)
    print()

print("Precisiones promedio")
print('kNN: {0: .3f}'.format(np.mean(precisionPromkNN)))
print('Perceptrón: {0: .3f}'.format(np.mean(precisionPromPTRON)))
print('NaiveBayes: {0: .3f}'.format(np.mean(precisionPromBAYES)))
print('Ensamble: {0: .3f}'.format(np.mean(precisionPromEnsamble)))

sb.stripplot(x = "Real", y = "Prediccion",data = tablaComparacion, jitter=.35, size=4, linewidth=1)
sb.despine()

plt.title("Resultados del ensamble")
#plt.grid(b=True)
plt.xlabel("Clases reales")
plt.ylabel("Clases predichas")
print("Matriz de confusión del ensamble:")
print(matrizEnsamble)
'''
size=len(matrizEnsamble)
for x in range(int(size / 2)):
    for y in range(x, size - x - 1):
        nx = size - 1 - x
        ny = size - 1 - y

        swapVal = matrizEnsamble[x][y]
        matrizEnsamble[x][y] = matrizEnsamble[y][nx]
        matrizEnsamble[y][nx] = matrizEnsamble[nx][ny]
        matrizEnsamble[nx][ny] = matrizEnsamble[ny][x]
        matrizEnsamble[ny][x] = swapVal
        '''

fig = plt.gcf()
fig.canvas.set_window_title('Gráfica de dispersión - Ensamble')

for i in range(len(matrizEnsamble)):
    for j in range(len(matrizEnsamble)):
        text = plt.text(i, j, matrizEnsamble[i, j],
                       ha="center", va="center", color="black", size=9,
                       bbox=dict(boxstyle="square",
                   ec='white',
                   fc='white',
                   pad=0.09,
                   alpha=0.75
                   )
         )
plt.show()