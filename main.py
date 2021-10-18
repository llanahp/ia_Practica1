# ----- Si vas a ejercutar el codigo en colab    -------
# primero tendrás que ejecutar la siguiente linea
# !pip install yfinance

import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Creo una lista de '1' con tantos elementos como le pasen
def listaDeUnos(total):
    lista = []
    for i in range(total):
        lista.append(1)
    return lista


# Desplazo toda una lista, una posición hacia abajo
def desplazarColumnaAbajo(lista):
    lista.insert(0, 0)
    lista.pop(len(lista) - 1)
    return lista


'''
 -----Inicio Del programa-----
    Recolecto los datos de sp500
'''
msft = yf.Ticker("^GSPC")
DatosCrudos = pd.DataFrame(msft.history(end="2021-09-30", start="2021-01-04"))

cierre = []
rendimientos = []

# Recorro los datos en crudo para guardar los valores 'Close' y crear los rendimientos
for i in range(len(DatosCrudos.index)):
    volumen_Dia = DatosCrudos['Close'][i]
    if (i != 0):  # desde el segundo día puedo crear los rendimientos
        cierre.append(volumen_Dia)
        volumen_Dia_Anterior = DatosCrudos['Close'][i - 1]
        rendimiento = (volumen_Dia - volumen_Dia_Anterior) / volumen_Dia_Anterior
        rendimientos.append(rendimiento)

'''
Creo un DataFrame que contendrá todos los valores que necesito
'''
Matriz = pd.DataFrame()

Matriz['P(t)'] = cierre
Matriz['R(t)'] = rendimientos

unos = listaDeUnos(len(rendimientos))
Matriz['Uno'] = unos

lista_aux = desplazarColumnaAbajo(rendimientos)
Matriz['R(t-1)'] = lista_aux

lista_aux = desplazarColumnaAbajo(lista_aux)
Matriz['R(t-2)'] = lista_aux

lista_aux = desplazarColumnaAbajo(lista_aux)
Matriz['R(t-3)'] = lista_aux

lista_aux = desplazarColumnaAbajo(lista_aux)
Matriz['R(t-4)'] = lista_aux

lista_aux = desplazarColumnaAbajo(lista_aux)
Matriz['R(t-5)'] = lista_aux

# Nos falta eliminar las 5 primeras lineas ya que no contienen datos la columna R(t-5)
Matriz = Matriz.drop([0, 1, 2, 3, 4])

# Actualizo los indices
Matriz.reset_index(drop=True, inplace=True)

'''
Hacer predicciones
'''

r0 = Matriz['Uno'].values
r1 = Matriz['R(t-1)'].values
r2 = Matriz['R(t-2)'].values
r3 = Matriz['R(t-3)'].values
r4 = Matriz['R(t-4)'].values
r5 = Matriz['R(t-5)'].values

x = np.array([r0, r1, r2, r3, r4, r5]).T
y = np.array(Matriz['R(t)'].values)
'''
1. De cuantos parámetros consta el modelo? (1 punto)
    b, de 6  parámetros ya que ...
'''

reg = LinearRegression()
reg = reg.fit(x, y)
Y_pred = reg.predict(x)
error = np.sqrt(mean_squared_error(y,Y_pred))

print("El primer y el sexto regresor no son iguales, (",Y_pred[0],"!=",Y_pred[5],")")
'''
2. Los parámetros correspondientes al primero y sexto regresor son iguales? (1 punto)
    No son iguales porque ...
'''

#creo un DataFrame de dos columnas con las prediciones y los datos reales
matriz2 = pd.DataFrame()
matriz2['Rendimiento_Real'] = Matriz['R(t)']
matriz2['Rendimiento_Predicho'] = Y_pred

mse = 0
for i in range(181):
    valor = matriz2['Rendimiento_Real'][i] - matriz2['Rendimiento_Predicho'][i]
    mse = mse + pow(valor, 2)

mse = (1 / 181) * mse
print("mse: ", mse)

mae = 0

for i in range(181):
    valor = matriz2['Rendimiento_Real'][i] - matriz2['Rendimiento_Predicho'][i]
    mae = mae + abs(valor)

mae = mae/181
print("mae: ", mae)



mape = 0
for i in range(181):
    valor = matriz2['Rendimiento_Real'][i] - matriz2['Rendimiento_Predicho'][i]
    mape = mape + (abs(valor / matriz2['Rendimiento_Real'][i]))

mape = mape / 181
print("mape: ", mape)
'''
3. Las tres métricas, proporcionan los mismos resultados (el mismo valor)? (2 puntos)
    No proporcionan el mismo valor ya que en cada uno de los casos está calculando una cosa distinta....
'''

mape2 = 0
for i in range(181):
    valor = matriz2['Rendimiento_Real'][i] - 0
    mape2 = mape2 + (abs(valor / matriz2['Rendimiento_Real'][i]))

mape2 = mape2 / 181
print("mape2: ", mape2)

print("mi modelo es un ",(np.sum(mape)/np.sum(mape2)),"% mejor que el de rt=0")

'''
4. Suponga que comparamos las predicciones frente a un paseo aleatorio rt =0
empleando el mape, ¿las predicciones del modelo lineal son mejores? (2 puntos)
    Si, son mejores ...
'''



#Ventana deslizante
x_30 = np.delete(x, slice(30,181,1), axis=0)
y_30 = np.delete(y,slice(30,181,1), axis=0)
prediccion_ventana =np
rentabilidad_total=0
for i in range(30,182):
    reg = reg.fit(x_30, y_30)
    if(i<180):
        prediccion_dia_siguiente=reg.predict(np.array([[Matriz['Uno'][i+1],Matriz['R(t-1)'][i+1],Matriz['R(t-2)'][i+1],Matriz['R(t-3)'][i+1],Matriz['R(t-4)'][i+1],Matriz['R(t-5)'][i+1]]]))
        prediccion_ventana = np.append(prediccion_ventana,prediccion_dia_siguiente)
        rentabilidad_total= rentabilidad_total + (prediccion_dia_siguiente-Matriz['R(t)'][i])
    if(i!=181):# no tengo que actualizar la ventana deslizante en la ultima iteracion
        x_30 = np.delete(x_30,0,axis=0)
        y_30 = np.delete(y_30,0,axis=0)
        x_30 = np.append(x_30, [[Matriz['Uno'][i],Matriz['R(t-1)'][i],Matriz['R(t-2)'][i],Matriz['R(t-3)'][i],Matriz['R(t-4)'][i],Matriz['R(t-5)'][i]]], axis=0)
        y_30 = np.append(y_30, [y[i]], axis=0)



print("El valor del tercer regresor utilizando una ventana deslizante (",prediccion_ventana[2],") varia respecto a si no utilizamos dicha ventana (",Y_pred[2],")")
'''
5. ¿El valor del parámetro correspondiente al tercer regresor varia? (2 puntos)
    Si que varia el tercer regresor ...
'''

print("Rentabilidad total: ",rentabilidad_total)
'''
6. es dicha rentabilidad positiva? (2 puntos)
    si ....
'''