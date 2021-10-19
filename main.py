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

'''calculamos los parametros mediante la ecuación normal parametros = (X^t * X)^(-1) * X^t * Y'''
x = np.array([r0, r1, r2, r3, r4, r5]).T     #X
xt = np.array([r0, r1, r2, r3, r4, r5])      #X^t
xi = np.linalg.inv(np.dot(xt, x))            #(X^t * X)^(-1)
y = np.array(Matriz['R(t)'].values)          #Y


parametros = np.dot(np.dot(xi, xt), y)       #(X^t * X)^(-1)  * X^t * Y
print(parametros)

'''
1. De cuantos parámetros consta el modelo? (1 punto)
    b, de 6  parámetros ya que ...
'''

Y_pred = []
for i in range(len(x)):
    Y_pred.append(parametros[0] + parametros[1]*x[i][1] + parametros[2]*x[i][2] + parametros[3]*x[i][3] + parametros[4]*x[i][4])


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

prediccion_ventana =np
rentabilidad_total=0
contador=30
for i in range(30,182):
    x_30 = np.array([r0[contador-30:contador], r1[contador-30:contador], r2[contador-30:contador], r3[contador-30:contador], r4[contador-30:contador], r5[contador-30:contador]]).T  # X
    xt_30 = np.array([r0[contador-30:contador], r1[contador-30:contador], r2[contador-30:contador], r3[contador-30:contador], r4[contador-30:contador], r5[contador-30:contador]])     #X^t
    xi_30 = np.linalg.inv(np.dot(xt_30, x_30))  # (X^t * X)^(-1)
    y_30 = np.array(Matriz['R(t)'].values[contador-30:contador])
    parametros = np.dot(np.dot(xi_30, xt_30), y_30)
    if(i<180):
        prediccion_dia_siguiente=(parametros[0] + parametros[1]*x[1][1] + parametros[2]*x[1][2] + parametros[3]*x[1][3] + parametros[4]*x[1][4])
        prediccion_ventana = np.append(prediccion_ventana,prediccion_dia_siguiente)
        rentabilidad_total= rentabilidad_total + (prediccion_dia_siguiente-Matriz['R(t)'][i])
    contador=contador+1





print("El valor del tercer regresor utilizando una ventana deslizante (",prediccion_ventana[2],") varia respecto a si no utilizamos dicha ventana (",Y_pred[2],")")
'''
5. ¿El valor del parámetro correspondiente al tercer regresor varia? (2 puntos)
    Si que varia el tercer regresor ...
'''

print("Rentabilidad total: ",rentabilidad_total)
'''
6. es dicha rentabilidad positiva? (2 puntos)
    Si ....
'''