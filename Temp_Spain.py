import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import zscore
# leer el file
df = pd.read_excel("Route",sheet_name=None)
# Data wrangling
temp_Spain = df["avg_tem_Spain"]
temp_Spain= temp_Spain.rename(columns={'Avg. Temp (°C) Spain': 'Avg. Temp (°C)'})
temp_Spain['Month'] = pd.to_datetime(temp_Spain['Month'], format='%Y-%m')
temp_Madrid= temp_Spain.rename(columns={'Area': 'Area'})

temp_Madrid = df["avg_tem_Madrid"]
temp_Madrid= temp_Madrid.rename(columns={"Avg. Temp (°C) Madrid":"Avg. Temp (°C)" })
temp_Madrid= temp_Madrid.rename(columns={'Area ': 'Area'})
temp_Madrid['Month'] = pd.to_datetime(temp_Madrid['Month'],format='%Y-%m')

temp = pd.concat([temp_Spain, temp_Madrid], ignore_index=True)

#Change format of datetime
temp['Month'] = temp["Month"].dt.strftime('%Y-%m')
#print(temp.head())

# Convertir a frecuencias
#temp_frec_spain = temp[temp['Area'] == 'Spain']['Avg. Temp (°C)'].value_counts().sort_index()
#temp_frec_madrid = temp[temp['Area'] == 'Madrid']['Avg. Temp (°C)'].value_counts().sort_index()


temp_frec_spain = temp[temp['Area'] == 'Spain']['Avg. Temp (°C)']
temp_frec_madrid = temp[temp['Area'] == 'Madrid']['Avg. Temp (°C)']

#Grafica ambas distribuciones de temepratura en un plot
bins = np.linspace(0, 30, 20)
frec_spain, _ = np.histogram(temp_frec_spain, bins=bins)
frec_madrid, _ = np.histogram(temp_frec_madrid, bins=bins)

# Calcular la media y la desviación estándar de cada conjunto de datos
mean1, std1 = np.mean(temp_frec_spain), np.std(temp_frec_spain)
mean2, std2 = np.mean(temp_frec_madrid), np.std(temp_frec_madrid)

plt.bar(bins[:-1], frec_spain, alpha=0.5, color="none", edgecolor='black', label='Spain', width=1.5)
plt.bar(bins[:-1]+0.1, frec_madrid, alpha=0.5, color='none', edgecolor='red', label='Madrid', width=1.5)


# Agregar línea vertical y punto para la media y la desviación estándar
plt.axvline(mean1, color='black', linestyle='--', label='Media Spain')
plt.axvline(mean2, color='grey', linestyle='--', label='Media Madrid')
plt.scatter(mean1, 0, color='black', marker='o', s=50)
plt.scatter(mean2, 0, color='grey', marker='o', s=50)

plt.axvline(mean1-std1, color='black', linestyle='-.', label='Stan Dev Spain')
plt.axvline(mean1 + std1, color='black', linestyle='-.', label='_nolegend_')
plt.axvline(mean2-std2, color='grey', linestyle='-.', label='Stan Dev Madrid')
plt.axvline(mean2 + std2, color='grey', linestyle='-.', label='_nolegend_')


# Agregar título y etiquetas de los ejes
plt.title('Temperature Frecuency Spain & Madrid 2010 - 2020 ')
plt.xlabel('Monthly Average Temperature (C°)')
plt.ylabel('Frecuency')

# Mostrar el gráfico
plt.show()

#estadisticas descriptivas MTC, dispersion y Posicion 
print(temp_frec_spain.describe())
print(temp_frec_madrid.describe())

media_spain = temp_frec_spain.mean()
print("La media es:", media_spain )

mediana_spain = temp_frec_spain.median()
print("La mediana es:", mediana_spain )

moda_spain = temp_frec_spain.mode()[0]
print("La moda es:", moda_spain )

std_spain = temp_frec_spain.std()
print("La DS es:", std_spain )

media_madrid = temp_frec_madrid.mean()
print("La media es:", media_madrid)

mediana_madrid = temp_frec_madrid.median()
print("La mediana es:", mediana_madrid)

moda_madrid = temp_frec_madrid.mode()[0]
print("La moda es:", moda_madrid)

std_madrid = temp_frec_madrid.std()
print("La DS es:", std_madrid )

#desviacion estandar muestral n-1
print(np.std(temp_frec_spain, ddof=1))
print(np.std(temp_frec_madrid, ddof=1))


#La media de la temperatura de España 16.08 es mientras que la de Madrid es 15.52  
#Las desviaciones estandar poblacionales son 6.03 y 6.66 respectivamente
#Las desviaciones estandar muestrales son las mismas poblacionales
#Para España la mediana es: 15.55 y la moda es: 8.0
#Para Madrid la mediana es: 14.5 y la moda es: 25.7

q1, q3 = np.percentile(temp_frec_spain, [25, 75])
iqr_spain = q3 - q1
print("Primer cuartil Spain: ", q1)
print("Tercer cuartil: ", q3)
#Rango intercuartilico es la diferencia entre el 3er cuartil y el 1ro
print("Rango intercuartílico: ", iqr_spain)

q1, q3 = np.percentile(temp_frec_madrid, [25, 75])
iqr_madrid = q3 - q1

print("Primer cuartil Madrid: ", q1)
print("Tercer cuartil: ", q3)
print("Rango intercuartílico: ", iqr_spain)


#COVERAGE INTERVALS
n_spain = len(temp[temp['Area'] == 'Spain'])
ci_spain = t.interval(0.95, n_spain - 1, loc=media_spain, scale=std_spain / (n_spain ** 0.5))
print(ci_spain)


n_madrid = len(temp[temp['Area'] == 'Madrid'])
ci_madrid = t.interval(0.95, n_madrid - 1, loc=media_madrid, scale=std_madrid / (n_madrid ** 0.5))
print(ci_madrid)
"""
si usamos un nivel de confianza del 95%, el intervalo de cobertura para España puede ser 
(15,04, 17,12), lo que significa que podemos estar seguros al 95% de que la temperatura 
media real de la población en España se encuentra entre 15,04 °C y 17,12 °C. 
"""

temp_frec_spain = temp[temp['Area'] == 'Spain']['Avg. Temp (°C)']
temp_frec_madrid = temp[temp['Area'] == 'Madrid']['Avg. Temp (°C)']

data = [temp_frec_spain, temp_frec_madrid]
labels = ['Spain', 'Madrid']

#Boxplot de frecuencias promedio mes de temperatura import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import zscore

df = pd.read_excel("C:/Users/camil/Documents/COURSES/Diplomado Data Sience/Data Sciences/Kaggle/Temp_Spain/Temperature_Spain_vs_Madrid.xlsx",sheet_name=None)

temp_Spain = df["avg_tem_Spain"]
temp_Spain= temp_Spain.rename(columns={'Avg. Temp (°C) Spain': 'Avg. Temp (°C)'})
temp_Spain['Month'] = pd.to_datetime(temp_Spain['Month'], format='%Y-%m')
temp_Madrid= temp_Spain.rename(columns={'Area': 'Area'})

temp_Madrid = df["avg_tem_Madrid"]
temp_Madrid= temp_Madrid.rename(columns={"Avg. Temp (°C) Madrid":"Avg. Temp (°C)" })
temp_Madrid= temp_Madrid.rename(columns={'Area ': 'Area'})
temp_Madrid['Month'] = pd.to_datetime(temp_Madrid['Month'],format='%Y-%m')

temp = pd.concat([temp_Spain, temp_Madrid], ignore_index=True)

#Change format of datetime
temp['Month'] = temp["Month"].dt.strftime('%Y-%m')
#print(temp.head())

# Convertir a frecuencias
#temp_frec_spain = temp[temp['Area'] == 'Spain']['Avg. Temp (°C)'].value_counts().sort_index()
#temp_frec_madrid = temp[temp['Area'] == 'Madrid']['Avg. Temp (°C)'].value_counts().sort_index()


temp_frec_spain = temp[temp['Area'] == 'Spain']['Avg. Temp (°C)']
temp_frec_madrid = temp[temp['Area'] == 'Madrid']['Avg. Temp (°C)']



#Grafca ambas distribuciones de temepratura en un plot
bins = np.linspace(0, 30, 20)
frec_spain, _ = np.histogram(temp_frec_spain, bins=bins)
frec_madrid, _ = np.histogram(temp_frec_madrid, bins=bins)

# Calcular la media y la desviación estándar de cada conjunto de datos
mean1, std1 = np.mean(temp_frec_spain), np.std(temp_frec_spain)
mean2, std2 = np.mean(temp_frec_madrid), np.std(temp_frec_madrid)

plt.bar(bins[:-1], frec_spain, alpha=0.5, color="none", edgecolor='black', label='Spain', width=1.5)
plt.bar(bins[:-1]+0.1, frec_madrid, alpha=0.5, color='none', edgecolor='red', label='Madrid', width=1.5)


# Agregar línea vertical y punto para la media y la desviación estándar
plt.axvline(mean1, color='black', linestyle='--', label='Media Spain')
plt.axvline(mean2, color='grey', linestyle='--', label='Media Madrid')
plt.scatter(mean1, 0, color='black', marker='o', s=50)
plt.scatter(mean2, 0, color='grey', marker='o', s=50)

plt.axvline(mean1-std1, color='black', linestyle='-.', label='Stan Dev Spain')
plt.axvline(mean1 + std1, color='black', linestyle='-.', label='_nolegend_')
plt.axvline(mean2-std2, color='grey', linestyle='-.', label='Stan Dev Madrid')
plt.axvline(mean2 + std2, color='grey', linestyle='-.', label='_nolegend_')


# Agregar título y etiquetas de los ejes
plt.title('Temperature Frecuency Spain & Madrid 2010 - 2020 ')
plt.xlabel('Monthly Average Temperature (C°)')
plt.ylabel('Frecuency')

# Agregar leyenda
plt.legend()

# Mostrar el gráfico
plt.show()


print(temp_frec_spain.describe())
print(temp_frec_madrid.describe())

media_spain = temp_frec_spain.mean()
print("La media es:", media_spain )

mediana_spain = temp_frec_spain.median()
print("La mediana es:", mediana_spain )

moda_spain = temp_frec_spain.mode()[0]
print("La moda es:", moda_spain )

std_spain = temp_frec_spain.std()
print("La DS es:", std_spain )

media_madrid = temp_frec_madrid.mean()
print("La media es:", media_madrid)

mediana_madrid = temp_frec_madrid.median()
print("La mediana es:", mediana_madrid)

moda_madrid = temp_frec_madrid.mode()[0]
print("La moda es:", moda_madrid)

std_madrid = temp_frec_madrid.std()
print("La DS es:", std_madrid )

#desviacion estandar muestral n-1
print(np.std(temp_frec_spain, ddof=1))
print(np.std(temp_frec_madrid, ddof=1))


#La media de la temperatura de España 16.08 es mientras que la de Madrid es 15.52  
#Las desviaciones estandar poblacionales son 6.03 y 6.66 respectivamente
#Las desviaciones estandar muestrales son las mismas poblacionales
#Para España la mediana es: 15.55 y la moda es: 8.0
#Para Madrid la mediana es: 14.5 y la moda es: 25.7

q1, q3 = np.percentile(temp_frec_spain, [25, 75])
iqr_spain = q3 - q1
print("Primer cuartil Spain: ", q1)
print("Tercer cuartil: ", q3)
#Rango intercuartilico es la diferencia entre el 3er cuartil y el 1ro
print("Rango intercuartílico: ", iqr_spain)

q1, q3 = np.percentile(temp_frec_madrid, [25, 75])
iqr_madrid = q3 - q1

print("Primer cuartil Madrid: ", q1)
print("Tercer cuartil: ", q3)
print("Rango intercuartílico: ", iqr_spain)


#COVERAGE INTERVALS
n_spain = len(temp[temp['Area'] == 'Spain'])
ci_spain = t.interval(0.95, n_spain - 1, loc=media_spain, scale=std_spain / (n_spain ** 0.5))
print(ci_spain)


n_madrid = len(temp[temp['Area'] == 'Madrid'])
ci_madrid = t.interval(0.95, n_madrid - 1, loc=media_madrid, scale=std_madrid / (n_madrid ** 0.5))
print(ci_madrid)
"""
si usamos un nivel de confianza del 95%, el intervalo de cobertura para España puede ser 
(15,04, 17,12), lo que significa que podemos estar seguros al 95% de que la temperatura 
media real de la población en España se encuentra entre 15,04 °C y 17,12 °C. 
"""

temp_frec_spain = temp[temp['Area'] == 'Spain']['Avg. Temp (°C)']
temp_frec_madrid = temp[temp['Area'] == 'Madrid']['Avg. Temp (°C)']

data = [temp_frec_spain, temp_frec_madrid]
labels = ['Spain', 'Madrid']
#Boxplot temepratura promedio mensual España vs Madrid
fig, ax = plt.subplots()
ax.boxplot(data)
ax.set_xticklabels(labels)
ax.set_ylabel('Average Temperature (°C)')
plt.title('Montly Average Temperatures Spain vs Madrid')
plt.show()

#Scatterplot rela
#plt.scatter(temp_frec_spain, temp_frec_madrid, c=['red', 'blue', 'green', 'purple', 'orange'])

plt.scatter(temp_frec_spain, temp_frec_spain, color='grey', label='Spain')
plt.scatter(temp_frec_spain, temp_frec_madrid, color='red', label='Madrid')
plt.xlabel('Average Temperature in Spain (°C)')
plt.ylabel('Average Temperature in Madrid (°C)')
plt.title('Montly Average Temperatures')
plt.show()

temp_Spain['z-score_spain'] = zscore(temp['Avg. Temp (°C)'])
temp_Madrid['z-score_madrid'] = zscore(temp['Avg. Temp (°C)'])

spain_zscore = (10.5 - temp_Spain['Avg. Temp (°C)'].mean()) / temp_Spain['Avg. Temp (°C)'].std()
madrid_zscore = (5.5 - temp_Madrid['Avg. Temp (°C)'].mean()) / temp_Madrid['Avg. Temp (°C)'].std()

print(spain_zscore)
print(madrid_zscore)

# Comparar los z-scores y determinar qué temperatura es más extrema con su valor absoluto(abs)
if abs(spain_zscore) > abs(madrid_zscore):
    print("La temperatura más extrema es la de Spain.")
else:
    print("La temperatura más extrema es la de Madrid.")


temp_Spain['z-score_spain'] = zscore(temp['Avg. Temp (°C)'])
temp_Madrid['z-score_madrid'] = zscore(temp['Avg. Temp (°C)'])

spain_zscore = (10.5 - temp_Spain['Avg. Temp (°C)'].mean()) / temp_Spain['Avg. Temp (°C)'].std()
madrid_zscore = (5.5 - temp_Madrid['Avg. Temp (°C)'].mean()) / temp_Madrid['Avg. Temp (°C)'].std()

print(spain_zscore)
print(madrid_zscore)

# Comparar los z-scores y determinar qué temperatura es más extrema con su valor absoluto(abs)
if abs(spain_zscore) > abs(madrid_zscore):
    print("La temperatura más extrema es la de Spain.")
else:
    print("La temperatura más extrema es la de Madrid.")

