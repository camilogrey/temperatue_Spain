import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import axes_style
import re
from seaborn import load_dataset
from scipy.stats import t
from scipy.stats import zscore

df = pd.read_excel("C:/Users/camil/Documents/COURSES/Diplomado Data Sience/Data Sciences/Kaggle/temp_Spain/Temperature_Spain_vs_Madrid.xlsx",sheet_name=None)

temp_Sevilla = df["avg_tem_Sevilla"]
temp_Sevilla['Month'] = pd.to_datetime(temp_Sevilla['Month'], format='%Y-%m')
temp_Sevilla= temp_Sevilla.rename(columns={'Area ': 'Area'})
#Ojo debe tener el mimso encabezado ambos conjuntos de datos para poder concatenarlos
temp_Sevilla= temp_Sevilla.rename(columns={'Avg. Temp (°C) Sevilla': 'Avg. Temp (°C)'})

temp_Madrid = df["avg_tem_Madrid"]
temp_Madrid['Month'] = pd.to_datetime(temp_Madrid['Month'],format='%Y-%m')
temp_Madrid= temp_Madrid.rename(columns={'Area ': 'Area'})
temp_Madrid= temp_Madrid.rename(columns={"Avg. Temp (°C) Madrid":"Avg. Temp (°C)" })

temp = pd.concat([temp_Sevilla, temp_Madrid], ignore_index=True)

#Change format of datetime
temp['Month'] = temp["Month"].dt.strftime('%Y-%m')

# Convertir a frecuencias, conteo de las temperaturas que se repiten
#temp_frec_sevilla = temp[temp['Area'] == 'Spain']['Avg. Temp (°C)'].value_counts().sort_index()
#temp_frec_madrid = temp[temp['Area'] == 'Madrid']['Avg. Temp (°C)'].value_counts().sort_index()

temp_frec_sevilla = temp[temp['Area'] == 'Sevilla']['Avg. Temp (°C)']
temp_frec_madrid = temp[temp['Area'] == 'Madrid']['Avg. Temp (°C)']


# Graficar ambas distribuciones de temperatura en un plot
"""
bins = np.linspace(0, 35, 20)
frec_sevilla, _ = np.histogram(temp_frec_sevilla, bins=bins)
frec_madrid, _ = np.histogram(temp_frec_madrid, bins=bins)

# Calcular la media y la desviación estándar de cada conjunto de datos
mean1, std1 = np.mean(temp_frec_sevilla), np.std(temp_frec_sevilla)
mean2, std2 = np.mean(temp_frec_madrid), np.std(temp_frec_madrid)

plt.bar(bins[:-1], frec_sevilla, alpha=0.5, color="none", edgecolor='black', label='Sevilla', width=1.5)
plt.bar(bins[:-1]+0.1, frec_madrid, alpha=0.5, color='none', edgecolor='red', label='Madrid', width=1.5)

# Agregar línea vertical y punto para la media y la desviación estándar
plt.axvline(mean1, color='black', linestyle='--', label='Media Sevilla')
plt.axvline(mean2, color='red', linestyle='--', label='Media Madrid')
plt.scatter(mean1, 0, color='black', marker='o', s=50)
plt.scatter(mean2, 0, color='grey', marker='o', s=50)

plt.axvline(mean1-std1, color='black', linestyle='-.', label='Stan Dev Sevilla')
plt.axvline(mean1 + std1, color='black', linestyle='-.', label='_nolegend_')
plt.axvline(mean2-std2, color='grey', linestyle='-.', label='Stan Dev Madrid')
plt.axvline(mean2 + std2, color='grey', linestyle='-.', label='_nolegend_')

# Agregar título y etiquetas de los ejes
plt.title('Temperature Frequency Sevilla & Madrid 2010 - 2020')
plt.xlabel('Monthly Average Temperature (C°)')
plt.ylabel('Frequency')

# Agregar leyenda
plt.legend()

# Mostrar el gráfico
plt.show()
"""


"""
temp_frec_sevilla = temp[temp['Area'] == 'Sevilla']['Avg. Temp (°C)']
temp_frec_madrid = temp[temp['Area'] == 'Madrid']['Avg. Temp (°C)']


print(temp_frec_sevilla)
print(temp_frec_madrid)
"""

print(temp_frec_sevilla.describe())
print(temp_frec_madrid.describe())

media_sevilla = temp_frec_sevilla.mean()
print("La media en Sevilla es:", media_sevilla)

mediana_sevilla = temp_frec_sevilla.median()
print("La mediana en Sevilla es:", mediana_sevilla)


moda_sevilla = temp_frec_sevilla.mode()[0]
print("La moda en Sevilla es:", moda_sevilla)

std_sevilla = temp_frec_sevilla.std()
print("La DS en Sevilla es:", std_sevilla)

media_madrid = temp_frec_madrid.mean()
print("La media en Madrid es:", media_madrid)

mediana_madrid = temp_frec_madrid.median()
print("La mediana en Madrid es:", mediana_madrid)

moda_madrid = temp_frec_madrid.mode()[0]
print("La moda en Madrid es:", moda_madrid)

std_madrid = temp_frec_madrid.std()
print("La DS en Madrid es:", std_madrid)

# desviacion estandar muestral n-1 (sin embargo en el ejercicio se tomo la poblacion)
print(np.std(temp_frec_sevilla, ddof=1))
print(np.std(temp_frec_madrid, ddof=1))


#La media de la temperatura de Sevilla 19.53 es mientras que la de Madrid es 15.52  
#Las desviaciones estandar poblacionales son 6.03 y 6.66 respectivamente
#Las desviaciones estandar muestrales son las mismas poblacionales
#Para Sevilla la mediana es: 19.25 y la moda es: 12.3
#Para Madrid la mediana es: 14.5 y la moda es: 25.7

q1, q3 = np.percentile(temp_frec_sevilla, [25, 75])
iqr_sevilla = q3 - q1
print("Primer cuartil Sevilla: ", q1)
print("Tercer cuartil Sevilla: ", q3)
# Rango intercuartílico es la diferencia entre el tercer cuartil y el primero
print("Rango intercuartílico Sevilla: ", iqr_sevilla)

q1, q3 = np.percentile(temp_frec_madrid, [25, 75])
iqr_madrid = q3 - q1

print("Primer cuartil Madrid: ", q1)
print("Tercer cuartil Madrid: ", q3)
print("Rango intercuartílico Madrid: ", iqr_sevilla)

# COVERAGE INTERVALS
n_sevilla = len(temp[temp['Area'] == 'Sevilla'])
ci_sevilla = t.interval(0.95, n_sevilla - 1, loc=media_sevilla, scale=std_sevilla / (n_sevilla ** 0.5))
print("coverage interval Sevilla", ci_sevilla)

n_madrid = len(temp[temp['Area'] == 'Madrid'])
ci_madrid = t.interval(0.95, n_madrid - 1, loc=media_madrid, scale=std_madrid / (n_madrid ** 0.5))
print("coverage interval Madrid ",ci_madrid)

temp_frec_sevilla = temp[temp['Area'] == 'Sevilla']['Avg. Temp (°C)']
temp_frec_madrid = temp[temp['Area'] == 'Madrid']['Avg. Temp (°C)']

data = [temp_frec_sevilla, temp_frec_madrid]
labels = ['Sevilla', 'Madrid']

fig, ax = plt.subplots()
ax.boxplot(data)
ax.set_xticklabels(labels)
ax.set_ylabel('Average Temperature (°C)')
plt.title('Boxplot of Average Temperatures')
plt.show()

plt.scatter(temp_frec_sevilla, temp_frec_madrid)
plt.xlabel('Average Temperature in Sevilla (°C)')
plt.ylabel('Average Temperature in Madrid (°C)')
plt.title('Scatter Plot of Average Temperatures')
plt.show()

#Calculo de Zscoring para ver que temperatura promedio es mas extrema relativamente

temp_Sevilla['z-score_sevilla'] = zscore(temp['Avg. Temp (°C)'])
temp_Madrid['z-score_madrid'] = zscore(temp['Avg. Temp (°C)'])



sevilla_zscore = (10.5 - temp_Sevilla['Avg. Temp (°C)'].mean()) / temp_Sevilla['Avg. Temp (°C)'].std()
madrid_zscore = (5.5 - temp_Madrid['Avg. Temp (°C)'].mean()) / temp_Madrid['Avg. Temp (°C)'].std()
print(sevilla_zscore)
print(madrid_zscore)
# Comparar los z-scores y determinar qué temperatura es más extrema con su valor absoluto(abs)
if abs(sevilla_zscore) > abs(madrid_zscore):
    print("La temperatura más extrema es la de Sevilla.")
else:
    print("La temperatura más extrema es la de Madrid.")


