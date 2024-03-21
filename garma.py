#################################################
#                Función GARMA
#
#@autor: Luis Armando Garcia Rodriguez
#################################################

#Librerias:
import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from scipy.stats import chi2
from numpy.ma.extras import isin
from scipy.stats import skew, kurtosis
import streamlit as st
from sklearn.preprocessing import StandardScaler

#Funcion pre_librerias


# garma.py

def mi_funcion():
    return "Hola garma.py, esta funcionando!!"


def pre_librerias():
 prelib = "#Librerias:\nimport numpy as np\nfrom numpy.linalg import inv\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport scipy.stats as stats\nfrom scipy.stats import chi2\nfrom scipy.spatial.distance import mahalanobis\nimport matplotlib.pyplot as plt\nfrom scipy.stats import chi2\nfrom numpy.ma.extras import isin\nfrom scipy.stats import skew, kurtosis\n"
 return print(prelib)

#Funciones de ordenamiento de datos y creacion de dataframes

def mi_base(formato):
  nombre_del_archivo = input("Escribe la ruta de acceso de tu archivo: ")

  if formato == 0:
     Base = pd.read_csv(nombre_del_archivo)
  elif formato == 1:
     Base = pd.read_excel(nombre_del_archivo)
  print("La base ha quedado establecida")

  return Base

#Funcion crear base

def crear_base(Base,columnas):
  Base_ = Base[columnas].copy()
  return Base_

#Funcion estadisticos

def estadisticos(Base_Descriptiva):
    resultados = ""  # Inicia la cadena vacía para almacenar los resultados

    minimos, maximos, medias, medianas, varianzas, desviaciones, asimetrias, kurt = ([] for i in range(8))

    for col in Base_Descriptiva.columns:
        datos = Base_Descriptiva[col]
        minimos.append(min(datos))
        maximos.append(max(datos))
        medias.append(np.mean(datos))
        medianas.append(np.median(datos))
        varianzas.append(np.var(datos, ddof=1))  # ddof=1 para varianza muestral
        desviaciones.append(np.std(datos, ddof=1))  # ddof=1 para desviación estándar muestral
        asimetrias.append(skew(datos))
        kurt.append(kurtosis(datos))

    for i, col in enumerate(Base_Descriptiva.columns):
        resultados += f"\nEstadísticos Descriptivos de: {col}\n"
        resultados += "-------------------------------------------------------\n"
        resultados += f"Min: {round(minimos[i],2)}, Max: {round(maximos[i],2)}\n"
        resultados += f"Media: {round(medias[i],2)}, Mediana: {round(medianas[i],2)}\n"
        resultados += f"Varianza: {round(varianzas[i],2)}, Desv Est: {round(desviaciones[i],2)}\n"
        resultados += f"Asimetría: {round(asimetrias[i],2)}, Kurtosis: {round(kurt[i],2)}\n"
        resultados += "-------------------------------------------------------\n\n"

    return resultados

import numpy as np
from scipy.stats import skew, kurtosis

def estadisticos2(datos):
  minimo = min(datos)
  maximo = max(datos)
  media = np.mean(datos)
  mediana = np.median(datos)
  varianza = np.var(datos, ddof=1)  # ddof=1 para varianza muestral
  desviacion = np.std(datos, ddof=1)  # ddof=1 para desviación estándar muestral
  asimetria = skew(datos)
  kurtosis_valor = kurtosis(datos)

  resultados = ""
  resultados += "\n\n"
  resultados += "Estadisticos Descriptivos de los Residuos\n"
  resultados += "-------------------------------------------------------\n"
  resultados += f"Min: {round(minimo,2)}, Max: {round(maximo,2)}\n"
  resultados += f"Media: {round(media,2)}, Mediana: {round(mediana,2)}\n"
  resultados += f"Varianza: {round(varianza,2)}, Desv Est: {round(desviacion,2)}\n"
  resultados += f"Asimetría: {round(asimetria,2)}, Kurtosis: {round(kurtosis_valor,2)}\n"
  resultados += "-------------------------------------------------------\n"
  resultados += "\n\n"

  return resultados


"""
#Funcion histogramas
def histogramas(Base_Descriptiva,i):
    sns.set_style("white")
        
        # Inicia una nueva figura de matplotlib
    imagen = plt.figure()
        
    sns.histplot(Base_Descriptiva[i], color ='#1776A7')
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    plt.title(f'Histograma de {i}')
        
        # Usa st.pyplot() en lugar de plt.show() para renderizar la figura en Streamlit
    return imagen
"""


def histogramas(Base_Descriptiva, i, bins=10, color='#E53511', edgecolor='black'):
    # Configuración del estilo
    sns.set_style("whitegrid")  # Añade una cuadrícula para facilitar la lectura
    
    # Inicia una nueva figura de matplotlib
    imagen = plt.figure()
    
    # Creación del histograma con opciones para ajustar
    sns.histplot(Base_Descriptiva[i], color=color, bins=bins, edgecolor=edgecolor, alpha=0.7)
    
    # Ajustes de las etiquetas y título
    plt.xlabel('Valores', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title(f'Histograma de {i}', fontsize=14)
    
    return imagen
"""
#Funcion densidades

def densidades(Base_Descriptiva,i):
   imagen = plt.figure()
   sns.set_style("white")
   sns.kdeplot(Base_Descriptiva.loc[:,i],color ='#1776A7')
   plt.xlabel('Valores')
   plt.ylabel('Densidad')
   plt.title(f'Curva de densidad de {i}')
   
   return imagen
  """
def densidades(Base_Descriptiva, i, color='#E53511', linewidth=2.5, fill=True):
    # Configuración del estilo
    sns.set_style("whitegrid")  # Uso de cuadrícula para facilitar la lectura
    
    imagen = plt.figure()
    # Creación de la curva de densidad con opciones personalizables
    sns.kdeplot(Base_Descriptiva.loc[:,i], color=color, linewidth=linewidth, fill=fill)
    # Ajustes de las etiquetas y título
    plt.xlabel('Valores', fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.title(f'Curva de densidad de {i}', fontsize=14)
    
    return imagen

"""
#Funcion Boxplot
def boxplot(Base_Descriptiva,i):
   imagen = plt.figure()
   sns.boxplot(x=Base_Descriptiva.loc[:,i], color='#1776A7', linewidth=2, width=0.4)
   plt.xlabel('Unidades')
   plt.ylabel(i)
   plt.title('Boxplot')
   return imagen
   """

def boxplot(Base_Descriptiva):
    # Estandarizar las columnas numéricas del DataFrame
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(Base_Descriptiva.select_dtypes(include=['float64', 'int64']))
    datos_escalados_df = pd.DataFrame(datos_escalados, columns=Base_Descriptiva.select_dtypes(include=['float64', 'int64']).columns)
    
    # Configurar el estilo de seaborn
    sns.set(style="whitegrid")
    
    # Inicializar la figura de matplotlib
    plt.figure(figsize=(12, 6))
    
    # Crear el boxplot con los datos estandarizados
    sns.boxplot(data=datos_escalados_df, palette="inferno", width=0.5)
    
    # Ajustar los detalles del gráfico
    plt.xticks(rotation=45, ha="right")  # Rotar las etiquetas del eje x para mejor lectura
    plt.title('Boxplots con variables estandarizadas')
    
    plt.tight_layout()  # Ajustar automáticamente los parámetros de la subtrama para que la subtrama se ajuste al área de la figura
    return plt

#Funcion dispersion
def dispersion(Base_Descriptiva,dependiente,j):
    fig=sns.jointplot(x= Base_Descriptiva[j], y= Base_Descriptiva[dependiente],data = Base_Descriptiva, color ='#E53511')
    return fig

#Funcion dispersion_regresion
def dispersion_regresion(Base_Descriptiva,dependiente,independiente):
  for j in independiente:
    sns.jointplot(x= j, y= dependiente, data= Base_Descriptiva,
                   kind="reg",  #kind={ “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” }
                   color="#1776A7", height=7)

#Funcion correlaciones
def Mcor(Base_Descriptiva):
  correlation_matrix = Base_Descriptiva.corr()
  correlacion = plt.figure(figsize=(7,7))
  ax=sns.heatmap(correlation_matrix,  vmax=1, vmin=-1,cbar_kws={"shrink": .8},square=True, annot=True,fmt='.2f', cmap ='inferno',center=0)
  bottom, top = ax.get_ylim()
  ax.set_ylim(bottom + 0.1, top - 0.1)
  plt.title('Matriz de correlaciones')
  return correlacion

#Funcion de variables

def dep_indep(Base,dependiente,independiente):
  VY = pd.DataFrame(Base[dependiente].copy())
  MVX = pd.DataFrame()
  for i in independiente:
   MVX[i] = Base[i].copy()
  return VY,MVX

#FUNCION REGRESION

def R_MCO(VY,MVX,nivel_confianza):

  # Variable dependiente Y

  y = [i for i in VY.iloc[:,0]]

  # Variable aleatoria multivariante con constante

  x = []
  ones = [1 for i in y]
  x.append(ones)
  for i in range(len(MVX.columns)):
    v = [j for j in MVX.iloc[:,i]]
    x.append(v)

  # n, k y df

  n = len(y)
  k = len(MVX.columns)
  df = n-k-1

 # Conversion a matrices

  Y = np.array(y)
  XT = np.array(x)

  #X
  X = XT.T

  #X transpuesta X
  XTX = np.dot(XT, X)

  # Inversa X transpuesta X
  invXTX = np.linalg.inv(XTX)

  # X transpuesta Y
  XTY = np.dot(XT,Y)

  # Betas
  B = np.dot(invXTX,XTY)

  # Y estimada
  YE = np.dot(X,B)

  # Residuos
  Residuos = Y - YE

  #SCE
  SCE = sum(Residuos**2)

  #SCR
  SCR = sum((YE - np.mean(Y))**2)

  #SCT
  SCT = sum((Y - np.mean(Y))**2)

  #R2 y R2 ajustada
  R2 = 1 - (SCE/SCT)
  R2a = 1 - ((SCE/(df))/(SCT/(n-1)))

  #Varianza del Modelo
  VarM = SCE/(df)

  #Error Estandar
  ES = np.sqrt(VarM)

  # Matris de varianzas y covarianzas de los betas
  MVC = np.dot(VarM,invXTX)

  #Varianza de betas
  Varianzas_betas = np.diag(MVC)

  #Errores estandar de los betas
  Errores_Estandar = [i**(1/2) for i in Varianzas_betas]

  #Estadistico F
  Estadistico_F = ((SCR/(k))/(SCE/(df)))
  x = abs(Estadistico_F)  # Valor para el cálculo de la probabilidad acumulativa
  dfn = k  # Grados de libertad del numerador
  dfd = df  # Grados de libertad del denominador
  probability = stats.f.cdf(x, dfn, dfd)
  pvalue_F = (1-probability)*2

  # Valores t
  t = B/Errores_Estandar

  #p-value
  pvalues_t = []
  for i in t:
   x =  abs(i)  # Valor para el cálculo de la probabilidad acumulativa
   probability = stats.t.cdf(x, df)
   p_valor_t = (1-probability)*2
   pvalues_t.append(round(p_valor_t,2))

  #Significancia
  signific = []
  nivel_confianza = (nivel_confianza/100)
  nivel_significancia = 1 - nivel_confianza
  for i in pvalues_t:
   if i < nivel_significancia:
    signific.append(True)
   else:
    signific.append(False)
  #Nombres
  Names2 = ["B0"]
  for va in MVX.columns:
    Names2.append(va)
  #Names = [f"{i}" for i in list(MVX.columns)]
  # Diccionario
  Resultados_dict = {
     'Estimador': Names2,
     'Coeficiente': B,
     'Error Estandar':Errores_Estandar,
     'Valor t':t,
     'P-value':pvalues_t,
     '¿Signif?': signific,
    }

  Estimaciones ={
      'Y': Y,
      'YE': YE,
      'Residuos': Residuos
  }

  Resultados_df = pd.DataFrame(Resultados_dict)
  Resultados_df.set_index("Estimador",inplace=True)

 

# Suponiendo que tienes las variables R2, R2a, VarM, ES, Estadistico_F, pvalue_F, n, MVX previamente definidas

  st.markdown("***Resumen de la regresion por MCO***")
  st.write(Resultados_df)
  st.write(f"***R2 = {round(R2,4)}, R2 ajustada = {round(R2a,4)}, Varianza del modelo = {round(VarM,4)}, Error Estandar = {round(ES,4)}, Estadistico F = {round(Estadistico_F,2)} en {len(MVX.columns)} y {df} grados de libertad -> p-value = {round(pvalue_F,4)}***")


  return Resultados_dict, Estimaciones, k, n


def Regresion_dada(VY,MVX,b,nivel_confianza):

  #Convertir en listas
  # Variable dependiente Y

  y = [i for i in VY.iloc[:,0]]

  # Variable aleatoria multivariante con constante

  x = []
  ones = [1 for i in y]
  x.append(ones)
  for i in range(len(MVX.columns)):
    v = [j for j in MVX.iloc[:,i]]
    x.append(v)

  # n, k y df

  n = len(y)
  k = len(MVX.columns)
  df = n-k-1

  # Conversion a matrices

  Y = np.array(y)
  XT = np.array(x)

 #X
  X = XT.T


 #X transpuesta X
  XTX = np.dot(XT, X)

 # Inversa X transpuesta X
  invXTX = np.linalg.inv(XTX)

 #Betas
  B = np.array(b)

 #Estimaciones

  YE = np.dot(X,B)

  # Residuos
  Residuos = Y - YE

  #SCE
  SCE = sum(Residuos**2)

  #SCR
  SCR = sum((YE - np.mean(Y))**2)

  #SCT
  SCT = sum((Y - np.mean(Y))**2)

  #R2 y R2 ajustada
  R2 = 1 - (SCE/SCT)
  R2a = 1 - ((SCE/(df))/(SCT/(n-1)))

  #Varianza del Modelo
  VarM = SCE/(df)

  #Error Estandar
  ES = np.sqrt(VarM)

  # Matris de varianzas y covarianzas de los betas
  MVC = np.dot(VarM,invXTX)

  #Varianza de betas
  Varianzas_betas = np.diag(MVC)

  #Errores estandar de los betas
  Errores_Estandar = [i**(1/2) for i in Varianzas_betas]

  #Estadistico F
  Estadistico_F = ((SCR/(k))/(SCE/(df)))
  x = abs(Estadistico_F)  # Valor para el cálculo de la probabilidad acumulativa
  dfn = k  # Grados de libertad del numerador
  dfd = df  # Grados de libertad del denominador
  probability = stats.f.cdf(x, dfn, dfd)
  pvalue_F = (1-probability)*2

  # Valores t
  t = B/Errores_Estandar

  #p-value
  pvalues_t = []
  for i in t:
   x =  abs(i)  # Valor para el cálculo de la probabilidad acumulativa
   probability = stats.t.cdf(x, df)
   p_valor_t = (1-probability)*2
   pvalues_t.append(p_valor_t)

  #Significancia
  signific = []
  nivel_confianza = (nivel_confianza/100)
  nivel_significancia = 1 - nivel_confianza
  for i in pvalues_t:
   if i < nivel_significancia:
    signific.append(True)
   else:
    signific.append(False)
  #Nombres
  Names = [f"Beta{i}" for i in range(len(B))]
  # Diccionario
  Resultados_dict = {
     'Estimador': Names,
     'Coeficiente': B,
     'Error Estandar':Errores_Estandar,
     'Valor t':t,
     'P-value':pvalues_t,
     '¿Signif?': signific,
    }

  Estimaciones ={
      'Y': Y,
      'YE': YE,
      'Residuos': Residuos
  }

  Resultados_df = pd.DataFrame(Resultados_dict)
  Resultados_df.set_index("Estimador",inplace=True)

  #Tabla

  print("")
  print("\033[1;30m"+"------------------------------------------------------------------------------------"+"\033[0m")
  print("\033[1;34m"+"Resumen de la regresion "+"\033[0m")
  print("\033[1;30m"+"------------------------------------------------------------------------------------"+"\033[0m")
  print(f"En {n} observaciones y {df} grados de libertad")
  print(f"Donde {VY.columns[0]} es la variable explicada, y: ")
  print(f"{[i for i in MVX.columns]};")
  print("Son las variables explicativas ")
  print("\033[1;30m"+"------------------------------------------------------------------------------------"+"\033[0m")
 #print(df_Results)
  print(Resultados_df)
  print("\033[1;30m"+"------------------------------------------------------------------------------------"+"\033[0m")
  print(f"R2 = {R2}, R2 ajustada = {R2a}")
  print("\033[1;30m"+"------------------------------------------------------------------------------------"+"\033[0m")
  print(f"Varianza del modelo = {round(VarM,4)}, ES del modelo = {round(ES,4)}")
  print(f"Estadistico F = {round(Estadistico_F,4)} -> p-value = {round(pvalue_F,4)} ")
  print("\033[1;30m"+"------------------------------------------------------------------------------------"+"\033[0m")
  print(f"Buscar el valor en tablas del Durbin Watson en n = {n}, y k = {len(MVX.columns)} en:")
  print("Tablas DW: https://www.ugr.es/~romansg/material/WebEco/01-comunes/dw.pdf")
  print("\033[1;30m"+"------------------------------------------------------------------------------------"+"\033[0m")
  print("")
  print("")
  return Resultados_dict, Estimaciones

#Normalidad

def normalidad(variable, nivel_confianza):
  #Prueba de Normalidad Jarque - Bera
  #Asimetria y Kurtosis
  nivel_significancia = 1 -(nivel_confianza/100)
  n = len(variable)
  s = stats.skew(variable)
  k = stats.kurtosis(variable)
  JB = (n)*((s**2/6)+(k**2/24))
  probabilidad_acumulada_para_jarque = chi2.cdf(JB, 2)
  pvalue_normalidad = 1 - probabilidad_acumulada_para_jarque
  if pvalue_normalidad >= nivel_significancia:
   resultado_JB = f"mayor al nivel de significancia de {round(nivel_significancia,2)}, por lo tanto: **No existe evidencia suficiente para rechazar la H0, por lo tanto; los residuos se distribuyen como una normal**"
   pasa_normalidad = True
  else:
   resultado_JB = f"menor al nivel de significancia de {round(nivel_significancia,2)}, por lo tanto: **Existe evidencia suficiente para rechazar la H0, por lo tanto; los residuos NO se distribuyen como una normal**"
   pasa_normalidad = False

  
  st.write("------------------------------------------------------------------------------------")
  st.markdown("***Prueba de Normalidad de los Residuos: Jarque - Bera***")
  st.write(f"**Variable**: Residuos, **grados de libertad**: {2}, **Estadístico Jarque - Bera**: {round(JB,4)}, **p_value**: {round(pvalue_normalidad,4)}")
  st.write("* H0: La distribución de los residuos se asimila a una normal")
  st.write("* H1: La distribución de los residuos NO se asimila a una normal")
  st.write(f"**Contraste de hipótesis**: el p-value asociado al estadístico JB es de {round(pvalue_normalidad,4)}, {resultado_JB}")

  imagen = plt.figure()
  sns.kdeplot(variable,color='#E53511', linewidth=2.5, fill=True)
  plt.xlabel('Valores', fontsize=12)
  plt.ylabel('Densidad', fontsize=12)
  plt.title(f'Curva de densidad de los Residuos', fontsize=14)
    
  return imagen,pasa_normalidad



#Prueba heterocedasticidad
#Funcion auxiliar para Breusch Pagan
def breusch_pagan(Residuos,MVX):
  #Regresion auxiliar
  y_auxiliar_bp = [i**2 for i in Residuos]
  x = []
  unos = [1 for i in MVX.iloc[:,0]]
  x.append(unos)
  for i in MVX.columns:
   v = [j for j in MVX.loc[:,i]]
   x.append(v)
  # X transpuesta y Y
  XT = np.array(x)
  Y = np.array(y_auxiliar_bp)
  # n, k , y df
  n = len(y_auxiliar_bp)
  k = len(x)
  df = n-k
  # X
  X = XT.T
  #X transpuesta X
  XTX = np.dot(XT, X)
  # Inversa X transpuesta X
  invXTX = np.linalg.inv(XTX)
  # X transpuesta Y
  XTY = np.dot(XT,Y)
  # Betas
  B = np.dot(invXTX,XTY)
  # Y estimada
  YE = np.dot(X,B)
  # Residuos
  Residuos_A = Y - YE
  #SCE
  SCE = sum(Residuos_A**2)
  #SCR
  SCR = sum((YE - np.mean(Y))**2)
  #SCT
  SCT = sum((Y - np.mean(Y))**2)
  #R2 y R2 ajustada
  R2 = 1 - (SCE/SCT)
  #Breusch Pagan
  BP = n*R2
  probabilidad_acumulada_para_BP = chi2.cdf(BP, k-1)
  pvalue_BP = 1 - probabilidad_acumulada_para_BP

  return BP, pvalue_BP

#Prueba Heterocedasticidad
def heterocedasticidad(Residuos,MVX,nivel_confianza):
  nivel_significancia = 1 - (nivel_confianza/100)
  bp_test = breusch_pagan(Residuos,MVX)
  estadistico_bp = bp_test[0]
  pvalor_bptest = bp_test[1]

  if pvalor_bptest >= nivel_significancia:
    inter_bp = f"mayor al nivel de significancia de {round(nivel_significancia,2)}, por lo tanto: **No existe evidencia suficiente para rechazar la H0; no existe heterocedasticidad**"
    pasa_heterocedaticidad = True
  else:
    inter_bp = f"menor al nivel de significancia de {round(nivel_significancia,2)}, por lo tanto: **Existe evidencia suficiente para rechazar la H0; existe heterocedasticidad**"
    pasa_heterocedaticidad = False

  st.write("------------------------------------------------------------------------------------")
  st.markdown("***Prueba de Heterocedasticidad: Breusch Pagan***")
  st.write(f"**Estadístico BP** = {bp_test[0]}, **p-value** = {round(bp_test[1],4)}")
  st.write("* H0: Homocedasticidad en la regresión (la varianza de los residuos es constante)")
  st.write("* H1: Heterocedasticidad en la regresión (la varianza de los residuos no es constante)")
  st.write(f"**Contraste de hipótesis**: El p-value asociado al estadístico BP es {round(pvalor_bptest,4)}, {inter_bp}")
  return pasa_heterocedaticidad

#Prueba de multicolinealidad

#Funcion adicional para las regresiones auxiliares en el VIF
#Esta funcion hace regresiondes auxiliares entre las variables explicativas y regresa los r^2 de estas regresiones
def auxiliar_regres_VIF(MVX):
 x=[]
 unos = [1 for i in range(len(MVX))]
 x.append(unos)
 for i in MVX.columns:
   v = [j for j in MVX.loc[:,i]]
   x.append(v)

 R2_aux_vif = []

 for i in range(len(x)):
   if i == len(x)-1:
     break
   else:
    i += 1
    y_aux = [j for j in x[i]]
    x_aux = [k for k in x[:i]+x[i+1:]]
    XT = np.array(x_aux)
    Y = np.array(y_aux)
    X = XT.T
    XTX = np.dot(XT,X)
    # Inversa X transpuesta X
    invXTX = np.linalg.inv(XTX)
    # X transpuesta Y
    XTY = np.dot(XT,Y)
    # Betas
    B = np.dot(invXTX,XTY)
    # Y estimada
    YE = np.dot(X,B)
    # Residuos
    Residuos = Y - YE
    #SCE
    SCE = sum(Residuos**2)
    #SCR
    SCR = sum((YE - np.mean(Y))**2)
    #SCT
    SCT = sum((Y - np.mean(Y))**2)
    #R2 y R2 ajustada
    R2 = 1 - (SCE/SCT)
    R2_aux_vif.append(R2)

 return R2_aux_vif

def multicolinealidad(MVX):

#Pruebas de Multicolinealidad
  #Para hacer las regresiones auxiliares
  R2_auxiliares = auxiliar_regres_VIF(MVX)
  VIF = []
  for i in R2_auxiliares:
      k = 1/(1-i)
      VIF.append(k)

  maximo_VIF = max(VIF)
  promedio_VIF = sum(VIF)/len(VIF)

  if maximo_VIF and promedio_VIF < 7:
    inter_VIF = "ninguno de los factores es indivudualmente o en promedio mayor a 7;"
    inter_VIF2 = "**no existe multicolinealidad**"
    pasa_multicolinealidad = True
  else:
    inter_VIF = "uno o mas de los factores fueron indivudualmente o en promedio mayores a 7;"
    inter_VIF2 = "**existe multicolinealidad**"
    pasa_multicolinealidad = False

  st.write("------------------------------------------------------------------------------------")
  st.markdown("***Prueba de Multicolinealidad: VIF***")
  for i in range(len(VIF)):
    st.write(f"**Variable**{i+1}: {round(VIF[i],4)}")
  st.write(f"**Maximo**: {round(maximo_VIF,4)}, **Promedio**:{round(promedio_VIF,4)}")
  st.write(f"Dado que {inter_VIF} {inter_VIF2}")
  return pasa_multicolinealidad






#Prueba autocorrelacion
def durbin_watson(Residuos):
  e_dw = [i for i in Residuos]
  e_r_dw = [j for j in Residuos]
  e_dw.pop(0)
  nu_dw = [(k-l)**2 for k,l in zip(e_dw,e_r_dw) ]
  num_dw = sum(nu_dw)
  de_dw = [i**2 for i in Residuos]
  den_dw = sum(de_dw)
  DW = num_dw/den_dw
  return DW

#Prueba de Autocorrelacion serial de orden uno


def autocorrelacion(Residuos,dL,dU,DW):
  if DW <= 2:
    if 0 < DW < dL:
      inter_dw = "**El estadistico DW ha caido en la zona de rechazo de la H0; la autocorrelación es diferente de 0; existe autocorrelación positiva**"
    elif dL < DW < dU:
      inter_dw = "**El estadistico DW ha caido en la zona de incertidumbre; puede o no existir autocorrelación positiva**"
    elif dU < DW <= 2:
      inter_dw = "**El estadistico DW ha caido en la zona de no rechazo de la H0; NO existe autocorrelación**"
  elif 2 < DW <= 4:
    if 2 < DW < (4-dU):
      inter_dw = "**El estadistico DW ha caido en la zona de no rechazo de la H0; NO existe autocorrelación**"
    elif (4-dU) < DW < (4-dL):
      inter_dw = "**El estadistico DW ha caido en la zona de incertidumbre; puede o no existir autocorrelación negativa**"
    elif (4-dL) < DW < 4:
      inter_dw = "**El estadistico DW ha caido en la zona de rechazo de la H0; la autocorrelación es diferente de 0; existe autocorrelación negativa**"

    # Mostrar los resultados usando Streamlit
  st.write("------------------------------------------------------------------------------------")
  st.markdown("***Prueba de Autocorrelación serial: Durbin Watson***")
  st.write("H0: No existe autocorrelación serial de primer orden")
  st.write("H1: Existe autocorrelación de primer orden")
  st.write(f"**DW = {DW}, dL = {dL}, dU = {dU}**")
  st.write(inter_dw)



#Graficos del modelo

import matplotlib.pyplot as plt
"""
def graf_md1(Base, Y, YE, udi, uvd, x_values):
    fig, ax1 = plt.subplots()
    # Usar x_values para el eje x en lugar de Base.index
    ax1.plot(x_values, Y, '#1776A7', label='Observaciones')
    ax1.plot(x_values, YE, '#922B21', label='Estimaciones')
    ax1.set_xlabel(udi)
    ax1.set_ylabel(uvd)

    # Ajustar la rotación de las etiquetas del eje x para mejorar la legibilidad, si es necesario
    plt.xticks(rotation=45)
    ax1.tick_params(axis='x', which='major', labelsize=10)
    lines = ax1.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')
    plt.title('Observaciones vs Estimaciones del modelo')
    plt.tight_layout()  
    return fig
"""

def graf_md1(Base, Y, YE, udi, uvd, x_values):
    fig, ax1 = plt.subplots()
    
    # Mejoras en el estilo de las líneas y la claridad visual
    ax1.plot(x_values, Y, color='#E53511', marker='o', linestyle='-', linewidth=2, markersize=5, label='Observaciones')
    ax1.plot(x_values, YE, color='#E56811', marker='s', linestyle='--', linewidth=2, markersize=5, label='Estimaciones')
    
    ax1.set_xlabel(udi, fontsize=12)
    ax1.set_ylabel(uvd, fontsize=12)
    
    # Mejora en la legibilidad de las etiquetas del eje x
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Ajuste de la leyenda para mejorar la claridad y evitar obstruir los datos
    ax1.legend(loc='best', fontsize=10)
    
    plt.title('Observaciones vs Estimaciones del modelo', fontsize=14)
    plt.tight_layout()
    
    return fig

"""

def graf_md2(Base, Y, YE, udi, uvd):
    fig, ax1 = plt.subplots()
    ax1.scatter(Base.index, Y, color='#1776A7', label='Observaciones')
    ax1.scatter(Base.index, YE, color='#922B21', label='Estimaciones')
    ax1.set_xlabel(udi)
    ax1.set_ylabel(uvd)
    ax1.legend(loc='upper right')

    plt.title('Observaciones vs Estimaciones del modelo')
    plt.tight_layout()

    return fig

"""
def graf_md2(Base, Y, YE, udi, uvd):
    fig, ax1 = plt.subplots()
    
    # Uso de marcadores distintos para observaciones y estimaciones
    ax1.scatter(Base.index, Y, color='#1776A7', marker='o', s=50, alpha=0.7, label='Observaciones')
    ax1.scatter(Base.index, YE, color='#922B21', marker='x', s=50, alpha=0.7, label='Estimaciones')
    
    ax1.set_xlabel(udi, fontsize=12)
    ax1.set_ylabel(uvd, fontsize=12)
    
    # Ajuste de la leyenda para mejorar la claridad y presentación
    ax1.legend(loc='best', fontsize=10)
    
    plt.title('Observaciones vs Estimaciones del modelo', fontsize=14)
    plt.tight_layout()
    
    return fig