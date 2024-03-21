import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
import garma


Resumen_Gen = {"Normalidad": None, "Heterocedasticidad": None, "Multicolinealidad": None, "Autocorrelación": None}

st.markdown("""<h1><center>Introducción a los Modelos de Regresión Lineal para Economistas.</center></h1>""",unsafe_allow_html=True)
st.markdown("**<h3><center>Luis Armando García Rodríguez</center></h3><br><br>**",unsafe_allow_html=True)
st.markdown("""
<style>
.justify {
  text-align: justify;
  text-justify: inter-word;
}
</style>

<div class="justify">
<h3>Presentación</h3>
                      
Hola, soy <strong>Luis Armando García Rodríguez</strong>. Soy estudiante de 8vo semestre en la Facultad de Economía de la UNAM. He desarrollado esta aplicación web con el objetivo de mejorar la eficiencia en la enseñanza y el aprendizaje de la econometría y el desarrollo de los modelos de regresión lineal.  
            
Desde mi punto de vista, para hacer econometría, no se necesita saber programar, ni memorizarse la construcción matemática de los modelos, pues cada vez existen más herramientas como esta, que facilitan ese trabajo. Es importante conocer la teoría estadística-matemática que respalda los modelos, pero lo más importante es saber interpretar estadísticas, gráficas, coeficientes y contrastes de hipótesis para poder describir e inferir con seguridad sobre nuestros datos.

El objetivo de esta aplicación web sigue esa filosofia, y busca enfocar al usuario a analizar e interpretar datos de forma sencilla y práctica, pero sin dejar de lado la teoría estadística-matemática detrás de los modelos.  

Con esta herramienta, podrás analizar datos, construir modelos de regresión lineal y verificar la calidad de los mismos en minutos y haciendo solo un par de clicks.  

Te invito a probar la app y a dejar tus comentarios en el cuestionario final. ¡Sigue las instrucciones y empieza a modelar!
        

</div>
""", unsafe_allow_html=True)

st.write("-------------------------------------")

st.markdown("""
<style>
  .justify {
  text-align: justify;
  text-justify: inter-word;
  } 
</style>
<div class="justify">
   <h4> 1 - Selecciona y configura tus datos</h4>
<div>""",unsafe_allow_html=True)

tipos_datos = ["Serie de Tiempo", "Corte Transversal"]
tipo_dato = st.selectbox("Selecciona la estructura de las datos a analizar", tipos_datos)

if st.button("¿Que estructura de datos debo seleccionar?"):
     st.markdown("""
<div class="justify">

Para analizar nuestros datos y entender como cambian las variables entre diferentes momentos del tiempo, o entre individuos y entidades. Debemos clasificar correctamente la estructura de nuestros datos.
Para ello, se suelen usar tres estructuras de datos que son:
* 1- <strong> Series de tiempo: </strong>  Son observaciones secuenciales (en el tiempo) de una o más variables. Estas observaciones tienen un atributo llamado frecuencia, que es el intervalo temporal que distancía una observación de otra, por ejemplo anualmente, mensualmente, semanalmente, diariamente, etc. En las series de tiempo se pueden analizar tendencias, ciclos, estacionalidades y otros patrones temporales. Alguno ejemplos de series de tiempo son: el PIB de México trimestral de 2020 a 2022, o el tipo de cambio (pesos por dolar) mensual de 2016 a 2023, etc.
                 
* 2 - <strong> Corte transversal: </strong> Son datos recopilados en un solo punto en el tiempo, pero de múltiples sujetos, individuos, empresas, países, etc. Esta estructura permite comparar las diferencias entre las unidades observadas en un momento específico. Algunos ejemplos de datos de corte transversal son: el nivel de deserción escolar por entidad federativa en 2023, el ingreso medio por alcaldia en la CDMX en  2022, etc.
                 
* 3 - <strong> Datos de panel: </strong> Son una combinacion entre elementos de series de tiempo y cortes transversales. Pues consisten en observaciones a lo largo del tiempo (como una serie de tiempo) de distintos sujetos, individuos, empresas, paises, etc (como en corte transversal). Esto permite ver las diferencias entre las unidades (como en los cortes transversales) y también cómo estas diferencias cambian a lo largo del tiempo (como en las series de tiempo). Algunos ejemplos de datos panel son: el indice de deserción escolar por enditad federativa de 2018 a 2024, el ingreso medio por alcaldia en la CDMX de 2018 a 2022, etc.

</div>
                 
""", unsafe_allow_html=True)
nivel_de_confianza = [90,95,99]
nc = st.selectbox("Selecciona el nivel de confianza (%) con el que evaluaremos el modelo",nivel_de_confianza)
ns = round(1-(nc/100),2)

if st.button("¿Que es el nivel de confianza?"):
     st.markdown("""
<div class="justify">
<p>El nivel de confianza representa el grado de certeza o seguridad que se tiene en que un parámetro poblacional se encuentre dentro de un intervalo de confianza calculado a partir de una muestra.<p>
<p>Es decir, es una medida que indica qué tan seguro podemos estar de que los intervalos de confianza calculados para los coeficientes de regresión contienen los verdaderos valores de los parámetros poblacionales. El nivel de confianza es un porcentaje y refleja la probabilidad de que, si el muestreo se repitiera muchas veces, el intervalo de confianza calculado a partir de cada muestra incluya al valor real del parámetro poblacional.<p>
    
</div>""", unsafe_allow_html=True)

# Carga de archivo
uploaded_file = st.file_uploader("Sube tu base de datos como un archivo Excel", type=['xlsx'])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Primeras 5 observaciones de tu base de datos")
    st.write(df.head(5))
    st.write("Ultimas 5 observaciones de tu base de datos")
    st.write(df.tail(5))
    if tipo_dato == "Serie de Tiempo":
        # Lista de columnas para seleccionar el índice de fechas
        var_series = df.columns.tolist()
        indice_serie = st.selectbox("Selecciona la columna que tiene el índice de tiempo (perido, fecha, mes, año, etc)", var_series)

        # Convertir la serie de índice a NumPy array (opcional)
        indice_serie_np = df[indice_serie].to_numpy()

        # Eliminar la columna del índice de fechas del DataFrame
        df.drop(indice_serie, axis=1, inplace=True)

        
    
         

    st.write("-----------------------------------------")
    

    st.markdown("""
<style>
.justify-text {
    text-align: justify;
    text-justify: inter-word;
}
</style>

<div class="justify-text">


<h4>2 - Análisis Exploratorio de Datos</h4>

                
***Estadísticas descriptivas***

1. **Mínimo**: El valor más pequeño entre todas las muestras registradas de una variable aleatoria.

2. **Máximo**: El valor más grande entre todas las muestras registradas de una variable aleatoria.

3. **Media**: El promedio de las muestras registradas de una variable aleatoria, calculada al sumar todos los valores y dividirlos entre el número de valores.

4. **Mediana**: El valor de en medio entre las muestras registradas y ordenadas de una variable aleatoria. Si hay un número par de datos, es el promedio de los dos valores medios.

5. **Varianza**: Una medida de la dispersión que muestra cuánto varían los valores de la media de una variable aleatoria. Se calcula como el promedio de las diferencias al cuadrado entre cada valor y la media.

6. **Desviación Estándar**: La raíz cuadrada de la varianza, proporciona una medida de la dispersión de los datos en torno a la media, en las mismas unidades que su respectiva variable aleatoria.

7. **Asimetría (Skewness)**: Una medida de la falta de simetría (¿que tan cargada esta a la derecha o izquierda?) en la distribución de una variable aleatoria.

8. **Curtosis (Kurtosis)**: Una medida del "apuntalamiento" o "aplanamiento" de la distribución de una variable aleatoria en comparación con la distribución normal. Una curtosis mayor a cero, indica una distribución leptocurtica; una curtosis de cero, indica una distribución mesocurtica; y una curtosis menor a cero, indica una distribución platocurtica.

</div>
""", unsafe_allow_html=True)


    if st.button("Conoce los estadisticos descriptivos clave de tus datos"):
        resultado_estadisticos = garma.estadisticos(df)
        st.text(resultado_estadisticos)

    st.write("-----------------------------------------")
    st.write("-----------------------------------------")

    import streamlit as st

    st.markdown("""
<style>
.justifyText {
    text-align: justify;
}
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div class='justifyText'>
<strong>Histogramas</strong><br><br>
Ayudan a visualizar la distribución de una variable aleatoria. Estos permiten identificar rápidamente cómo se agrupan los valores. Ademas, ayudan a detectar patrones, anomalías y comparar distribuciones entre diferentes variables. Son herramientas para el análisis exploratorio y descriptivo, que presentan una visión de la distribución de los datos.
<br><br>
</div>
""", unsafe_allow_html=True)


    if st.button("Grafica histogramas de tus datos"):
        for i in df.columns:
            histograma = garma.histogramas(df,i)
            st.pyplot(histograma)

    st.write("-----------------------------------------")
    st.write("-----------------------------------------")

    st.markdown("""
<div class='justifyText'>
<strong>Densidades</strong><br><br>
Las curvas de densidad representan la distribución de una variable aleatoria continua. A diferencia de los histogramas, las curvas de densidad dibujan una línea que ofrece una visión continua de la distribución de los datos.
<br><br>
</div>
""", unsafe_allow_html=True)
    
    if st.button("Grafica las curvas de densidad de tus datos"):
        for i in df.columns:
            densidad = garma.densidades(df,i)
            st.pyplot(densidad)

    st.write("-----------------------------------------")
    st.write("-----------------------------------------")

    st.markdown("""
<div class='justifyText'>
<strong>Boxplots</strong><br><br>
Los boxplots, o diagramas de cajas y bigotes, muestran la distribución de un conjunto de datos a través del primer cuartil (Q1), la mediana (Q2) y el tercer cuartil (Q3). Tienen los siguientes atributos:<br><br>
1- <strong>Caja Central</strong>: Representa el rango intercuartílico (IQR), es decir, la distancia entre el primer y tercer cuartil (Q3 - Q1). La caja muestra la dispersion de 50 por ciento de los datos centrales.<br>
2- <strong>Línea de la Mediana</strong>: Dentro de la caja, una línea indica la mediana (Q2), el valor de en medio en la distribución de la variable aleatoria.<br>
3- <strong>Bigotes</strong>: Se extienden desde la caja hasta los valores máximo y mínimo que están dentro de un rango de 1.5 veces el IQR.<br>
4- <strong>Puntos Externos</strong>: Cualquier dato fuera de los bigotes se considera un valor atípico y se representa con puntos.<br><br>
Los boxplots son utiles para visualizar la dispersión, la tendencia central y la simetría de los datos, así como para identificar valores atípicos. Son muy efectivos al comparar distribuciones entre varias variables de datos cuando se usan valores estandarizados, es decir, restados por su media y divididos por su desviación estandar.
<br><br>
</div>
""", unsafe_allow_html=True)
    
    if st.button("Grafica boxplots de tus datos"):
        boxplot = garma.boxplot(df)
        st.pyplot(boxplot)

    st.write("-----------------------------------------")
    st.write("-----------------------------------------")

    st.markdown("""
<div class="justifyText">
    <p><b>Correlaciones</b></p>
    <p>La matriz de correlaciones se utiliza para medir y mostrar el grado de asociacion lineal entre un conjunto de variables. En esta matriz, cada fila y columna representa una variable diferente, y cada recuadro muestra el coeficiente de correlación de Pearson (que asume una relación lineal) entre las dos variables que se encuentran en esa posición. Tienen los siguientes atributos:</p>
    <ol>
        <li><b>Correlación de Pearson</b>: Son valores que oscilan entre -1 y 1. Un valor de 1 indica una correlación positiva perfecta; -1, una correlación negativa perfecta; y 0, que no existe correlación lineal.</li>
        <li><b>Simetría</b>: La matriz es simétrica, lo que significa que la correlación entre dos variables es la misma tanto arriba como abajo de la diagonal principal.</li>
        <li><b>Diagonal Principal</b>: En la diagonal principal, donde una variable se encuentra consigo misma, el valor de correlación es siempre 1.</li>
    </ol>
<br>
</div>
""", unsafe_allow_html=True)

    if st.button("Obten la matriz de correlaciones de tus datos"):
        Mcorrelaciones = garma.Mcor(df)
        st.pyplot(Mcorrelaciones)

    st.write("-----------------------------------------")
    st.write("-----------------------------------------")

    # Mostrar las columnas del DataFrame para que el usuario elija
    st.write("Selecciona la variable dependiente y las independientes")
    columnas = df.columns.tolist()
    dependiente_y = st.selectbox("Variable Dependiente", columnas)
    independiente_X = st.multiselect("Variables Independientes", columnas, default=columnas[0])

    X = df[independiente_X]
    y = df[dependiente_y]

    VY, MVX = garma.dep_indep(df,dependiente_y,independiente_X)

    X1 = X.copy()
    y1 = y.copy()
    X1 = sm.add_constant(X1)

    


    st.write("-----------------------------------------")
    st.write("-----------------------------------------")



    st.markdown("""
<div class="justifyText">
    <p><b>Dispersiones</b></p>
    <p>Los gráficos de dispersión son usados para visualizar la relación entre dos variables cuantitativas. Cada punto en el gráfico corresponde a un par de valores de estas dos variables. En un eje se representa una variable y en el otro, otra variable.</p>
    <p>Estos gráficos permiten observar si existe una relación o correlación entre las dos variables. También ayudan a ver cómo se distribuyen los datos en el espacio bidimensional y a identificar patrones o agrupaciones o incluso valores atípicos en el espacio bivariante, que no siguen la tendencia general de los demás datos.</p>
</div>
""", unsafe_allow_html=True)

    if st.button(" Grafica diagramas de dispersión para tus datos"):
        for j in independiente_X:
            dispersion = garma.dispersion(df,dependiente_y,j)
            st.pyplot(dispersion)

    st.write("-----------------------------------------")
    st.write("-----------------------------------------")
    st.markdown("""<div class="justify"><h4>3 - Análisis de Regresión Lineal</h4><div>""",unsafe_allow_html=True)
    # Justificar texto sin ecuaciones
    st.markdown("""
<div class="justifyText">
    <p>La regresión lineal por el Método de Mínimos Cuadrados Ordinarios (MCO) se utiliza para modelar la relación entre una variable dependiente y una o más variables independientes. El objetivo es encontrar la línea (en el caso de una sola variable independiente) o el plano (con múltiples variables independientes) que mejor se ajuste a los datos.</p>
    <p>El caso más simple, con una sola variable independiente, se expresa como Y = b<sub>0</sub> + b<sub>1</sub>X + ε, donde Y es la variable dependiente; X es la variable independiente; b<sub>0</sub> es el término de intercepción; b<sub>1</sub> es el coeficiente de la variable independiente; y ε es el término de error.</p>
    <p><b>Mínimos Cuadrados:</b> Se utilizan los mínimos cuadrados para estimar los parámetros del modelo b<sub>0</sub> y hasta b<sub>n</sub>. Esto se hace minimizando la suma de los cuadrados de las diferencias entre los valores observados y los valores predichos por el modelo. Lo anterior, se logra con la siguiente ecuación matricial:</p>
</div>
""", unsafe_allow_html=True)


# Ecusaciones en formato LaTeX para mínimos cuadrados
    st.latex(r"\beta = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}")
    st.markdown("""
                <div class='justifyText'>
                <p>La ecuación anterior se lee como: La matriz de coeficientes beta, es igual a la matriz inversa de X transpuesta X, multiplicada por la matriz X transpuesta Y.<p>
                <p>Donde:<p>
                <ul>
                 <li>B: Es la matriz de coeficentes Beta de tamaño p x 1</li>
                 <li>X: Es la matriz de diseño (una multivariable aleatoria) de tamaño n x p. Donde n es el numero de observaciones y p el número de variables independientes (incluyendo un vector de unos si el modelo incluye constante)</li>
                 <li>Y: Es la variable dependiente de tamaño n x 1</li>
                </ul>
                <br>
                <br>
                </div>""", unsafe_allow_html=True)


    Resultados_dict, Estimaciones, k_, n = garma.R_MCO(VY,MVX,nc)

    if st.button("¿Como interpretar el resumen de la regresión?"):
         st.markdown("""
<div class="justifyText">
    <p> En el cuadro anterior, se muestran 6 columnas, la primera de ellas <strong>Estimador</strong> indica el nombre del respectivo estimador empleado en la regresión, el primero de ellos, es b<sub>0</sub>, que es el intecepto o constante, el cual en la columna <strong>Coeficiente</strong> indica el valor esperado de Y cuando cualquier otra variable empleada en la regresion es 0.</p>
    <p> Por otro lado desde b<sub>1</sub> y hasta b<sub>n</sub> la columna <strong>Coeficiente</strong> indican el valor del cambio marginal de Y por X. Es decir: cuanto cambia el valor de la variable Y, cuando la variable asociada a ese coeficiente aumenta en 1 unidad. Suponiendo que las demas variables en la regresión se mantienen constantes.
    <p> La columna <strong>Error Estandar</strong> muestra este estadistico relativo a cada coeficiente. El error estandar indica la variación de estas estimaciones si seleccionáramos muchas submuestras diferentes de la misma muestra de datos que tenemos y calculáramos los coeficientes para cada una de ellas. Un error estándar pequeño indica que las estimaciones del coeficiente son precisas y no varían mucho de una muestra a otra, mientras que un error estándar grande indica mayor variabilidad y menor precisión.<p>
    <p> La columna <strong>Valor t</strong> indica dichos valores, que son calculados como la división del valor del coefiente y su respectivo Error Estandar. Estos valores son estadísticos de prueba que se utilizan para evaluar la significancia indivual de cada coeficiente estimado en el modelo de regresión. Dicho estadistico se compara con un valor critico en la distribución t a determinado nivel de significancia y grados de libertad para constrastar la hipótesis de que sean estadísticamente iguales a 0.<p>
    <p> La columna siguiente <strong>P-value</strong> nos dice la probabilidad de observar una relación; la estimada entre la variable independiente y la dependiente, asumiendo que no existe dicha relación en la población, es decir, asumiendo que el verdadero valor del coeficiente b es igual a cero. En terminos sencillos, el p-value puede entenderse como la probabilidad de que la hipótesis nula sea verdadera. Si este valor es menor al nivel de significancia, se rechaza la hipótesis nula y se acepta la alternativa; pero si es mayor, no se puede rechazar la hipótesis nula.<p>
    <p> Por ultimo, la columna <strong>¿Signif?</strong> indica dos valores True o False, que se representan como una casilla con una palomita (True) o una casilla vacía (False), que indica si el estimador empleado es estadísticamente significativo y ayuda a explicar a la variable dependiente o no. Esto se calcula constrastando el p-valor del estimador con el nivel de significancia, como se explica anteriormente.<p>
    <p> Al final del cuadro se muestran algunos estadisticos, los cuales proporcionan información para evaluar el modelo y sobre cómo este se ajusta a los datos y la significancia de las relaciones entre variables.<p>
    <p> <strong>R<sup>2</sup>:</strong> Mide la variabilidad de la variable dependiente que es explicada por el modelo de regresión. Es un indicador que nos dice qué tan bien las variables independientes explican a la variable dependiente. Este estadistico varía de 0 a 1, donde un valor más alto indica un mejor ajuste del modelo a los datos. Un 1 significa que el modelo explica toda la variabilidad en los datos, mientras que un 0 indica que no explica ninguna.<p>
    <p> <strong>R<sup>2</sup> ajustada:</strong> Es el mismo principio del R<sup>2</sup>, solo que este tiene en cuenta el número de predictores (variables independientes) en el modelo. Y penaliza la adición de predictores que no mejoran significativamente el modelo. Ya que por construcción el  R<sup>2</sup> siempre mejorará al incorporar más variables explicativas aunque estas no aporten información relevante.<p>
    <p> <strong>Varianza del modelo:</strong> La varianza del modelo indica cuánto varían las estimaciones del modelo. Es decir, nos habla de la dispersión de los valores estimados por el mismo.<p>
    <p> <strong>Error Estandar:</strong> Mide la dispersión media de los valores observados alrededor de los valores estimados por el modelo. Es decir, es la estimación de la desviación estándar de los errores y la raiz cuadrada de la varianza del modelo. Se interpreta como una medida de evaluación del modelo, ya que un error estandar bajo indica que las estimaciones del modelo están, en promedio, cercanas a los valores reales.<p>
    <p> <strong>Estadistico F y su p-valor asociado:</strong> Se usa para probar la significancia conjunta del modelo de regresión. Compara un modelo con al menos un estimador contra un modelo nulo que no tiene estimadores(solo b<sub>0</sub>). Un valor alto del estadistico F indica que el modelo con los estimadores es significativamente mejor para explicar la variable dependiente, en comparación con el modelo nulo. El p-value asociado, se utiliza para contrastar la hipótesis de que no haya ninguna relación entre las variables independientes y la dependiente (es decir, que todos los coeficientes asociados a variables son estadísticamente iguales 0). Un p-valor menor al nivel de significancia indica que los coeficientes asociados a variable del modelo son conjuntamente, estadísticamente significativos y distintos de 0. Mientras que con un p-value mayor a 0.05, no se puede rechazar la hipótesis nula (que los coefientes asociados a variable son estadísticamente iguales a 0), por lo cual, el modelo no es estadísticamente significativo en conjunto.<p>
</div>
""", unsafe_allow_html=True)

    st.write("-----------------------------------------")
    st.write("-----------------------------------------")

    st.markdown("""
<div class="justifyText">
    <h4>4 - El Teorema de Gauss-Markov</h4>
    <p>Los supuestos de Gauss-Markov llevan los nombres de Carl Friedrich Gauss y Andrey Markov, dos matemáticos importantes que contribuyeron al desarrollo de la teoría de la estadística y la probabilidad en el siglo XIX y principios del siglo XX.</p>
    <p>Estos supuestos se establecieron para proporcionar condiciones bajo las cuales los estimadores de mínimos cuadrados ordinarios (MCO) en la regresión lineal serían los mejores estimadores lineales insesgados. Al cumplirse estos supuestos, según el teorema, los estimadores de MCO tendrían las propiedades de insesgadez, consistencia, eficiencia y linealidad. Es decir, estos serían los estimadores más precisos y útiles para estimar los parámetros desconocidos en un modelo de regresión lineal.</p>
    <p>Dichos supuestos pueden resumirse en los siguientes:</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
- **1- Linealidad en parámetros:** La relación entre la variable dependiente y las variables independientes debe ser lineal en los parámetros. Esto significa que la relación no puede ser no lineal (es decir, no puede ser exponencial, logarítmica, o de otro tipo) en los parámetros.

- **2- Esperanza de los errores igual a cero:** La esperanza matemática de los errores debe ser igual a cero, lo que implica que los errores no están sesgados.

- **3- Homocedasticidad:** La varianza de los errores debe ser constante para todos los valores de las variables independientes, lo que significa que debe haber una dispersión constante de los errores a través de todas las observaciones.

- **4- No multicolinealidad:** Las variables independientes no deben estar altamente correlacionadas entre sí, ya que esto dificulta la estimación e interpretación de los coeficientes y puede hacer que las estimaciones de los parámetros sean inexactas.

- **5- No autocorrelación:** Los errores deben ser independientes entre sí, es decir, no debe haber correlación entre los errores de diferentes observaciones.
""")

    st.markdown("""
<div class="justifyText">
    <p>Este aplicativo web ha sido diseñado no solo para estimar las regresiones, sino también para verificar el cumplimiento de dichos supuestos a través de distintos contrastes de hipótesis que se muestran a continuación.</p>
    <h5>Linealidad</h5>
    <p>El supuesto de linealidad se cumple desde la estimación, ya que los parámetros presentados por esta web son lineales, y fueron obtenidos por MCO.</p>
</div>
""", unsafe_allow_html=True)


    # Texto justificado sin la ecuación
    st.markdown("""
<div class="justifyText">
    <h5>Normalidad de los errores</h5>
    <p>Este supuesto asume que la media de los errores debe ser cero, lo cual, por teorema, siempre suele cumplirse. En caso de no ser así, hablamos de que el modelo está sobrestimando (media mayor que cero) o subestimando (media menor a cero). Otra cosa importante referente a este supuesto es la normalidad de los errores, pues se asume que los errores tienen distribución normal con media cero (supuesto 1) y varianza constante (supuesto 3). Si los errores no siguen esta distribución, los intervalos de confianza fallan y los contrastes de hipótesis sobre los parámetros del modelo pueden llevar a falsos positivos o verdaderos negativos en los mismos. Cuando los errores no siguen una distribución normal, las propiedades de los estimadores de mínimos cuadrados ordinarios (MCO) no son las óptimas.</p>
    <p><b>La prueba de normalidad de Jarque-Bera:</b></p>
    <p>Se utiliza para determinar si la distribución de una variable tiene (o no) una distribución normal. Su calculo se basa en los coeficientes de asimetría y curtosis de la muestra, y compara estos valores con los que se esperarían de una distribución normal.</p>
    <p>En una distribución normal, la asimetría debe ser cercana a cero, indicando que la distribución es simétrica. Y la curtosis ser cercana a tres, indicando una distribución "normal" en términos de la altura y la amplitud de las colas.</p>
    <p>La fórmula del estadistico de prueba de Jarque-Bera es:</p>
</div>
""", unsafe_allow_html=True)

# Ecusación en formato LaTeX
    formula = r"JB = n \left( \frac{S^2}{6} + \frac{(K-3)^2}{24} \right)"
    st.latex(formula)

# Detalles de la fórmula
    st.markdown("""
<div class="justifyText">
    <p>Donde:</p>
    <ul>
        <li><strong>n</strong> es el tamaño de la muestra.</li>
        <li><strong>S</strong> es la asimetría de la muestra.</li>
        <li><strong>K</strong> es la curtosis de la muestra.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
    p_normalidad = None
    if st.button("Realiza la prueba de Jarque-Bera"):
            imagen, p_normalidad = garma.normalidad(Estimaciones['Residuos'],nc)
            rest = garma.estadisticos2(Estimaciones['Residuos'])
            st.write("-----------------------------------")
            st.text(rest)
            st.pyplot(imagen)
    if p_normalidad == True:
         Resumen_Gen["Normalidad"] = True
    elif p_normalidad == False:
         Resumen_Gen["Normalidad"] = True
    ResGen1 = Resumen_Gen.copy()


    st.write("-------------------------------------------")
    st.write("-------------------------------------------")

    st.markdown("""
<div class="justifyText">
    <h5>Heterocedasticidad</h5>
    <p>Establece que la varianza de los errores es constante para todas las observaciones, es decir, que la dispersión de los errores es la misma en toda la gama de valores de las variables independientes.</p>
    <p>Cuando este supuesto se viola y la varianza de los errores no es constante a lo largo de los valores de las variables independientes, se dice que existe heterocedasticidad. La heterocedasticidad tiene varios efectos no deseados en el análisis de regresión, como la ineficiencia de los estimadores de mínimos cuadrados ordinarios (MCO), la incorrecta interpretación de los intervalos de confianza y las pruebas de hipótesis, así como la producción de estimaciones de errores estándar sesgadas.</p>
    <p><strong>La prueba de heterocedasticidad de Breusch-Pagan</strong> es utilizada para detectar la presencia de heterocedasticidad en un modelo de regresión. El procedimiento de la prueba de Breusch-Pagan es el siguiente:</p>
    <p>Primero, se estima el modelo de regresión lineal y se calculan los residuos. Luego, se realiza una regresión de los cuadrados de los residuos del modelo original sobre las mismas variables independientes.</p>
    <p>Posteriormente se calcula el Multiplicador de Lagrange (LM) de la siguiente forma:</p>
</div>
""", unsafe_allow_html=True)
    
    formula = r"LM =  n\times R^2 "
    st.latex(formula)

    st.markdown("""
<div class="justifyText">
    <p>Este multiplicador sigue una distribución X<sup>2</sup>(ji cuadrada) con k grados de libertad, donde k es el número de variables independientes en la regresión auxiliar (excluyendo el término constante b<sub>0</sub>). Entonces se plantea el contraste de hipótesis, donde se rechaza la hipótesis nula (Homocedasticidad) si el valor del LM es mayor que el valor crítico de la distribución X<sup>2</sup>(ji cuadrada) en k grados de libertad, y al nivel de significancia determinado.<p>
""", unsafe_allow_html=True)


    p_heterocedasticidad = None
    if st.button("Realiza la prueba de Breusch - Pagan"):
            p_heterocedasticidad = garma.heterocedasticidad(Estimaciones["Residuos"], MVX,nc)
    if p_heterocedasticidad == True:
         ResGen1["Heterocedasticidad"] = True
    elif p_heterocedasticidad == False:
         ResGen1["Heterocedasticidad"] = False
    ResGen2 = ResGen1.copy()

    st.write("-------------------------------------------")
    st.write("-------------------------------------------")


    st.markdown("""
<div class="justifyText">
    <h5>Multicolinealidad</h5>
    <p>Este supuesto indica que las variables independientes en un modelo de regresión lineal no deben ser linealmente dependientes, o estar altamente correlacionadas entre sí. La multicolinealidad significa que una o más variables independientes en el modelo están altamente correlacionadas lo que tiene efectos negativos en el análisis de regresión.</p>
    <p>Cuando esto ocurre, se vuelve difícil determinar el efecto individual de cada variable en la variable dependiente, ya que los efectos de una variable pueden ser atribuidos incorrectamente a otra. Además, la multicolinealidad puede hacer que los coeficientes estimados sean menos precisos, lo que resulta en intervalos de confianza más amplios y menos confiables. También causa que pequeños cambios en los datos puedan llevar a cambios drásticos en los coeficientes estimados, lo que hace que el modelo sea menos estable y más sensible a los cambios en los datos.</p>
    <p><strong>El Factor de Inflación de la Varianza (VIF)</strong> se utiliza en los modelos de regresión para detectar la multicolinealidad entre las variables independientes. El VIF mide cuánto se infla la varianza de un coeficiente estimado debido a la multicolinealidad. Un VIF de 1 indica que no hay correlación entre la variable independiente en cuestión y las otras variables independientes, y por lo tanto no hay multicolinealidad. A medida que el VIF aumenta, también lo hace la multicolinealidad.</p>
    <p>El VIF se calcula así:</p>
    <p>Para cada variable independiente X<sub>i</sub>, se realiza una regresión lineal de X<sub>i</sub> utilizando todas las demás variables independientes en el modelo.</p>
    <p>El VIF para cada variable independiente se calcula como:</p>
</div>
""", unsafe_allow_html=True)

# Ecusación en formato LaTeX
    formula = r"\text{VIF}_i = \frac{1}{1 - R^2_i}"
    st.latex(formula)

# Continuación del texto justificado
    st.markdown("""
<div class="justifyText">
    <p>Donde R<sup>2</sup><sub>i</sub> es el coeficiente de determinación de la regresión lineal de X<sub>i</sub> sobre las otras variables independientes.</p>
    <p>Un VIF superior a 7 indica una alta multicolinealidad, para solucionarlo, se pueden eliminar variables, combinarlas o aplicar técnicas como PCA.</p>
</div>
""", unsafe_allow_html=True)
    p_multicolinealidad = None
    if st.button("Obten el factor de inflación de la varianza de tu modelo"):
            p_multicolinealidad = garma.multicolinealidad(MVX)
    if p_multicolinealidad == True:
         ResGen2["Multicolinealidad"] = True
    elif p_multicolinealidad == False:
         ResGen2["Multicolinealidad"] = False
    ResGen3 = ResGen2.copy()


    st.write("-------------------------------------------")
    st.write("-------------------------------------------")

    st.markdown("""
<div class="justifyText">
    <h5>No Autocorrelación</h5>
    <p>El supuesto de no autocorrelación establece que los errores (o residuos) en la regresión lineal no deben estar correlacionados entre sí.</p>
    <p>La autocorrelación ocurre cuando los residuos de un modelo de regresión no son independientes entre sí, es decir, el valor de un residuo está influenciado o puede ser explicado a partir de los valores de residuos anteriores. Esto es más común al intentar modelar series de tiempo con regresión lineal, donde los datos están ordenados cronológicamente y los valores cercanos en el tiempo suelen estar correlacionados.</p>
    <p>El supuesto de no autocorrelación es importante porque la presencia de autocorrelación puede llevar a una subestimación de la varianza de los estimadores, y a su vez, llevar a que las pruebas de hipótesis sean menos fiables. Cuando los residuos están correlacionados, el uso de técnicas estándar de inferencia estadística basadas en MCO puede dar lugar a conclusiones erróneas sobre la significancia de los coeficientes del modelo.</p>
    <p><strong>Prueba de Breusch-Godfrey:</strong> Es un constraste de hipótesis estadístico utilizado para detectar la presencia de autocorrelación en los residuos de un modelo de regresión lineal. A diferencia de la prueba de Durbin-Watson, la prueba de Breusch-Godfrey es más flexible y puede usarse en situaciones donde la estructura de la autocorrelación es más compleja o cuando se sospecha de autocorrelación de orden superior.</p>
    <p>La prueba se realiza en dos pasos:</p>
    <ul>
        <li>Estimación del modelo de regresión original: Primero, se estima el modelo de regresión lineal y se obtienen los residuos de este modelo.</li>
        <li>Regresión de los residuos estimados: Se realiza una regresión auxiliar donde los residuos estimados del modelo original se regresan sobre las variables independientes originales y los valores rezagados (o retardos) de los residuos. El número de retardos incluidos como regresores depende del orden de la autocorrelación que se está probando.</li>
    </ul>
    <p>La prueba utiliza el Multiplicador de Lagrange (LM) para evaluar la significancia de los coeficientes de los términos de los residuos rezagados en la regresión auxiliar. El estadístico LM se calcula como:<p></div>""", unsafe_allow_html=True)
    formula = r"LM = n \times R^2"
    st.latex(formula) 
    st.markdown(""" <div class="justifyText">
                Este sigue una distribución X<sup>2</sup>(ji cuadrada) con p grados de libertad, donde p es el número de rezagos de los residuos incluidos en el modelo auxiliar. Para determinar si hay evidencia de autocorrelación, se contrasta el LM con el valor crítico de la distribución X<sup>2</sup>(ji cuadrada) correspondiente a un nivel de significancia.</p>
</div>
""", unsafe_allow_html=True)
    
    
    p_autocorrelacion = None
    rezagos= st.number_input(f'Cuantos rezagos deseas evaluar?',min_value=1,max_value=100, step=1)
    modelo = sm.OLS(y1, X1).fit()
    if st.button("Realiza el test de Breusch-Godfrey"):
         resultado_bg = acorr_breusch_godfrey(modelo, nlags=rezagos)
         estadistico_bg = resultado_bg[0]
         pvalue_bg = resultado_bg[1]
         if pvalue_bg >= ns:
              contraste_bg = f"El p-value asociado al estadístico BG es {round(pvalue_bg,4)}, mayor o igual al nivel de significancia de {round(ns,2)}, por lo tanto; **No existe evidencia suficiente para rechazar la H0, no existe autocorrelación serial de orden {rezagos}**"
              p_autocorrelacion = True
         elif pvalue_bg < ns:
              contraste_bg = f"El p-value asociado al estadístico BG es {round(pvalue_bg,4)}, menor al nivel de significancia de {round(ns,2)}, por lo tanto; **Existe evidencia sufiente para rechazar la H0; existe autocorrelación serial de orden {rezagos}**"
              p_autocorrelacion = False
                 
         st.write("-------------------------------------------")
         st.write("***Prueba de Autocorrelación: Breusch-Godfrey***")
         st.write(f"Estadístico LM = {round(estadistico_bg,4)}, p-value asociado = {round(pvalue_bg,4)}")
         st.write("""
                  * H0: No existe autocorrelación
                  * H1: Existe autocorrelación""")
         st.write("**Contraste de hipótesis:**", contraste_bg)

    st.write("-------------------------------------------")
    st.write("-------------------------------------------")

    if p_autocorrelacion == True:
         ResGen3["Autocorrelacion"] = True
    elif p_autocorrelacion == False:
         ResGen3["Autocorrelacion"] = False
    ResGen4 = ResGen3.copy()








    st.markdown("""
<div class="justifyText">
    <h4> 5 - Gráfica de Evaluación del Modelo</h4>
    <p>Realiza gráficas para evaluar el modelo y comparar las estimaciones con los valores reales de la variable dependiente (objetivo).</p>
</div>
""", unsafe_allow_html=True)

   
    

    if tipo_dato == "Serie de Tiempo":
         uvd = st.text_input(f'En que unidades esta la variable independiente?')
         udi = "Fecha"
         if st.button("Graficar como serie de tiempo"):
              graf = garma.graf_md1(df,Estimaciones['Y'],Estimaciones['YE'],udi,uvd,x_values=indice_serie_np)
              st.pyplot(graf)
              st.set_option('deprecation.showPyplotGlobalUse', False)


    if tipo_dato == "Corte Transversal":
        udi = st.text_input(f'Que son las observaciones (personas, autos, casas, etc)?')
        uvd = st.text_input(f'En que unidades esta la variable independiente?')
        if st.button("Graficar como corte transversal"):
             graf = garma.graf_md2(df,Estimaciones['Y'],Estimaciones['YE'],udi,uvd)
             st.pyplot(graf)
             st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write("-------------------------------------------")
    st.write("-------------------------------------------")


    st.markdown("""
<div class="justifyText">
            <p><h4>5 - Estimación</h4><p>
            <p>Ahora que has construido y evaluado tu el modelo, puedes obtener estimaciones con él. Introduce valores para cada variable independiente. Recuerda, que para que tu estimación sea fiable, el modelo debe haber pasado todos los supuestos del teorema Gauss - Markov.<P>
</div>
             """,unsafe_allow_html=True)

    betas_para_predecir = [1]
    b = {}
    for i in independiente_X:
         clave = i  
         b[clave] = st.number_input(f"Escribe el valor de {i}")  # Asignamos el valor 0 a esta clave en el diccionario b
         betas_para_predecir.append(b[clave])

    betas_para_predecir = np.array(betas_para_predecir)
    estimacion_prediccion = np.dot(betas_para_predecir,Resultados_dict['Coeficiente'])
    st.write("----------------------------------------------------------------")
    st.write(f"**El valor estimado de {dependiente_y} con los valores de {independiente_X} proporcionados, es de: {round(estimacion_prediccion,4)}**")
    st.write("----------------------------------------------------------------")

    st.markdown("""
<div class="justifyText">
            <p><h4>6 - Referencias</h4><p>
            <ul>
                <li> García, J. A. B., Concepción, C. G., & Piquero, J. C. M. (2006). Álgebra matricial para economía y empresa. Delta Publicaciones.</li>
                <li> Góngora, J. & Hernández, R. (1999). Estadística descriptiva. Trillas.</li>
                <li> Greene, W. H. (2000). Econometric analysis.</li>
                <li> Gujarati, D. (2011). Econometrics by Example. Palgrave Macmillan.</li>
                <li> Wooldridge, J. M. (2013). Introductory Econometrics: a Modern Approach. Cengage Learning.</li>
            <ul>
</div>
             """,unsafe_allow_html=True)
    
    st.write("----------------------------")

    st.markdown("""
<div class="justifyText">
            <p><h3>¡¡Gracias por usar esta app!!</h3><p>
            <p>Te agradezco por haber probado esta aplicación web, espero te haya gustado, que haya sido intuitiva y sobretodo espero que sea una herramienta muy útil en el aprendizaje de la econometría y en la creación de tus modelos de regresión lineal :)<P>
</div>
             """,unsafe_allow_html=True)
