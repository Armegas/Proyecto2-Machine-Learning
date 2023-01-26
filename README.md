# ***Proyecto2-Machine-Learning***
### Student: Nelson Alejandro Castro Andrews

# ![HenryLogo](https://transferencia.tec.mx/wp-content/uploads/2020/12/Machine-learning.jpeg)
​
# Proyecto individual 2
​
¡Ejecutando y poniendo en práctica mis habilidades en el campo de la predicción de datos. usare ciertas métricas para medir la performance del modelo y esta será usada para elegir los mejores modelos. El análisis predictivo emplea datos históricos para predecir eventos futuros. Normalmente, los datos históricos se utilizan para crear un modelo matemático que capture las tendencias importantes. Este modelo predictivo se usa entonces con los datos actuales para predecir lo que pasará a continuación, o bien para sugerir acciones que llevar a cabo con el fin de obtener resultados óptimos.

El análisis predictivo ha recibido mucha atención en los últimos años debido a los avances en la tecnología que lo respalda, especialmente en las áreas de big data y aprendizaje automático.

# ![aprendizaje_predictivo](https://es.mathworks.com/discovery/predictive-analytics/_jcr_content/mainParsys3/discoverysubsection_1475583375/mainParsys3/image_copy.adapt.1200.medium.svg/1611924128874.svg)
​
# ***Acerca de la actividad***
​
# ***El Aumento del big data***

A menudo se habla del análisis predictivo en el contexto del big data; los datos de ingeniería, por ejemplo, proceden de sensores, instrumentos y sistemas conectados del mundo real. Los datos de sistemas empresariales de una compañía podrían incluir datos de transacciones, resultados de ventas, quejas de clientes e información sobre marketing. Cada vez más, los negocios toman decisiones basadas en los datos procedentes de esta valiosa mina de información.

# ***Aumento de la competitividad***

Con el aumento en la competitividad, los negocios buscan una ventaja a la hora de proporcionar productos y servicios a mercados saturados. Los modelos predictivos basados en datos pueden ayudar a las empresas a resolver problemas de toda la vida de maneras nuevas.

Los fabricantes de equipos, por ejemplo, pueden encontrar difícil innovar en el hardware exclusivamente. Los desarrolladores de productos pueden agregar capacidades predictivas a las soluciones existentes para aumentar el valor de cara al cliente. El uso del análisis predictivo para el mantenimiento de equipos, o mantenimiento predictivo, puede anticipar fallos de los equipos, pronosticar las necesidades de energía y reducir los costes operativos. Por ejemplo, los sensores que miden las vibraciones de las piezas de automoción pueden indicar la necesidad de mantenimiento antes de que el vehículo falle en carretera.

Las empresas también utilizan el análisis predictivo para crear predicciones más precisas, tales como pronosticar la demanda de electricidad de la red de suministro. Estas predicciones permiten planificar los recursos (por ejemplo, la planificación de diversas plantas de energía) de manera más efectiva.

## ***Tecnologías de vanguardia para big data y aprendizaje automático: Con el propósito de extraer valor del big data, los negocios aplican algoritmos a grandes conjuntos de datos mediante herramientas como Hadoop y Spark. Las fuentes de datos pueden ser bases de datos transaccionales, archivos de registro de equipos, imágenes, vídeo, audio, datos de sensores u otros tipos de datos. La innovación a menudo surge de la combinación de los datos de diversas fuentes.*** ##
## ***Con todos estos datos, se necesitan herramientas para extraer conocimiento y tendencias. Las técnicas de aprendizaje automático se emplean para localizar patrones en los datos y crear modelos que pronostiquen los resultados futuros. Hay disponible una amplia gama de algoritmos de aprendizaje automático, incluidos regresión lineal y no lineal, redes neuronales, máquinas de vectores de soporte, árboles de decisión, etc.***


`Este proyecto es una instancia de evaluación INDIVIDUAL y OBLIGATORIO para los alumnos de Data Science de Henry. Es obligatorio que todos disponibilicen el código utilizado, para validar los modelos construidos.`
​
### **Premisas***: Sector de Mercado: inmobiliario 
​
Dentro de la sociedad globalizada e industrializada, es sabido que los precios de los inmuebles han presentado un constante cambio, por lo que quienes deseen invertir o vender una propiedad se enfrentan al fenómeno especulativo existente en la valorización de éstos. Esto, debido a la constante tendencia de las ciudades a crecer demográfica y comercialmente, llegando a un punto en donde no se tiene certeza de la valorización real dentro del sector en donde se desee invertir. 
​
Pese a que el precio depende, en cierta medida, de las tendencias que esté teniendo el mercado inmobiliario en un determinado tiempo, poder estimar adecuadamente el valor de una propiedad es una referencia clave para entender si es una buena oportunidad, ya sea de compra o de venta.
​
## ***Descripción del problema***
​
Usted ha sido contactado para el área de Machine Learning de una importante empresa inversora dentro del rubro de la inmobiliaria en Estados Unidos. 
​
El Team Lider le propone dos predicciones posibles, de las cuales puede elegir cuál realizar (o ambas si así lo quiere):
​
1. Implementar un modelo de clasificación con aprendizaje supervisado que permita clasificar el precio de las propiedades en venta, utilizando los datos que se han puesto a su disposición.
​
Para esto debe crear la columna `category_price`, en la cual se consideran las categorías
   * 'low': Para precios entre 0 y 999 dólares.
   * 'medium': Para precios entre 1000 y 1999 dólares.
   * 'high': Para precios desde 2000 dólares en adelante. 
​
    Considerando esta categorización, el objetivo es predecir si una propiedad pertenece a la categoría de precios bajos (low).

# ***Para Ello utilizaremos:***

`variable_new = [-math.inf,999, 1999,math.inf]`

`category2= ['low','medium','high']`

`df_t2= df.assign(category_price= pd.cut(x=df['price'], bins=valores, labels=category, include_lowest=True))`

​
2. Implementar un modelo de clasificación con aprendizaje no supervisado, utilizando clustering que agrupe las propiedades por segun las **3 categorias** a las que pueden pertenecer. Para ello, solo usaran el dataset de test provisto, eliminando previamente las caracteristicas que presenten nulos.

​# ![aprendizaje_predictivo](https://th.bing.com/th/id/OIP.Hv2dAxZeSKIxXzKLMOG33AHaE7?pid=ImgDet&w=770&h=513&rs=1)


## ***Disponibilizando***
​
El código se ha facilitado en un Jupyter Notebook .ipynb, el cual debe inclute un buen EDA, feature engineerging y, tambien un pipeline de Machine Learning para el procesamiento de datos que considere necesario. se **explicar claramente cada paso realizado** mediante comentarios en el texto formato markdown dentro del Notebook.
​
El repositorio esta disponible en github, bien ordenado y con un README inroducctorio al contenido.
​
Esta disponible un script dentro de data.ipynb que genera el archivo Armegas.csv sólo con las predicciones
teniendo únicamente **una sola columna** llamarda 'pred' conteniendo todos los valores de las predicciones, con un valor por fila.
​​
## ***Métrica a utilizar***
​
Utilice el metodo de aprendizaje supervisado, y la métrica `Accuracy` para las propiedades de precio bajo (low):
​
$$ Recall=\frac{TP+ TN}{TP+TN+FP+FN}$$
​
Donde $TP$ son los verdaderos positivos, $FP$ los falsos positivos, $FN$ los falsos negativos y $FN$ los falsos negativos. 
​
## **Criterio desarrollados**

- Entrenamiento y predicción utilizando un Modelo de Machine Learning adecuado al problema (clasificación o regresión).
- Análisis exploratorio de los datos (EDA).
- Repositorio de GitHub propio, con un readme redactado introduciendo al proyecto
- Comentarios y redacción con la fundamentación de la solución propuesta, escrita en Markdown en el Jupyter Notebook (.ipynb).
- Entregar ambos modelos, uno supervisado y otro no supervisado.

Se atendieron las siguientes Recomendaciones:
- División de dataset en train y test utilizando train_test_split, CV, KFold o similares.
- Utilización de Pipelines en la producción del modelo.

## Archivos provistos
​
 Link al dataset: https://drive.google.com/drive/folders/1nJ9ZMj6E6zh6McC9NwCA6KopfUIOG_1O
​
## Descripción de las dimensiones
- id: Identificador del anuncio. 
- url: Link web del anuncio.
- region: Región de Estados Unidos en donde se encuentra la propiedad.
- region_url: Link web de los anuncios pertenecientes a la región. 
- price: Precio de la propiedad en petrodólares.
- type: Tipo de propiedad.
- sqfeet: Metros cuadrados de la propiedad.
- beds: Cantidad de dormitorios.
- baths: Cantidad de baños.
- cats_allowed: Si se permiten gatos en la propiedad toma el valor 1, 0 para caso contrario.
- dogs_allowed: Si se permiten perros en la propiedad toma el valor 1, 0 para caso contrario.
- smoking_allowed: Si se permite fumar en la propiedad toma el valor 1, 0 para caso contrario.
- wheelchair_access: Si la propiedad posee acceso para sillas de ruedas toma el valor 1, 0 para caso contrario.
- electric_vehicle_charge: Si la propiedad posee cargador para vehículos eléctricos toma el valor 1, 0 para caso contrario.
- comes_furnished: Si la propiedad viene amueblada toma el valor 1, 0 para caso contrario.
- laundry_options: Opciones de lavandería (w/d in unit: Lavadora/secadora en la propiedad, w/d hookups: conexión para lavadora/secadora, laundry on site: servicio de lavandería en el lugar, laundry in bldg: servicio de lavandería en el edificio, no laundry on sit: sin servicio de lavandería).
- parking_options: Opciones de estacionamiento (off-street parking: zona de estacionamiento, attached garage: garaje incluido, carport: cochera/garaje abierto, detached garage: garaje separado, street parking: estacionamiento delimitado en la calle, no parking: sin estacionamiento, valet parking: estacionamiento con servicio valet).
- image_url: Link web de la imagen de la propiedad en el anuncio. 
- description: Descripción de la propiedad puesta en el anuncio. 
- lat: Latitud.
- long: Longitud.
- state: Código del estado al que pertenece la propiedad.


## Disclaimer del Alumno  
De parte de mi persona, quiero aclarar y remarcar que el fin de este proyecto propuesto es exclusivamente pedagógicos, con el objetivo de realizar proyectos que simulen un entorno laboral, en el cual se trabajo diversas temáticas ajustadas a la realidad.
 No reflejan necesariamente la filosofía y valores de mi persona. Además, Yo no aliento ni tampoco recomienda a los alumnos y/o cualquier persona leyendo los repositorios (y entregas de proyectos) que tomen acciones en base a los datos que pudieran o no haber recabado. Toda la información expuesta y los resultados obtenidos en los proyectos, nunca deben ser tomados en cuenta para la toma real de decisiones (especialmente en la temática de finanzas, salud, política, etc.).
