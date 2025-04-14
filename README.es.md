<!-- hide -->
# Naive Bayes - Guía paso a paso
<!-- endhide -->

- Comprender un dataset nuevo.
- Procesarlo aplicando un análisis exploratorio (EDA).
- Modelar los datos utilizando un Naive Bayes.
- Analizar los resultados y optimizar el modelo.

## 🌱 Cómo iniciar este proyecto

Sigue las siguientes instrucciones:

1. Crea un nuevo repositorio basado en el [proyecto de Machine Learning](https://github.com/4GeeksAcademy/machine-learning-python-template) [haciendo clic aquí](https://github.com/4GeeksAcademy/machine-learning-python-template/generate).
2. Abre el repositorio creado recientemente en Codespace usando la [extensión del botón de Codespace](https://docs.github.com/es/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository).
3. Una vez que el VSCode del Codespace haya terminado de abrirse, comienza tu proyecto siguiendo las instrucciones a continuación.

## 🚛 Cómo entregar este proyecto

Una vez que hayas terminado de resolver el caso práctico, asegúrate de confirmar tus cambios, haz push a tu repositorio y ve a 4Geeks.com para subir el enlace del repositorio.

## 📝 Instrucciones

### Análisis de sentimientos

Los modelos Naive Bayes son muy útiles cuando queremos analizar sentimientos, clasificar textos en tópicos o recomendaciones, ya que las características de estos desafíos cumplen muy bien con los supuestos teóricos y metodológicos del modelo.

En este proyecto practicarás con un conjunto de datos para crear un clasificador de reseñas de la tienda de Google Play.

#### Paso 1: Carga del conjunto de datos

El conjunto de datos se puede encontrar en esta carpeta de proyecto bajo el nombre `playstore_reviews.csv`. Puedes cargarlo en el código directamente desde el sigiente enlace: 

```text
https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv
```
O descargarlo y añadirlo a mano en tu repositorio. En este conjunto de datos encontrarás las siguientes variables:

- `package_name`. Nombre de la aplicación móvil (categórico)
- `review`. Comentario sobre la aplicación móvil (categórico)
- `polarity`. Variable de clase (0 o 1), siendo 0 un comentario negativo y 1, positivo (categórico numérico)


#### Paso 2: Procesamiento del texto

### ¿Por qué no podemos usar texto plano en Machine Learning?

Los algoritmos de Machine Learning no pueden trabajar directamente con texto: **necesitan números**. Por eso, debemos convertir los comentarios (reviews) en representaciones numéricas. Este proceso se llama **vectorización del texto**.

Una de las técnicas más sencillas y efectivas para esto es el **modelo de bolsa de palabras (Bag of Words)**, que se implementa en Python con CountVectorizer.

#### ¿Qué hace `CountVectorizer`?

`CountVectorizer` transforma cada comentario en un vector que indica **cuántas veces aparece cada palabra**. Por ejemplo:

```text
Comentario original: "Me encanta esta app"
Vector resultante:    [1, 1, 1]  ← (una vez “me”, una vez “encanta”, una vez “esta”)
```

Además, permite eliminar las **palabras vacías** (como “de”, “la”, “y”) usando el parámetro `stop_words="english"`.

Ahora sí, los pasos concretos para preparar los datos son los siguientes:

- Eliminar espacios y convertir todo a minúsculas:

    ```python
    df["review"] = df["review"].str.strip().str.lower()
    ```

- Eliminar la columna que no aporta información predictiva:

    ```python
    df = df.drop("package_name", axis=1)
    ```

- Dividir los datos en entrenamiento y prueba:

    ```python
    from sklearn.model_selection import train_test_split

    X = df["review"]
    y = df["polarity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

- Vectorizar el texto usando CountVectorizer:

    ```python
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words="english")

    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    ```

Una vez hayamos terminado tendremos listas las predictoras para entrenar el modelo.


#### Paso 3: Construye un naive bayes

Comienza a resolver el problema implementando un modelo del que tendrás que elegir cuál de las tres implementaciones utilizar: `GaussianNB`, `MultinomialNB` o `BernoulliNB`, según lo que hemos estudiado en el módulo. Prueba ahora a entrenarlo con las dos otras implementaciones y confirma si el modelo que has elegido es el adecuado.

#### Paso 4: Optimiza el modelo anterior

Después de entrenar el modelo en sus tres implementaciones, elige la mejor opción y trata de optimizar sus resultados con un random forest, si es posible.

#### Paso 5: Guarda el modelo

Almacena el modelo en la carpeta correspondiente.

#### Paso 6: Explora otras alternativas

¿Qué otros modelos de los que hemos estudiado podrías utilizar para intentar superar los resultados de un Naive Bayes? Arguméntalo y entrena el modelo.

> Nota: También incorporamos muestras de solución en `./solution.ipynb` que te sugerimos honestamente que solo uses si estás atascado por más de 30 minutos o si ya has terminado y quieres compararlo con tu enfoque.


## 🚀 Haz visible tu trabajo

Ahora es tu turno de comunicar en tu **LinkedIn** lo que tu modelo aprendió del lenguaje humano.

### ¿Qué compartir?

Publicá una reflexión o insight poderoso que surja del análisis de reseñas. Puede ser sobre cómo la gente se expresa al escribir críticas, qué palabras predicen con mayor fuerza una opinión negativa, o cómo tu modelo logra entender el “sentimiento” detrás de las palabras.

También podés acompañarlo con una visualización, como las palabras más comunes en críticas negativas, o ejemplos donde tu modelo acertó (o falló) sorprendentemente.

---

### ✨ Ejemplos posteables

> **¿Tu IA detecta frustración?**  
> Entrené un modelo de clasificación con Naive Bayes para detectar sentimientos en reseñas de apps.  
> Descubrí que palabras como *“bug”*, *“crashes”*, *“ads”* aparecen de forma desproporcionada en comentarios negativos.  
> Lo increíble: el modelo acertó con más del 90% de precisión sin haber “entendido” ni una sola palabra.  
> Solo estadística pura. 🤖💬



> **La emoción también se entrena**  
> ¿Puede una IA saber si estás feliz con una app?  
> Después de entrenar un clasificador con más de 50.000 reseñas, descubrí que palabras como *“love”*, *“helpful”* y *“easy”* son predictores clave de reseñas positivas.  
> Con solo unas líneas de texto, el modelo sabe si sos fan o hater.  
> #MachineLearning #NLP


## 🚛 Cómo entregar este proyecto

Una vez que hayas terminado de resolver el caso práctico, asegúrate de confirmar tus cambios, haz push a tu repositorio y ve a 4Geeks.com para subir el enlace del repositorio.

