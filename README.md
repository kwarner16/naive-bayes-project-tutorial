<!-- hide -->
# Naive Bayes - Step by step guide
<!-- endhide -->

- Understand a new dataset.
- Process it by applying exploratory data analysis (EDA).
- Model the data using Naive Bayes.
- Analyze the results and optimize the model.

<how-to-start>
  
## 🌱 How to start this project

Follow the instructions below:

1. Create a new repository based on [machine learning project](https://github.com/4GeeksAcademy/machine-learning-python-template) by [clicking here](https://github.com/4GeeksAcademy/machine-learning-python-template/generate).
2. Open the newly created repository in Codespace using the [Codespace button extension](https://docs.github.com/en/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository).
3. Once the Codespace VSCode has finished opening, start your project by following the instructions below.

</how-to-start>

## 🚛 How to deliver this project

Once you have finished solving the exercises, be sure to commit your changes, push them to your repository, and go to 4Geeks.com to upload the repository link.

## 📝 Instructions

### Sentiment analysis

Naive Bayes models are very useful when we want to analyze sentiment, classify texts into topics or recommendations, as the characteristics of these challenges meet the theoretical and methodological assumptions of the model very well.

In this project you will practice with a dataset to create a review classifier for the Google Play store.

#### Step 1: Loading the dataset

The dataset can be found in this project folder under the name `playstore_reviews.csv`. You can load it into the code directly from the link:

```text
https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv
```

Or download it and add it by hand in your repository. In this dataset, you will find the following variables:

- `package_name`. Name of the mobile application (categorical)
- `review`. Comment about the mobile application (categorical)
- `polarity`. Class variable (0 or 1), being 0 a negative comment and 1, positive (categorical numeric)

#### Step 2: Text Processing

### Why can't we use plain text in Machine Learning?

Machine Learning algorithms cannot work directly with text: **they need numbers**. Therefore, we must convert the reviews into numerical representations. This process is called **text vectorization**.

One of the simplest and most effective techniques for this is the **Bag of Words model**, which is implemented in Python using CountVectorizer.

#### What does `CountVectorizer` do?

`CountVectorizer` transforms each review into a vector that indicates **how many times each word appears**. For example:

```text
Original comment: "I love this app"
Resulting vector: [1, 1, 1]  ← (once “I”, once “love”, once “this”)
```

Additionally, it allows removing **stop words** (like “of”, “the”, “and”) using the parameter `stop_words="english"`.

Now, the specific steps to prepare the data are as follows:

- Remove spaces and convert everything to lowercase:

    ```python
    df["review"] = df["review"].str.strip().str.lower()
    ```

- Drop the column that does not provide predictive information:

    ```python
    df = df.drop("package_name", axis=1)
    ```

- Split the data into training and testing sets:

    ```python
    from sklearn.model_selection import train_test_split

    X = df["review"]
    y = df["polarity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

- Vectorize the text using CountVectorizer:

    ```python
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words="english")

    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    ```

Once completed, the predictors will be ready to train the model.

#### Step 3: Build a naive bayes model

Start solving the problem by implementing a model, from which you will have to choose which of the three implementations to use: `GaussianNB`, `MultinomialNB` or `BernoulliNB`, according to what we have studied in the module. Try now to train it with the two other implementations and confirm if the model you have chosen is the right one.

#### Step 4: Optimize the previous model

After training the model in its three implementations, choose the best option and try to optimize its results with a random forest, if possible.

#### Step 5: Save the model

Store the model in the appropriate folder.

#### Step 6: Explore other alternatives

Which other models of the ones we have studied could you use to try to overcome the results of a Naive Bayes? Argue this and train the model.

> Note: We also incorporated the solution samples on `./solution.ipynb` that we strongly suggest you only use if you are stuck for more than 30 min or if you have already finished and want to compare it with your approach.

## 🚀 Make Your Work Visible

Now it’s your turn to share on **LinkedIn** what your model has learned from human language.

### What to Share?

Post a powerful insight or reflection that came from your analysis of app reviews. You can explore how people express themselves when writing feedback, which words most strongly predict negative opinions, or how your model “understands” sentiment from text.

You can also include a visualization, such as the most common words in negative reviews, or examples where your model made a surprising correct (or incorrect) prediction.

---

### ✨ Shareable Examples

> **Can your AI detect frustration?**  
> I trained a Naive Bayes classifier to detect sentiment in app reviews.  
> I discovered that words like *“bug”*, *“crashes”*, and *“ads”* appear disproportionately in negative comments.  
> The amazing part? The model achieved over 90% accuracy without “understanding” a single word.  
> Just pure statistics. 🤖💬


> **Emotion can be trained too**  
> Can an AI tell if you're happy with an app?  
> After training a classifier on over 50,000 reviews, I found that words like *“love”*, *“helpful”*, and *“easy”* are strong predictors of positive reviews.  
> With just a few lines of text, the model knows if you’re a fan or a hater.  
> #MachineLearning #NLP

## 🚛 How to deliver this project

Once you have completed the practical case, make sure to commit your changes, push them to your repository, and go to 4Geeks.com to submit the repository link.