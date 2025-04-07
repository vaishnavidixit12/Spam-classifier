# Spam Classifier: EDA and Model Selection

Welcome to my Kaggle notebook, **"Spam Classifier: EDA and Model Selection"**, where I delve into the development of a machine learning model to classify SMS messages as either spam or ham (non-spam). This project encompasses a thorough exploratory data analysis (EDA) followed by the selection and evaluation of various classification models.

🔗 [View Notebook on Kaggle](https://www.kaggle.com/code/vaishnavidixit12/spam-classifier-eda-and-model-selection)

## Objective

The primary aim of this notebook is to build an effective spam classifier by:

1. Performing an in-depth exploratory data analysis to understand the characteristics of spam and ham messages.
2. Preprocessing the text data to make it suitable for machine learning algorithms.
3. Selecting and evaluating multiple models to determine the most effective classifier for this task.

## Dataset

The dataset utilized in this analysis is the **SMS Spam Collection Dataset**, which comprises a set of SMS messages labeled as either 'spam' or 'ham'. This dataset is widely used for text classification tasks and provides a solid foundation for building a spam detection system.

## Approach

The notebook is structured into the following key sections:

### 1. Exploratory Data Analysis (EDA)
- **Data Overview:** Examined the distribution of spam and ham messages to understand class imbalances.
- **Text Analysis:** Analyzed the length of messages, word frequency distributions, and common terms in both spam and ham messages.
- **Visualization:** Utilized word clouds and bar plots to visualize common words and their frequencies in the dataset.

###  2. Data Preprocessing
- **Text Cleaning:** Removed punctuation, stopwords, and performed stemming to standardize the text data.
- **Feature Extraction:** Converted text data into numerical features using techniques like Term Frequency-Inverse Document Frequency (TF-IDF).

###  3. Model Selection and Evaluation
- **Model Training:** Implemented various classification algorithms, including:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machines (SVM)
  - Random Forest
- **Performance Evaluation:** Assessed models using metrics such as accuracy, precision, recall, and F1-score to determine the most effective classifier.

## Results

The analysis led to the identification of a robust model capable of accurately distinguishing between spam and ham messages. Detailed performance metrics and comparisons are documented within the notebook.

## Tools & Libraries

- **Programming Language:** Python
- **Data Manipulation and Analysis:** Pandas, NumPy
- **Natural Language Processing:** NLTK, Scikit-learn
- **Visualization:** Matplotlib, Seaborn, WordCloud

## How to Use

To explore this notebook:

1. **Access the Notebook:** Visit the [notebook on Kaggle](https://www.kaggle.com/code/vaishnavidixit12/spam-classifier-eda-and-model-selection).
2. **Fork the Notebook:** Click on "Copy & Edit" to create your own editable copy.
3. **Run the Cells:** Execute the cells sequentially to replicate the analysis and model training.
4. **Experiment:** Modify the code and parameters to experiment with different preprocessing techniques or models.

## 💬 Feedback

I welcome any feedback, suggestions, or discussions to enhance this analysis further. Please feel free to leave your comments in the [notebook's comments section](https://www.kaggle.com/code/vaishnavidixit12/spam-classifier-eda-and-model-selection/comments) on Kaggle.

