"""
    Simple Streamlit webserver application for serving developed classification
    models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib, os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

# Data dependencies
import pandas as pd
import numpy as np

# Load your raw data 
raw_data_path = 'test.csv'
raw_data = pd.read_csv(raw_data_path)

# Load your pickled models
models = {
    "Naive Bayes": "NaiveBayes_model.pkl",
    "xgboost": "xgboost_model.pkl",
    "Random Forest": "RandomForest_model.pkl"
}

# Load vectorizer
vectorizer_path = "vectorizers.pkl"
with open(vectorizer_path, 'rb') as file:
    vectorizers = pickle.load(file)

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def plot_bar_graph(models, test_accuracy):
    # Plotting the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(models, test_accuracy, color=['blue', 'green', 'orange'])
    plt.xlabel('Models')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy of Top 3 Models')
    plt.ylim(95, 97)  # Adjust the y-axis limits if needed
    plt.grid(axis='y')

    # Adding the accuracy values on top of the bars
    for i, v in enumerate(test_accuracy):
        plt.text(i, v + 0.1, f'{v}%', ha='center', va='bottom', fontsize=12)

    # Display the plot in Streamlit
    st.pyplot(plt)

# The main function where we will build the actual app
def main():
    """News ArticleClassifier Pro with Streamlit """


    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News ArticleClassifier Pro")
    st.subheader("Analysing news articles")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = [ "Dataset Overview","Mission and Vision", "Prediction", "Information","Conclusion"]
    selection = st.sidebar.selectbox("Choose Option", options)
    
    # Building out the "Home" page
    if selection == "Dataset Overview":
        st.info("ArticleClassifier Pro App")
        st.markdown("Use this application to classify news articles using different machine learning models.")
        
        # Display the dataset
        st.subheader("Dataset Overview")
        st.dataframe(raw_data.head())

     # Building out the "The Team" page
    if selection == "Mission and Vision":
        st.info("Our Mission and Vision")
        st.markdown("""
        - **Mission :** As we continue to grow, we remain steadfast in our mission to use data science for good. 
        We are actively seeking new opportunities to collaborate with organizations and individuals who share our passion for creating a more sustainable and equitable world.

        Contact us today to learn more about how our data-driven solutions can help you achieve your goals.

        - **Vision:** We are confident that our team's expertise, combined with our proven track record, makes us the ideal partner for organizations seeking to harness the power of data for a better future.
                    
        """)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")

        # Adding model selection
        model_choice = st.sidebar.selectbox("Choose Machine Learning Model", list(models.keys()))

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_texts = []
            for column in vectorizers:
                vect_text = vectorizers[column].transform([news_text]).toarray()
                vect_texts.append(vect_text)
            vect_text = np.hstack(vect_texts)
 
            # Load the chosen model
            model_path = models[model_choice]
            model = load_model(model_path)

            # Make predictions
            prediction = model.predict(vect_text)

            # Mapping prediction to category name
            mapping = {0: 'business', 1: 'education', 2: 'entertainment', 3: 'sports', 4: 'technology'}
            category = mapping.get(prediction[0], "Unknown")

            # When model has successfully run, will print prediction
            st.success(f"Text Categorized as: {category}")
       
    # Building out the "Information" page
    elif selection == "Information":
        st.info("Information about the Models")
        st.markdown("""
        The following machine learning models are used in this application:
                    
        - **Naive Bayes :** is a family of simple yet effective probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
        - **A Random Forest :** is an ensemble learning method for classification, regression, and other tasks that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees
        - **XGBoost:** which stands for eXtreme Gradient Boosting, is a powerful and scalable machine learning library for gradient boosting algorithms.

        You can choose any of these models from the prediction page to classify your text.
        """)
    

    # Building out the "Insights" page
    elif selection == "model metrics":
        st.info("Insights about the Models")

        # Dummy performance metrics for each model (for example purposes)
        performance_metrics = {
            "Naive bayes": 0.96,
            "xgboost": 0.96,
            "Random Forest": 0.96
        }

        # Display the models and the comparison button
        st.markdown("### Model Performance Comparison")
        st.markdown("""
        The performance of different machine learning models used in this application is as follows:

        - **Naive Bayes:** 0.96
        - **xgboost:** 0.96
        - **Random Forest:** 0.96
        """)
        st.write("This section compares the performance of different machine learning models used in this application.")

    

    

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()