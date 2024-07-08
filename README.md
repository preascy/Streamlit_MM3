ML-FLOW-TEST 
# Streamlit App Team-MM3
<p align="center">
  <img src="https://github.com/Zamancwabe/Streamlit_MM3"
</p>

## Overview

News ArticleClassifier Pro is an innovative solution developed to revolutionize news categorization process. By harnessing the power of machine learning and natural language processing, this system automates the tagging and categorization of news articles, freeing up the editorial team to focus on high-value tasks. Through a user-friendly Streamlit web application, News ArticleClassifier Pro enables efficient bulk processing, intuitive search and filtering, and performance monitoring to ensure optimal accuracy. By enhancing workflow efficiency, reducing errors, and improving content discovery for readers, this solution empowers Newstoday to maintain a competitive edge in the digital media landscape.

## Table of Contents
* [Project Overview](#project-overview)
* [Meet the Team](#meet-the-team)
* [Installation](#installation)
* [Usage Instructions](#usage-instructions)
* [Contact us](#contact-us)
  
## Meet the Team
Zamancwabe Makhathini (zamancwabemakhathini@gmail.com),
Sibukiso Nhlengethwa, (sibukisot@gmail.com),
Khadijah Khan(khadijah.nontokozokhan@gmail.com),
Prescilla Matuludi(matuludip@gmail.com)


## Installation
To get a local copy up and running, follow these simple steps:

### Prerequisites
- Python 3.7 or higher
- Anaconda for managing the environment

### Installation Steps
1. **Clone the repository:**
    ```bash
    git clone 6598
    cd Streamlit_MM3
    ```

2. **Create a conda environment:**
    ```bash
    conda create --name classifierapp_env python=3.8
    ```
    
3. **Activate the conda environment:**
   ```bash
   conda activate classifierapp_env
   ```
   
5. **Install the required packages:**
    ```bash
    conda install -file requirements.txt
    ```

6. **Download necessary NLTK data:**
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Usage Instructions
1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2. **Input a news article:** Paste the text of the news article into the text area provided in the app.

3. **Classify the article:** Click the "Classify" button to get the predicted category of the news article.
