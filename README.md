# **Music Recommendation system using KNN**

This project is a Spotify music recommendation system that leverages a K-Nearest Neighbors (KNN) model to recommend songs based on user preferences or features of existing songs. The model is trained on a dataset containing various audio features and metadata of songs.

To use this project, you'll need to have Python installed along with several Python libraries. You can install the required libraries using the following command:

![image](https://github.com/user-attachments/assets/dbb9adbe-b0f6-4347-8950-b7e51fa3299f)

# Steps:
  **1. Load and Preprocess Data**
    Load the dataset, handle missing values, and preprocess it by selecting relevant numerical and categorical features. Encode categorical features and apply feature importance weights to numerical features.

    The Numerical Features are: 

    numerical_features = [
    'danceability', 'energy', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms']

    The Categorical Features are:

    categorical_features = ['track_artist', 'playlist_genre']
  
  **2. Train the Model**
    Standardize the numerical features, apply weights, and train the KNN model on a sampled dataset. Save the trained model, scaler, and feature columns for later use.

    feature_importance_weights = {
    'danceability': 1.2,
    'energy': 1.0,
    'loudness': 0.8,
    'mode': 1.0,
    'speechiness': 0.9,
    'acousticness': 0.7,
    'instrumentalness': 0.6,
    'liveness': 0.5,
    'valence': 1.1,
    'tempo': 0.9,
    'duration_ms': 0.8}

    categorical_weight = 2.0
  
  **3. Interactive Recommendation System**
    Use ipywidgets to create an interactive recommendation system. This allows users to input their preferences or select an existing song to get recommendations.

  ![image](https://github.com/user-attachments/assets/9a79b6f5-7c55-42de-a73d-1b77a40f40ae)


  **4. Save and Load Models**
    Save the trained model, scaler, and feature columns using joblib to facilitate easy loading and usage in the interactive recommendation system.

      joblib.dump(knn, 'knn_model.pkl')
      joblib.dump(scaler, 'scaler.pkl')
      joblib.dump(feature_columns, 'columns.pkl')

# **Dataset**
  The dataset used in this project is a CSV file containing various audio features and metadata of songs, such as danceability, energy, loudness, and more. It also includes categorical features like track artist and playlist genre.

  ![image](https://github.com/user-attachments/assets/010f072e-b144-4b78-bf1b-f25cebc67e4b)


  

# **Output**
  The user can provide input in one of two ways:
  
  i)  The user selects his/her preferred features and the system generates 10 songs based on the preferences
    
  ii) The user chooses a song and the system generates songs that match the features of the chosen song 
    
  ![image](https://github.com/user-attachments/assets/cb68bddc-4b74-495c-a706-89642e8a1836)

  All the input parameters song or features and the recommendations are saved to the file "recommendations.csv" 

  
