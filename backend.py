import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


#filter
i=0 
def load_dataset(filename):
    try:
        df = pd.read_csv(filename)
        print("Successfully loaded the dataset.\n")
        return df
    except FileNotFoundError as e:
        print(f"Error{e}: The file '{filename}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return None
    
def filter(df):
    try: 
        df_filtered = df[df['Language'].isin(['Kannada','English'])].copy()#case sentitive (try .lower())
        print("Sucessfully filtered.\n")
        return df_filtered
    
    except Exception as e:
        print(f"An error occured during filtering:{e}")
        return None
    
def plot_graph(for_data):
    global i
    i=i+1

    language_series = for_data['Language'].value_counts()
    languages = language_series.index.tolist()#The Index is the unique language name (e.g., 'English', 'Kannada')
    counts = language_series.values.tolist()#The Values are the counts (e.g., 1385, 369)

    fig,ax = plt.subplots(figsize=(20, 20)) # Creates a figure and a set of axes
    ax.bar(languages, counts, color=['black', 'green','blue'])
    ax.set_title('Language Sample Counts') # Title
    ax.set_xlabel('Language')              # X-axis label
    ax.set_ylabel('Number of Samples') # Y-axis label
    #saving
    os.makedirs('images_folder', exist_ok=True) 
    path=os.path.join("images_folder", f'graph_plot_{i}.png') #created path to save
    plt.savefig(path) #save the image


def main_func_to_filter():
    #loding...
    filename = "Language Detection.csv.zip"
    df = load_dataset(filename)

    if df is not None:
        df.info()
        print("\n")

    #filtering...
    filtered_df=filter(df)

    if filtered_df is not None:
        filtered_df.info()
        print("\n")

    filtered_df.to_csv("kannada_english_data.csv", index=False)#index=false for exclude row numbers
    print(f"Filtered data successfully saved \n")

    print(filtered_df.head().to_markdown(index=False, numalign="left", stralign="left") )#initially prints first five

    #ploting...
    plot_graph(df)
    plot_graph(filtered_df)
    return filtered_df


#train
def train_evalute_model(data):

    X = data['Text']
    y = data['Language']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 1. Define the Machine Learning Pipeline structure
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 5)
        )),
    
        # 2. CLASSIFIER WITH IMBALANCE CORRECTION
        ('classifier', LogisticRegression(
            class_weight='balanced',  # <--- THIS IS THE KEY PART
            max_iter=1000,
            random_state=42
        ))
    ])

    model_pipeline.fit(X_train, y_train_encoded)

    y_pred_encoded = model_pipeline.predict(X_test)
    target_names = le.classes_
    print("\nClassification Report (Balanced Model):")
    print(classification_report(y_test_encoded, y_pred_encoded, target_names=target_names))

    joblib.dump(model_pipeline, 'language_predictor_model.joblib')
    joblib.dump(le, 'label_encoder.joblib')


def main_func_to_train(filtered_df):
    train_evalute_model(filtered_df)

#predict
def predict_language(text):
    #load from joblib
    model_pipeline = joblib.load('language_predictor_model.joblib')
    le = joblib.load('label_encoder.joblib')

    numerical_prediction = model_pipeline.predict([text])
    language_label = le.inverse_transform(numerical_prediction)[0]
    return language_label

def main_func_to_predict():

    input_text=input("Enter the text to predict the language(kannda or english): ")
    ans=predict_language(input_text)
    print(f"Predicted language: {ans}")

#main func for all

if __name__ == "__main__":

    #data=main_func_to_filter()

    #main_func_to_train(data)
    
    main_func_to_predict()
        
    