import streamlit as st
import pickle 
import sklearn 


# Load the trained sentiment analysis model

loaded_LR_model=pickle.load(open(r"C:\Users\admin\sentiment analysis\trained_LR_model_sentiment.sav","rb")) 
loaded_vectorizer=pickle.load(open(r"C:\Users\admin\sentiment analysis\CountVectorizer.sav","rb"))  
# Define a function to predict the sentiment
def predict_sentiment(text):
    

    # Vectorize the preprocessed text using the loaded TF-IDF vectorizer
    text_vector = loaded_vectorizer.transform([text])

    # Make predictions using the loaded model
    sentiment_prediction = loaded_LR_model.predict(text_vector)

    return sentiment_prediction[0]   

        # Create the Streamlit web app
def main():
    st.title("Sentiment Analysis Web App")
    st.write("Enter some text, and we'll predict the sentiment.")

    # Get user input
    user_input = st.text_area("Enter text here:")

    if st.button("Predict"):
    
        if user_input.strip() != '':
            sentiment = predict_sentiment(user_input)
            st.write(f'Sentiment: {sentiment}')

        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()