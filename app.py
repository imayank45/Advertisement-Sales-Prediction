import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

with open('predict.pkl','rb') as model_file:
    model = pickle.load(model_file)


def main():
    st.title('Advertisement Sales Prediction')

    tv = st.slider('TV Sales', 0.0, 1000.0, 500.0)
    radio = st.slider('Radio Sales', 0.0, 1000.0, 500.0)
    newspaper = st.slider('Newspaper Sales', 0.0, 1000.0, 500.0)


    if st.button('Predict'):

        input_data = np.array([[tv, radio, newspaper]])
        prediction = model.predict(input_data)[0]

        st.success(f'Predicted Sales: {prediction: .2f}')
        st.write('Developed By Mayank Kathane')

if __name__ == "__main__":
    main()
