import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

species = ['Setosa', 'Versicolor', 'Virginica']

st.title("Iris Flower Species Predictor")
st.markdown("### Explore predictions with ML model trained on the classic Iris dataset")

st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict Species"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.success(f"Predicted Species: **{species[prediction[0]]}**")

    # Probability distribution plot
    fig, ax = plt.subplots()
    ax.bar(species, prediction_proba[0], color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

    st.markdown("#### Detailed Probabilities")
    for sp, prob in zip(species, prediction_proba[0]):
        st.write(f"{sp}: {prob * 100:.2f}%")

st.markdown("---")
st.markdown("**How it works?** This logistic regression model classifies iris flowers "
            "into one of three species based on four input measurements. "
            "Use the sliders to explore how changing measurements affect predictions.")
