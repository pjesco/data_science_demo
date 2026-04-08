import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.title('KNN Iris Dashboard')
st.header('Spring 2026')
st.subheader('Prof. Greg Reis')

st.divider()

st.header('KNN Classifier')

iris = load_iris()

X = iris.data
y = iris.target
with st.form('Selection'):
    test_size = st.number_input('Select test size %', 1, 100)
    test_size /= 100.0

    random_state = st.slider('Select random state', 1, 50)

    n_neighbors = st.slider('Select n value', 1, 5)

    st.form_submit_button()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)  # learn from the training data2

col1, col2 = st.columns(2)
with col1:
    st.subheader('Predicted Values:')
    y_pred = model.predict(X_test)
    st.dataframe(y_pred)

with col2:
    st.subheader('Test Values:')
    st.dataframe(y_test)
accuracy = (y_pred == y_test).mean()
st.subheader(f'Accuracy: {accuracy}')
