import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import pickle

# Constants
DATA_PATH = 'concertriccir2.csv'
RANDOM_STATE = 42

@st.cache_data
def load_data():
    """Load dataset from CSV."""
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, 0:2].values
    y = df.iloc[:, -1].values
    return X, y

def draw_meshgrid(X):
    """Create a mesh grid for plotting decision boundaries."""
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

def get_classifier(name):
    """Return a classifier based on user selection."""
    if name == 'Random Forest':
        n_estimators = st.sidebar.slider('Num Estimators', 10, 500, 100)
        max_features = st.sidebar.selectbox('Max Features', (None, 'sqrt', 'log2'))
        max_depth = st.sidebar.slider('Max Depth', 1, 50, None)
        bootstrap = st.sidebar.selectbox('Bootstrap', (True, False))
        return RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, bootstrap=bootstrap, random_state=RANDOM_STATE)
    elif name == 'Logistic Regression':
        C = st.sidebar.slider('C (Regularization parameter)', 0.01, 10.0, 1.0)
        max_iter = st.sidebar.slider('Max Iterations', 100, 1000, 200)
        return LogisticRegression(C=C, max_iter=max_iter, random_state=RANDOM_STATE)
    elif name == 'SVM':
        kernel = st.sidebar.selectbox('Kernel', ('linear', 'poly', 'rbf', 'sigmoid'))
        C = st.sidebar.slider('C (Regularization parameter)', 0.01, 10.0, 1.0)
        return SVC(kernel=kernel, C=C, random_state=RANDOM_STATE)
    elif name == 'KNN':
        n_neighbors = st.sidebar.slider('Num Neighbors', 1, 20, 5)
        return KNeighborsClassifier(n_neighbors=n_neighbors)
    elif name == 'Decision Tree':
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 10)
        min_samples_split = st.sidebar.slider('Min Samples Split', 2, 10, 2)
        return DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=RANDOM_STATE)
    elif name == 'Naive Bayes':
        var_smoothing = st.sidebar.slider('Var Smoothing', 1e-10, 1e-7, 1e-9, format="%.1e")
        return GaussianNB(var_smoothing=var_smoothing)
    elif name == 'Gradient Boosting':
        n_estimators = st.sidebar.slider('Num Estimators', 10, 500, 100)
        learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.1)
        max_depth = st.sidebar.slider('Max Depth', 1, 50, 3)
        return GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=RANDOM_STATE)

def tune_hyperparameters(clf, X_train, y_train):
    """Tune hyperparameters using GridSearchCV."""
    param_grid = {}
    if isinstance(clf, RandomForestClassifier):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': [None, 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],
            'bootstrap': [True, False]
        }
    elif isinstance(clf, GradientBoostingClassifier):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    # Add more parameter grids for other classifiers as needed

    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def plot_decision_boundary(X, y, clf):
    """Plot the decision boundary of the classifier."""
    XX, YY, input_array = draw_meshgrid(X)
    labels = clf.predict(input_array)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow', edgecolor='k', s=20)
    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    plt.title(f"Decision Boundary of {clf.__class__.__name__}")
    plt.colorbar(scatter)
    return fig

def display_metrics(y_test, y_pred):
    """Display evaluation metrics."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

@st.cache_resource
def save_model(_clf, filename):
    """Save the trained model to a file."""
    with open(filename, 'wb') as file:
        pickle.dump(_clf, file)

def load_model(filename):
    """Load a trained model from a file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

# Streamlit sidebar
st.sidebar.markdown("# Choose Classifier")
classifier_name = st.sidebar.selectbox(
    'Select Algorithm',
    ('Random Forest', 'Logistic Regression', 'SVM', 'KNN', 
     'Decision Tree', 'Naive Bayes', 'Gradient Boosting')
)

# Get and train classifier
clf = get_classifier(classifier_name)
clf = tune_hyperparameters(clf, X_train, y_train)  # Hyperparameter tuning
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Plot decision boundary
fig = plot_decision_boundary(X, y, clf)
st.pyplot(fig)

# Display metrics
display_metrics(y_test, y_pred)

# Save the model
if st.sidebar.button('Save Model'):
    save_model(clf, 'trained_model.pkl')
    st.success("Model saved successfully!")
