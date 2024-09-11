# Machine Learning Classifier Dashboard

## Overview
The Machine Learning Classifier Dashboard is an interactive web application built using Streamlit. It allows users to visualize and compare different machine learning classifiers on a dataset, providing an intuitive interface for hyperparameter tuning, performance evaluation, and decision boundary visualization.

## Features
- **Multiple Classifiers**: Choose from a variety of classifiers including Random Forest, Logistic Regression, SVM, KNN, Decision Tree, Naive Bayes, and Gradient Boosting.
- **Hyperparameter Tuning**: Optimize classifier performance through an interactive interface for adjusting hyperparameters like the number of estimators, maximum depth, regularization, and more.
- **Visualization**: Visualize decision boundaries of classifiers and the distribution of data points, providing insights into how classifiers separate different classes.
- **Evaluation Metrics**: Evaluate classifiers using accuracy, precision, recall, F1-score, and confusion matrix to assess their performance.
- **Model Saving**: Save trained models for future use, enabling easy reuse and deployment.

## Requirements
Ensure that the following packages are installed:

- Python 3.6 or higher
- Streamlit
- Matplotlib
- Scikit-learn
- Pandas
- NumPy

Install the required packages using pip:

```bash
pip install streamlit matplotlib scikit-learn pandas numpy
```

## Dataset
The application requires a dataset named `concentriccir2.csv` with features in the first two columns and the target label in the last column. Place this file in the same directory as the application script.

## Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sarthaknimbalkar/ML-Visualizer.git
   cd ML-Visualizer
   ```

2. **Run the Application**:
   Start the Streamlit application by running:
   ```bash
   streamlit run app.py
   ```
   Replace `app.py` with the name of your script if different.

3. **Access the Application**:
   Open your browser and navigate to `http://localhost:8501` to use the dashboard.

## Usage
1. **Select a Classifier**: Use the sidebar to choose a classifier from the dropdown menu. Options include Random Forest, Logistic Regression, SVM, KNN, Decision Tree, Naive Bayes, and Gradient Boosting.

2. **Tune Hyperparameters**: Adjust the classifier's hyperparameters using sliders and dropdowns in the sidebar. Parameters will vary depending on the selected classifier (e.g., number of estimators for Random Forest).

3. **Visualize Decision Boundary**: The decision boundary will be plotted, showing how the classifier separates different classes. Data points will be colored based on their class labels.

4. **Evaluate Metrics**: The application will display key metrics such as accuracy, precision, recall, F1-score, and a confusion matrix for the selected classifier on the test dataset.

5. **Save Model**: Click "Save Model" to save the trained model using pickle, which can be reloaded for future use.

## Contributing
Contributions are welcome! If you have ideas for enhancements or new features, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments
- [Streamlit](https://streamlit.io/) for the interactive web application framework.
- [Scikit-learn](https://scikit-learn.org/stable/) for providing the machine learning algorithms and evaluation tools.
- [Matplotlib](https://matplotlib.org/) for visualization and plotting tools.
