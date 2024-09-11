# Machine Learning Classifier Dashboard

## Overview
The Machine Learning Classifier Dashboard is a web-based application developed with Streamlit that enables users to visualize and compare various machine learning classifiers on a dataset. This application provides an intuitive interface for selecting classifiers, tuning their parameters, and visualizing their decision boundaries along with performance metrics.

## Features
- **Multiple Classifiers**: Choose from classifiers including Random Forest, Logistic Regression, SVM, KNN, Decision Tree, Naive Bayes, and Gradient Boosting.
- **Interactive Parameter Tuning**: Adjust classifier parameters using sliders and dropdown menus in the sidebar.
- **Visualization**: View decision boundaries of the selected classifier and the distribution of data points.
- **Accuracy Display**: Display the accuracy of the selected classifier on the test dataset.

## Requirements
To run this application, you need the following packages:

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
The application utilizes a dataset named `concertriccir2.csv`. Ensure this file is located in the same directory as the application script. The dataset should have features in the first two columns and the target label in the last column.

## Getting Started

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/sarthaknimbalkar/ML-Visualizer.git
   cd ML-Visualizer
   ```

2. **Run the Application**:

   Start the Streamlit application with:

   ```bash
   streamlit run app.py
   ```

   (Replace `app.py` with the name of your Python script if different.)

3. **Access the Application**:

   Open your web browser and navigate to `http://localhost:8501` to view the application.

## Usage

1. Select a classifier from the sidebar.
2. Adjust the parameters using the available sliders and dropdown menus.
3. The decision boundary and data point distribution will be displayed on the plot.
4. The classifier’s accuracy will be shown below the plot.

## Contributing
We welcome contributions! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Streamlit](https://streamlit.io/) - For the interactive web application framework.
- [Scikit-learn](https://scikit-learn.org/stable/) - For the machine learning algorithms.
- [Matplotlib](https://matplotlib.org/) - For data visualization.

Feel free to adjust any sections to better fit your project's specifics!
