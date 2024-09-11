# Machine Learning Classifier Dashboard

## Explanation of Sections:
- **Overview**: A brief description of what the project does.
- **Features**: Key functionalities of the application.
- **Requirements**: List of necessary packages and how to install them.
- **Dataset**: Information about the dataset used in the application.
- **Getting Started**: Instructions for cloning the repository and running the application.
- **Usage**: Step-by-step guide on how to use the application.
- **Contributing**: Information for users who want to contribute to the project.
- **License**: Licensing information.
- **Acknowledgments**: Credits to libraries and tools used in the project.


## Overview
This project is a web-based application built using Streamlit that allows users to visualize and compare different machine learning classifiers on a dataset. The application provides an interactive interface for selecting classifiers, tuning their parameters, and viewing their decision boundaries and accuracy.

## Features
- **Multiple Classifiers**: Choose from various classifiers including Random Forest, Logistic Regression, SVM, KNN, Decision Tree, Naive Bayes, and Gradient Boosting.
- **Interactive Parameter Tuning**: Adjust classifier parameters through sliders and dropdowns in the sidebar.
- **Visualization**: View decision boundaries of the selected classifier and the distribution of data points.
- **Accuracy Display**: See the accuracy of the chosen classifier on the test dataset.

## Requirements
To run this application, you need to have the following packages installed:

- Python 3.6 or higher
- Streamlit
- Matplotlib
- Scikit-learn
- Pandas
- NumPy

You can install the required packages using pip:

```bash
pip install streamlit matplotlib scikit-learn pandas numpy
```

## Dataset
The application uses a dataset named `concertriccir2.csv`. Make sure this file is in the same directory as the application script. The dataset should contain features in the first two columns and the target label in the last column.

## Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sarthaknimbalkar/ML-Visualizer.git
   cd ML-Visualizer
   ```

2. **Run the Application**:
   Start the Streamlit application by running the following command in your terminal:
   ```bash
   streamlit run app.py
   ```
   Replace `app.py` with the name of your Python script if it's different.

3. **Access the Application**:
   Open your web browser and go to `http://localhost:8501` to view the application.

## Usage
1. Select a classifier from the sidebar.
2. Adjust the parameters using the provided sliders and dropdowns.
3. The decision boundary will be displayed on the plot along with the data points.
4. The accuracy of the classifier will be shown below the plot.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Streamlit](https://streamlit.io/) - For creating the interactive web application framework.
- [Scikit-learn](https://scikit-learn.org/stable/) - For providing the machine learning algorithms.
- [Matplotlib](https://matplotlib.org/) - For data visualization.

Feel free to modify any section to better fit your project specifics!#   M L - V i s u a l i z e r  
 #   M L - V i s u a l i z e r  
 