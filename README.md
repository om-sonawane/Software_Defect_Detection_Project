# 🐞 Software Defect Detection Using SVM

Welcome to the **Software Defect Detection** project! This tool uses **Support Vector Machine (SVM)** to predict defects in software systems. The project is built using **Python** and comes with a user-friendly graphical interface designed with **Tkinter**. It aims to provide an efficient and accurate way to predict software defects based on historical data.

## 💡 Project Overview

The primary objective of this project is to detect potential software defects by analyzing various features using machine learning. The model is trained using the **SVM algorithm** from the `sklearn` library, which classifies software components as defective or not defective based on the input data.

### Key Features

- **Prediction Model**: Uses SVM for defect prediction.
- **Interactive UI**: Built with **Tkinter** for easy user interaction.
- **Data Handling**: Supports CSV file input for easy data loading using **Pandas**.
- **Visualization**: Displays results and data insights within the interface.
- **Preprocessing**: Data preprocessing (handling missing values, normalization, etc.) using **NumPy** and **Pandas**.

---

## 🛠️ Tech Stack & Libraries

The project leverages the following libraries and technologies:

- **Python**: Backend programming language.
- **Tkinter**: For building the graphical user interface.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Pillow**: For managing images in the UI.
- **scikit-learn (sklearn)**: For building the machine learning model (SVM).

---

## 🚀 Features

- **Software Defect Prediction**: Predict if a software component will likely be defective based on its features.
- **Interactive User Interface**: The easy-to-use interface built with Tkinter allows users to upload datasets and get predictions.
- **SVM Classification**: Leverages the power of Support Vector Machines for accurate classification.
- **Data Preprocessing**: Automatically cleans and prepares data for analysis.
- **Model Training & Testing**: Train the model using historical data and test it using new datasets.

---

## 📸 Screenshots

![Main Interface](https://via.placeholder.com/500x300)  
*Main interface of the software where the user can upload a dataset and view predictions.*

## 🎮 How to Use

### Prerequisites

Before running the project, make sure you have the following installed:

- **Python 3.x**
- **pip** (Python package installer)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/om-sonawane/software-defect-detection.git
## 📚 How the Model Works

- **Data Loading**: The model accepts a dataset in CSV format. The user uploads the data via the UI.
- **Preprocessing**: Missing values are handled, and features are scaled using **StandardScaler** from `sklearn`.
- **Training**: The **SVM model** is trained on a labeled dataset that contains historical defect information.
- **Prediction**: After training, the model predicts whether new software components will likely be defective.
- **Results**: Predictions are displayed in the UI with visual feedback on performance.

---

## 📊 Example Dataset

Here’s an example of how your dataset should look:

| Feature1 | Feature2 | Feature3 | Defect |
|----------|----------|----------|--------|
| 1.2      | 3.4      | 0.5      | 1      |
| 2.3      | 1.4      | 2.1      | 0      |
| ...      | ...      | ...      | ...    |

- **Feature1, Feature2, Feature3**: Attributes related to software components.
- **Defect**: Target label (1 = Defective, 0 = Non-defective).

---

## 🧠 SVM Model Details

The **Support Vector Machine (SVM)** algorithm is implemented using the `scikit-learn` library’s `SVC` class. The kernel used for the SVM model is **'linear'**, providing high performance for binary classification problems like defect detection.

### Model Training Example:

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Example: Training an SVM model
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

predictions = svm_model.predict(X_test)
```
## ✨ Future Enhancements

- **Cross-Validation**: Add cross-validation for better model evaluation.
- **Improved UI**: Enhance the user interface with additional features like real-time graph visualization.
- **More Algorithms**: Add support for other algorithms like Random Forest, KNN, and more.
- **Model Optimization**: Implement grid search for hyperparameter tuning of the SVM model.

---

## 🧑‍💻 Contribution

Contributions are welcome! If you’d like to improve the project or add new features, feel free to fork this repository, make your changes, and submit a pull request.

### How to Contribute:

1. **Fork the repository**
2. **Create a new branch** (`git checkout -b feature/my-feature`)
3. **Commit your changes** (`git commit -m 'Add some feature'`)
4. **Push to the branch** (`git push origin feature/my-feature`)
5. **Open a pull request**

---

## 🔗 Connect With Me

Feel free to connect with me on:

- [LinkedIn](https://www.linkedin.com/in/om-sonawane-23bab11b8/)
- [GitHub](https://github.com/om-sonawane)
- [Email](mailto:your-omsonawane03@gmail.com)


