import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.random.seed(12)  #Set a fixed seed for reproducibility

#Loads the dataset with handling different exceptions of different errors that could happen when loading the file
def load_dataset(path):
    try:
        print("Loading dataset...")
        dataset = pd.read_excel(path, header=0, engine="openpyxl")  # Removes the first row it is a header
        print("Dataset Loaded Successfully")
        print(f"Dataset Shape: {dataset.shape}")
        return dataset

    except ImportError:
        print("Missing 'openpyxl'. Install it with 'pip install openpyxl'.")
        return None
    except FileNotFoundError:
        print(f"File not found at path: {path}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


class NeuralNetwork:
    #Constructor (initializtions)
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initializes weights and biases
        self.weights_input_hidden = np.random.randn(hidden_size, input_size) * 0.1
        self.bias_hidden = np.zeros((hidden_size, 1))
        self.weights_hidden_output = np.random.randn(output_size, hidden_size) * 0.1
        self.bias_output = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #Differentiation of the sigmoid
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    #forward propagation done from scratch
    def forward_propagation(self, X):
        self.input_layer = X
        self.hidden_input = np.dot(self.weights_input_hidden, self.input_layer) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.output_input = np.dot(self.weights_hidden_output, self.hidden_output) + self.bias_output
        self.output = self.output_input

        return self.output

     #Back propagation is applied
    def backward_propagation(self, X, y):
        m = X.shape[1]


        output_error = self.output - y
        output_delta = output_error / m


        hidden_error = np.dot(self.weights_hidden_output.T, output_delta)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)


        self.weights_hidden_output -= self.learning_rate * np.dot(output_delta, self.hidden_output.T)
        self.bias_output -= self.learning_rate * np.sum(output_delta, axis=1, keepdims=True)

        self.weights_input_hidden -= self.learning_rate * np.dot(hidden_delta, self.input_layer.T)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_delta, axis=1, keepdims=True)

    def train(self, X_train, y_train):

        for epoch in range(self.epochs):
            mse = 0
            for i in range(X_train.shape[0]):
                x = X_train[i].reshape(-1, 1)
                y = y_train[i].reshape(-1, 1)

                # Perform forward and backward propagation
                prediction = self.forward_propagation(x)
                self.backward_propagation(x, y)

                mse += np.mean((prediction - y) ** 2)

            mse /= X_train.shape[0]
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, MSE: {mse:.4f}")

 #function to predict the output when the user enters new data
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            x = X[i].reshape(-1, 1)
            prediction = self.forward_propagation(x)
            predictions.append(prediction.flatten())
        return np.array(predictions)


# Loads data a specific path
dataset_path = "C:/Users/hazem/Downloads/concrete_data (3).xlsx"
dataset = load_dataset(dataset_path)

#Exception is handeled is the dataset is empty
if dataset is None or dataset.empty:
    raise ValueError("Dataset is empty or could not be loaded.")

# Seperates the 4 features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.reshape(-1, 1)

# Normalizes the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Split data into training and testing sets 75% and 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the data using neural network
input_size = X_train.shape[1]
hidden_size = 5
output_size = 1
learning_rate = 0.3
epochs = 1000

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, epochs)
nn.train(X_train, y_train)

# Applies prediction to test the model
predictions = nn.predict(X_test)

# MSE for normalized data
mse_normalized = np.mean((y_test - predictions) ** 2)
print(f"Test MSE (Normalized Data): {mse_normalized:.4f}")

# Inverse transform predictions
y_test_original = scaler_y.inverse_transform(y_test)
predictions_original = scaler_y.inverse_transform(predictions)

# MSE for original data
mse_original = np.mean((y_test_original - predictions_original) ** 2)
print(f"Test MSE (Original Data): {mse_original:.4f}")


#Part for the user to be able to enter new data and the machine learning model outputs an answer based on what it has learned
user_choice = input("Do you want to input values for prediction? (yes/no): ").strip().lower()

if user_choice == 'yes':
    print("Please enter values for the following features:")
    columns = dataset.columns[:-1]  # Removes the target column
    user_input = []

    for col in columns:
        while True:
            try:
                value = float(input(f"Enter value for {col}: "))
                user_input.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    # Normalizes the input of the user
    user_input_scaled = scaler_X.transform([user_input])

    # Predicts target feature (cement content)
    predicted_cement_scaled = nn.predict(user_input_scaled)
    predicted_cement = scaler_y.inverse_transform(predicted_cement_scaled.reshape(-1, 1))

    print(f"Predicted Cement Content: {predicted_cement[0][0]}")
else:
    print("You chose not to input values. Exiting the program.")