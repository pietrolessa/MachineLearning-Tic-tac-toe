from sklearn.neural_network import MLPClassifier

# Configure the MLP classifier with one hidden layer of 100 neurons
classifier = MLPClassifier(
    solver="adam",             # Uses the Adam optimizer
    shuffle=True,              # Shuffles data at each iteration
    activation="relu",         # Uses the ReLU activation function
    batch_size="auto",         # Automatically sets batch size
    learning_rate="constant",  # Uses a constant learning rate
    hidden_layer_sizes=(100),  # Defines one hidden layer with 100 neurons
    # Uncommenting the line below increases test accuracy but reduces in-game performance
    # hidden_layer_sizes=(100, 100),
)
