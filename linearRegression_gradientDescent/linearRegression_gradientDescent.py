import numpy as np
from load_auto import load_auto
import matplotlib.pyplot as plt
import math

def initialize_parameters(observation_dimension):
    # observation_dimension: number of features taken into consideration of the input
    # returns weights as a vector and offset as a scalar
    weights = np.zeros((observation_dimension, 1))
    offset_b = 0
    return weights, offset_b

def model_forward(train_dataset, weights, offset_b):
    # train_dataset: input data points
    # weights and offset_b: model parameters
    # returns the output predictions as a vector corresponding to each input data point
    number_observations = np.size(train_dataset, axis = 1)
    predictions = np.zeros((1, number_observations))
    for observation in range(0, number_observations):
        with np.errstate(over='raise', invalid='raise'):
            try:
                predictions[0, observation] = weights.T @ train_dataset[:, observation] + offset_b
            except:
                predictions[0, observation] = np.inf
    return predictions

def compute_cost(predictions, train_labels):
    # predictions: computed output values
    # train_labels: true output values (ground truth)
    # returns the cost function value
    number_observations = np.size(predictions, axis = 1)
    sum = 0
    with np.errstate(over='raise', invalid='raise'):
        try:
            for observation in range(0, number_observations):
                sum += (train_labels[observation, 0] - predictions[0, observation])**2
        except:
            return np.inf
    return sum/number_observations

def model_backward(observation_dimension, train_dataset, predictions, train_labels):
    # observation_dimension: number of features taken into consideration of the input
    # train_dataset: input data points
    # predictions: computed output values
    # train_labels: true output values (ground truth)
    # returns the gradient of the cost function with respect to all parameters
    number_observations = np.size(train_dataset, axis = 1)
    sum_weights = np.zeros((observation_dimension, 1))
    sum_offset = 0
    for observation in range(0, number_observations):
        diff = predictions[0, observation] - train_labels[observation, 0]
        with np.errstate(over='raise', invalid='raise'):
            try:
                sum_weights += train_dataset[:, observation].reshape(observation_dimension,-1) * diff
                sum_offset += diff
            except:
                return np.full(sum_weights.shape, np.inf), np.inf
    gradient_weights = sum_weights * (2/number_observations)
    gradient_offset = sum_offset * (2/number_observations)
    return gradient_weights, gradient_offset

def update_parameters(weights, offset_b, gradient_weights, gradient_offset, learning_rate):
    # weights and offset_b: parameters computed (or initialised) in this iteration
    # gradient_weights and gradient_offset: gradients of the cost function
    # learning_rate: step size
    # returns the updated parameters for the next iteration
    updated_weights = weights - (learning_rate * gradient_weights)
    updated_offset = offset_b - (learning_rate * gradient_offset)
    return updated_weights, updated_offset

def predict(train_dataset, weights, offset_b):
    return model_forward(train_dataset, weights, offset_b)

def train_linear_model(train_dataset, train_labels, number_iterations, learning_rate):
    # train_dataset: input data points
    # train_labels: true output values (ground truth)
    # number_iterations and learning_rate: user-defined hyperparameters
    # returns the model parameters and cost function values as a vector
    cost = []
    observation_dimension = np.size(train_dataset, axis = 0)
    weights, offset_b = initialize_parameters(observation_dimension)
    while number_iterations > 0:
        predictions = predict(train_dataset, weights, offset_b)
        cost.append(compute_cost(predictions, train_labels))
        gradient_weights, gradient_offset = model_backward(observation_dimension, train_dataset, predictions, train_labels)
        weights, offset_b = update_parameters(weights, offset_b, gradient_weights, gradient_offset, learning_rate)
        number_iterations -= 1
    return weights, offset_b, cost

def plotting_cost_iteration(learning_rates, cost_consolidated):
    for counter in range(0, cost_consolidated.shape[0]):
        plt.plot(np.arange(start=1, stop = (cost_consolidated.shape[1] + 1), step= 1), cost_consolidated[counter,:], label=r'$\alpha = $' + str(learning_rates[counter]))
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost per Iteration')
    plt.ylim(0,720)
    plt.legend()
    plt.show()

def plotting_horsepower_mpg(train_dataset, train_labels, weights, offset_b):
    plt.scatter(train_dataset[0,:], train_labels[:,0], label='Data points')
    plt.plot(train_dataset[0,:], np.array(train_dataset[0,:]*weights + offset_b).reshape(train_labels.shape),'r-', label='Linear Regression')
    plt.xlabel('(normalised) Horsepower')
    plt.ylabel('MPG')
    plt.title('MPG vs (normalised) Horsepower')
    plt.legend()
    plt.show()

PATH_DATASET = '/Users/marcellovendruscolo/Documents/vscode-workspace/DeepLearningForImageAnalysis/linearRegression_gradientDescent/Auto.csv'
train_dataset, train_labels = load_auto(PATH_DATASET)
train_dataset = np.array(train_dataset)
non_normalised_dataset = np.array(np.transpose(train_dataset))
non_normalised_horsepower = non_normalised_dataset[2,:].reshape(1,-1)
train_labels = np.array(train_labels)

mean = np.mean(train_dataset, axis=0)
sd = np.std(train_dataset, axis=0)
for col in range(0, train_dataset.shape[1]):
    train_dataset[:,col] = (train_dataset[:,col] - mean[col])/sd[col]
normalised_dataset = np.transpose(train_dataset)
horsepower_dataset = normalised_dataset[2,:].reshape(1,-1)

# Exercise 1.4.1 and Exercise 1.4.2:
# learning_rate = 0.1
# number_iterations = 1000

# print('\nChoice of input dataset: (i) Only horsepower feature.')
# weights, offset_b, cost_function_value = train_linear_model(horsepower_dataset, train_labels, number_iterations, learning_rate)
# print('Number of iterations: ' +str(number_iterations))
# print('Learning rate: ' +str(learning_rate))
# print('Cost function value: ' +str(cost_function_value[len(cost_function_value) - 1]))
# print('Weights: ' +str(weights))
# print('Offset: ' +str(offset_b))

# print('\nChoice of input dataset: (ii) All features except name.')
# weights, offset_b, cost_function_value = train_linear_model(normalised_dataset, train_labels, number_iterations, learning_rate)
# print('Number of iterations: ' +str(number_iterations))
# print('Learning rate: ' +str(learning_rate))
# print('Cost function value: ' +str(cost_function_value[len(cost_function_value) - 1]))
# print('Weights: ' +str(weights))
# print('Offset: ' +str(offset_b) + '\n')

# Exercise 1.4.3:
# learning_rates = [1, 1e-1, 1e-2, 1e-3, 1e-4]
# number_iterations = 1000
# cost_consolidated = np.ndarray(shape=(len(learning_rates), number_iterations))

# for counter in range(0, len(learning_rates)):
#     weights, offset_b, cost_consolidated[counter,:] = train_linear_model(normalised_dataset, train_labels, number_iterations, learning_rates[counter])

# plotting_cost_iteration(learning_rates, cost_consolidated)

# Exercise 1.4.4:
# learning_rate = [1, 1e-1, 1e-2, 1e-3, 1e-4]
# number_iterations = 1000
# cost_consolidated = np.ndarray(shape=(len(learning_rate), number_iterations))
# for counter in range(0, len(learning_rate)):
#     weights, offset_b, cost_consolidated[counter,:] = train_linear_model(non_normalised_dataset, train_labels, number_iterations, learning_rate[counter])
# plotting_cost_iteration(learning_rate, cost_consolidated)

# Exercise 1.4.5:
# learning_rate = 0.1
# number_iterations = 1000
# weights, offset_b, cost_function_value = train_linear_model(horsepower_dataset, train_labels, number_iterations, learning_rate)
# plotting_horsepower_mpg(horsepower_dataset, train_labels, weights, offset_b)