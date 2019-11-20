# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:16:43 2019

@author: Dledbe
"""
import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        
        
        self.synaptic_weights = 2 * np.random.random((3,1))-1
    
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1-x)
    
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
    
    def train(self, training_inputs, training_outputs, training_iter):
        for i in range(training_iter):
            output = self.think(training_inputs)
            error = training_outputs-output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments
            
if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)
    
    #input data
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1],
                                [1,1,0]])

    #output data
    training_outputs = np.array([[0],
                                  [1],
                                  [1],
                                  [0],
                                  [1]])
    neural_network.train(training_inputs, training_outputs, 9995)
    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)
    
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    print("New situation: input data = "+A+" "+B+" "+C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))
    
        