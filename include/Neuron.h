#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

enum class ActivationType {
    RELU,
    SOFTMAX
};

class Neuron {
private:
    std::vector<double> weights;  // Weights for connections to the previous layer
    double bias;                  // Bias term
    double output;                // Output value after activation
    double delta;                 // Error delta for backpropagation
    ActivationType activationType; // Type of activation function used
    
    // Random number generation for weight initialization
    static std::random_device rd;
    static std::mt19937 gen;
    static std::normal_distribution<double> distribution;
    
public:
    // Constructor
    Neuron(int inputConnections, ActivationType type);
    
    // Forward pass computation
    void computeOutput(const std::vector<double>& inputs);
    
    // Activation functions
    double activate(double x) const;
    double activateDerivative(double x) const;
    
    // Getters and setters
    double getOutput() const;
    void setOutput(double val);
    
    double getDelta() const;
    void setDelta(double val);
    
    ActivationType getActivationType() const;
    
    // Backpropagation
    void updateWeights(const std::vector<double>& inputs, double learningRate);
    
    // For output layer neurons to calculate initial deltas
    void calculateOutputDelta(double target);
    
    // For hidden layer neurons to calculate deltas based on next layer
    void calculateHiddenDelta(const std::vector<Neuron>& nextLayer, size_t myIndex);
    
    // Get weights for a specific connection
    double getWeight(size_t index) const;
    
    // Get all weights
    const std::vector<double>& getWeights() const;
};

#endif // NEURON_H 