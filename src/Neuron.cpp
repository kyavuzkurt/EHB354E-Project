#include "../include/Neuron.h"

// Initialize static random number generation members
std::random_device Neuron::rd;
std::mt19937 Neuron::gen(rd());
std::normal_distribution<double> Neuron::distribution(0.0, 0.1); // Xavier initialization approximation

Neuron::Neuron(int inputConnections, ActivationType type) : activationType(type), output(0.0), delta(0.0) {
    // Initialize weights with small random values (Xavier/He initialization principle)
    weights.resize(inputConnections);
    for (auto& weight : weights) {
        weight = distribution(gen);
    }
    
    // Initialize bias with small random value
    bias = distribution(gen);
}

void Neuron::computeOutput(const std::vector<double>& inputs) {
    // Check that input size matches weights size
    if (inputs.size() != weights.size()) {
        throw std::runtime_error("Input size doesn't match weights size in neuron");
    }
    
    // Compute weighted sum of inputs
    double sum = bias; // Start with the bias
    for (size_t i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * weights[i];
    }
    
    // Apply activation function
    output = activate(sum);
}

double Neuron::activate(double x) const {
    switch (activationType) {
        case ActivationType::RELU:
            // ReLU activation: max(0, x)
            return std::max(0.0, x);
            
        case ActivationType::SOFTMAX:
            // For Softmax, we just return the input because Softmax
            // operates on the entire layer, not individual neurons
            return x;
            
        default:
            throw std::runtime_error("Unknown activation type");
    }
}

double Neuron::activateDerivative(double x) const {
    switch (activationType) {
        case ActivationType::RELU:
            // Derivative of ReLU: 0 if x < 0, 1 if x > 0
            return x > 0 ? 1.0 : 0.0;
            
        case ActivationType::SOFTMAX:
            // For Softmax derivative, we handle it specially in the Network class
            // because it depends on all outputs in the layer
            return 1.0; 
            
        default:
            throw std::runtime_error("Unknown activation type");
    }
}

double Neuron::getOutput() const {
    return output;
}

void Neuron::setOutput(double val) {
    output = val;
}

double Neuron::getDelta() const {
    return delta;
}

void Neuron::setDelta(double val) {
    delta = val;
}

ActivationType Neuron::getActivationType() const {
    return activationType;
}

void Neuron::updateWeights(const std::vector<double>& inputs, double learningRate) {
    // Update all weights using gradient descent
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] += learningRate * delta * inputs[i];
    }
    
    // Update bias (bias can be considered as a weight with input 1.0)
    bias += learningRate * delta;
}

void Neuron::calculateOutputDelta(double target) {
    // For output neurons, delta is (target - output) * derivative of activation
    // This is simplified for cross-entropy loss with softmax, where the delta
    // is directly (target - output)
    delta = target - output;
}

void Neuron::calculateHiddenDelta(const std::vector<Neuron>& nextLayer, size_t myIndex) {
    // For hidden neurons, delta is the sum of (next_layer_deltas * weights) * derivative of activation
    double sum = 0.0;
    
    for (size_t i = 0; i < nextLayer.size(); i++) {
        sum += nextLayer[i].getDelta() * nextLayer[i].getWeight(myIndex);
    }
    
    // Multiply by derivative of our activation function
    delta = sum * activateDerivative(output);
}

double Neuron::getWeight(size_t index) const {
    if (index >= weights.size()) {
        throw std::out_of_range("Weight index out of range");
    }
    return weights[index];
}

const std::vector<double>& Neuron::getWeights() const {
    return weights;
} 