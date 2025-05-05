#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include "Neuron.h"

class Layer {
private:
    std::vector<Neuron> neurons;
    size_t neuronCount;
    ActivationType activationType;
    std::vector<double> layerInputs; // Stores the most recent inputs to this layer
    
public:
    // Constructor - creates a layer with specified neurons and activation type
    Layer(size_t neuronCount, size_t inputsPerNeuron, ActivationType type);
    
    // Forward propagation through this layer
    void forwardPropagate(const std::vector<double>& inputs);
    
    // Apply softmax activation to the layer (for output layer)
    void applySoftmax();
    
    // Backpropagation for output layer
    void calculateOutputLayerDeltas(const std::vector<double>& targets);
    
    // Backpropagation for hidden layer
    void calculateHiddenLayerDeltas(const Layer& nextLayer);
    
    // Update weights after backpropagation
    void updateWeights(double learningRate);
    
    // Getters
    size_t getNeuronCount() const;
    const std::vector<Neuron>& getNeurons() const;
    std::vector<Neuron>& getNeurons();
    
    // Get all outputs from this layer
    std::vector<double> getOutputs() const;
    
    // Get activation type
    ActivationType getActivationType() const;
};

#endif // LAYER_H 