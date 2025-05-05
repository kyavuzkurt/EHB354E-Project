#include "../include/Layer.h"

Layer::Layer(size_t nCount, size_t inputsPerNeuron, ActivationType type) 
    : neuronCount(nCount), activationType(type) {
    
    // Create the neurons
    for (size_t i = 0; i < neuronCount; i++) {
        neurons.emplace_back(inputsPerNeuron, type);
    }
}

void Layer::forwardPropagate(const std::vector<double>& inputs) {
    // Store inputs for later use in backpropagation
    layerInputs = inputs;
    
    // Forward propagate through each neuron
    for (auto& neuron : neurons) {
        neuron.computeOutput(inputs);
    }
    
    // If this is an output layer with softmax, apply softmax activation
    if (activationType == ActivationType::SOFTMAX) {
        applySoftmax();
    }
}

void Layer::applySoftmax() {
    // Get all outputs before softmax
    std::vector<double> rawOutputs;
    double maxOutput = -std::numeric_limits<double>::max();
    
    for (const auto& neuron : neurons) {
        double output = neuron.getOutput();
        rawOutputs.push_back(output);
        
        // Track maximum output to prevent overflow
        if (output > maxOutput) {
            maxOutput = output;
        }
    }
    
    // Calculate softmax: exp(x_i - max) / sum(exp(x_j - max))
    double sumExp = 0.0;
    std::vector<double> expOutputs;
    
    for (double output : rawOutputs) {
        // Subtract max for numerical stability
        double expOutput = std::exp(output - maxOutput);
        expOutputs.push_back(expOutput);
        sumExp += expOutput;
    }
    
    // Apply softmax to each neuron
    for (size_t i = 0; i < neurons.size(); i++) {
        double softmaxOutput = expOutputs[i] / sumExp;
        neurons[i].setOutput(softmaxOutput);
    }
}

void Layer::calculateOutputLayerDeltas(const std::vector<double>& targets) {
    // Make sure we have the correct number of targets
    if (targets.size() != neurons.size()) {
        throw std::runtime_error("Number of targets doesn't match number of output neurons");
    }
    
    // Calculate delta for each output neuron
    for (size_t i = 0; i < neurons.size(); i++) {
        neurons[i].calculateOutputDelta(targets[i]);
    }
}

void Layer::calculateHiddenLayerDeltas(const Layer& nextLayer) {
    // For each neuron in this layer
    for (size_t i = 0; i < neurons.size(); i++) {
        // Calculate delta based on next layer's deltas and weights
        neurons[i].calculateHiddenDelta(nextLayer.getNeurons(), i);
    }
}

void Layer::updateWeights(double learningRate) {
    // Update weights for each neuron
    for (auto& neuron : neurons) {
        neuron.updateWeights(layerInputs, learningRate);
    }
}

size_t Layer::getNeuronCount() const {
    return neuronCount;
}

const std::vector<Neuron>& Layer::getNeurons() const {
    return neurons;
}

std::vector<Neuron>& Layer::getNeurons() {
    return neurons;
}

std::vector<double> Layer::getOutputs() const {
    std::vector<double> outputs;
    outputs.reserve(neurons.size());
    
    for (const auto& neuron : neurons) {
        outputs.push_back(neuron.getOutput());
    }
    
    return outputs;
}

ActivationType Layer::getActivationType() const {
    return activationType;
} 