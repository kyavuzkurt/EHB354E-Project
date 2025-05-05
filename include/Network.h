#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <memory>
#include <cmath>
#include <iostream>
#include "Layer.h"

class Network {
private:
    std::vector<Layer> layers;
    double learningRate;
    
    // For shuffling training data
    std::random_device rd;
    std::mt19937 rng;
    
public:
    // Constructor
    Network(double learningRate = 0.01);
    
    // Add a layer to the network
    void addLayer(size_t neuronCount, ActivationType type);
    
    // Forward propagation through all layers
    std::vector<double> forwardPropagate(const std::vector<double>& inputs);
    
    // Train on a single sample
    double trainSingle(const std::vector<double>& inputs, const std::vector<double>& targets);
    
    // Train on a batch of samples
    double trainBatch(const std::vector<std::vector<double>>& batchInputs, 
                     const std::vector<std::vector<double>>& batchTargets);
    
    // Train on the entire dataset for multiple epochs
    void train(const std::string& trainFile, int epochs, int batchSize);
    
    // Test the network on a dataset
    double test(const std::string& testFile, int numSamples = -1);
    
    // Predict the digit for a single input
    int predict(const std::vector<double>& input);
    
    // Load MNIST data
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
    loadMNISTData(const std::string& filename, int numSamples = -1);
    
    // Convert a label (0-9) to a target vector for output
    std::vector<double> labelToTarget(int label);
    
    // Get the predicted digit (index of the largest output)
    int getMaxOutputIndex(const std::vector<double>& output) const;
    
    // Calculate cross-entropy loss for softmax outputs
    double calculateLoss(const std::vector<double>& outputs, const std::vector<double>& targets);
    
    // Get number of layers
    size_t getLayerCount() const;
    
    // Get layers
    const std::vector<Layer>& getLayers() const;
    
    // Get activations for all layers
    std::vector<std::vector<double>> getAllActivations(const std::vector<double>& input) const;
};

#endif // NETWORK_H 