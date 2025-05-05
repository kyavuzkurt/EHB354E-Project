#include "../include/Network.h"

Network::Network(double lr) : learningRate(lr), rng(rd()) {
    // Initialize random number generator
}

void Network::addLayer(size_t neuronCount, ActivationType type) {
    size_t inputsPerNeuron = 0;
    
    if (layers.empty()) {
        // This is the first layer (input layer)
        // For the MNIST dataset, each neuron will have 784 inputs (28x28 pixels)
        inputsPerNeuron = 784;
    } else {
        // For hidden/output layers, the number of inputs equals 
        // the number of neurons in the previous layer
        inputsPerNeuron = layers.back().getNeuronCount();
    }
    
    // Create and add the new layer
    layers.emplace_back(neuronCount, inputsPerNeuron, type);
}

std::vector<double> Network::forwardPropagate(const std::vector<double>& inputs) {
    if (layers.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    
    // Process through each layer
    std::vector<double> currentInputs = inputs;
    
    for (auto& layer : layers) {
        layer.forwardPropagate(currentInputs);
        currentInputs = layer.getOutputs();
    }
    
    // Return the final outputs (from the last layer)
    return currentInputs;
}

double Network::trainSingle(const std::vector<double>& inputs, const std::vector<double>& targets) {
    // Forward pass
    std::vector<double> outputs = forwardPropagate(inputs);
    
    // Calculate loss
    double loss = calculateLoss(outputs, targets);
    
    // Backward pass (backpropagation)
    
    // 1. Calculate deltas for output layer
    layers.back().calculateOutputLayerDeltas(targets);
    
    // 2. Calculate deltas for hidden layers, working backwards
    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; i--) {
        layers[i].calculateHiddenLayerDeltas(layers[i + 1]);
    }
    
    // 3. Update weights for all layers
    for (auto& layer : layers) {
        layer.updateWeights(learningRate);
    }
    
    return loss;
}

double Network::trainBatch(const std::vector<std::vector<double>>& batchInputs, 
                         const std::vector<std::vector<double>>& batchTargets) {
    if (batchInputs.size() != batchTargets.size()) {
        throw std::runtime_error("Number of inputs doesn't match number of targets in batch");
    }
    
    double totalLoss = 0.0;
    
    // Train on each sample in the batch
    for (size_t i = 0; i < batchInputs.size(); i++) {
        totalLoss += trainSingle(batchInputs[i], batchTargets[i]);
    }
    
    // Return average loss
    return totalLoss / batchInputs.size();
}

void Network::train(const std::string& trainFile, int epochs, int batchSize) {
    try {
        // Load training data
        auto [inputs, targets] = loadMNISTData(trainFile);
        
        if (inputs.empty() || targets.empty()) {
            std::cerr << "Error: No training data loaded from " << trainFile << std::endl;
            return;
        }
        
        std::cout << "Training on " << inputs.size() << " samples for " << epochs << " epochs..." << std::endl;
        
        // Create indices for shuffling
        std::vector<size_t> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Train for multiple epochs
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle the data
            std::shuffle(indices.begin(), indices.end(), rng);
            
            double epochLoss = 0.0;
            int numBatches = 0;
            
            // Process in batches
            for (size_t i = 0; i < inputs.size(); i += batchSize) {
                std::vector<std::vector<double>> batchInputs;
                std::vector<std::vector<double>> batchTargets;
                
                // Create batch
                size_t endIdx = std::min(i + batchSize, inputs.size());
                for (size_t j = i; j < endIdx; j++) {
                    size_t idx = indices[j];
                    batchInputs.push_back(inputs[idx]);
                    batchTargets.push_back(targets[idx]);
                }
                
                // Train on batch
                double batchLoss = trainBatch(batchInputs, batchTargets);
                epochLoss += batchLoss;
                numBatches++;
            }
            
            // Calculate average loss for the epoch
            epochLoss /= numBatches;
            
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                      << ", Loss: " << epochLoss << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during training: " << e.what() << std::endl;
    }
}

double Network::test(const std::string& testFile, int numSamples) {
    // Mark parameter as unused to silence compiler warning
    (void)numSamples;
    
    try {
        // Load test data
        auto [inputs, targets] = loadMNISTData(testFile);
        
        if (inputs.empty() || targets.empty()) {
            std::cerr << "Error: No test data loaded from " << testFile << std::endl;
            return 0.0;
        }
        
        int correct = 0;
        double totalLoss = 0.0;
        
        // Test each sample
        for (size_t i = 0; i < inputs.size(); i++) {
            // Forward pass
            std::vector<double> outputs = forwardPropagate(inputs[i]);
            
            // Get predicted digit
            int predicted = getMaxOutputIndex(outputs);
            
            // Get target digit
            int target = getMaxOutputIndex(targets[i]);
            
            // Check if prediction is correct
            if (predicted == target) {
                correct++;
            }
            
            // Calculate loss
            totalLoss += calculateLoss(outputs, targets[i]);
        }
        
        // Calculate accuracy and average loss
        double accuracy = static_cast<double>(correct) / inputs.size();
        double avgLoss = totalLoss / inputs.size();
        
        std::cout << "Test Accuracy: " << (accuracy * 100.0) << "%, Loss: " << avgLoss << std::endl;
        
        return accuracy;
    } catch (const std::exception& e) {
        std::cerr << "Exception during testing: " << e.what() << std::endl;
        return 0.0;
    }
}

int Network::predict(const std::vector<double>& input) {
    // Forward pass
    std::vector<double> output = forwardPropagate(input);
    
    // Return the predicted digit
    return getMaxOutputIndex(output);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
Network::loadMNISTData(const std::string& filename, int numSamples) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
    
    std::string line;
    int count = 0;
    
    while (std::getline(file, line) && (numSamples == -1 || count < numSamples)) {
        std::stringstream ss(line);
        std::string value;
        
        // Read the label (first value in the row)
        std::getline(ss, value, ',');
        int label = std::stoi(value);
        
        // Convert label to target vector
        std::vector<double> target = labelToTarget(label);
        
        // Read pixel values (remaining values in the row)
        std::vector<double> input;
        while (std::getline(ss, value, ',')) {
            // Normalize pixel values to [0,1]
            double pixelValue = std::stod(value) / 255.0;
            input.push_back(pixelValue);
        }
        
        inputs.push_back(input);
        targets.push_back(target);
        count++;
    }
    
    return {inputs, targets};
}

std::vector<double> Network::labelToTarget(int label) {
    // Create a target vector with 10 elements (for digits 0-9)
    std::vector<double> target(10, 0.0);
    
    // Set the element at index 'label' to 1.0
    if (label >= 0 && label < 10) {
        target[label] = 1.0;
    }
    
    return target;
}

int Network::getMaxOutputIndex(const std::vector<double>& output) const {
    // Find the index of the maximum value
    return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}

double Network::calculateLoss(const std::vector<double>& outputs, const std::vector<double>& targets) {
    // Calculate cross-entropy loss: -sum(target_i * log(output_i))
    double loss = 0.0;
    
    for (size_t i = 0; i < outputs.size(); i++) {
        // Add a small epsilon to prevent log(0)
        double clippedOutput = std::max(outputs[i], 1e-10);
        loss -= targets[i] * std::log(clippedOutput);
    }
    
    return loss;
}

size_t Network::getLayerCount() const {
    return layers.size();
}

const std::vector<Layer>& Network::getLayers() const {
    return layers;
}

std::vector<std::vector<double>> Network::getAllActivations(const std::vector<double>& input) const {
    std::vector<std::vector<double>> allActivations;
    
    if (layers.empty()) {
        return allActivations;
    }
    
    // Start with the input
    std::vector<double> currentInput = input;
    allActivations.push_back(currentInput);
    
    // Process through each layer
    for (auto& layer : layers) {
        // We need a non-const copy of the layer to call forwardPropagate
        Layer layerCopy = layer;
        layerCopy.forwardPropagate(currentInput);
        currentInput = layerCopy.getOutputs();
        allActivations.push_back(currentInput);
    }
    
    return allActivations;
} 