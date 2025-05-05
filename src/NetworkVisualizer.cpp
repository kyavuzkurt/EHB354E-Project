#include "../include/NetworkVisualizer.h"
#include <iostream>
#include <iomanip>
#include <sstream>

NetworkVisualizer::NetworkVisualizer(const Network* networkPtr, const sf::Vector2f& pos, 
                                   const sf::Vector2f& visualizerSize, const sf::Font& fontRef)
    : network(networkPtr), position(pos), size(visualizerSize), connectionsVisible(false) {
    // Initialize visual properties
    neuronRadius = 8.0f;
    layerSpacing = 0.0f;  // Will be calculated based on network size
    neuronSpacing = 0.0f; // Will be calculated based on network size
    
    // Set colors with our new scheme
    inputLayerColor = sf::Color(173, 216, 230, 120);    // Light blue
    hiddenLayerColor = sf::Color(144, 238, 144, 120);   // Light green
    outputLayerColor = sf::Color(255, 182, 193, 120);   // Light pink
    neuronColor = sf::Color(70, 70, 70);                // Dark gray
    activeNeuronColor = sf::Color(255, 140, 0);         // Dark orange (accent)
    connectionColor = sf::Color(180, 180, 180, 100);    // Semi-transparent gray
    activeConnectionColor = sf::Color(255, 140, 0, 150); // Semi-transparent accent
    
    // Copy font reference
    font = fontRef;
    
    // Initial calculation of neuron positions
    calculateNeuronPositions();
}

void NetworkVisualizer::calculateNeuronPositions() {
    if (!network || network->getLayerCount() == 0) {
        neuronPositions.clear();
        activations.clear();
        return;
    }
    
    std::cout << "Calculating neuron positions for " << network->getLayerCount() << " layers" << std::endl;
    
    // Get layer information from the network
    const std::vector<Layer>& layers = network->getLayers();
    
    // Calculate spacing between layers
    layerSpacing = size.x / (layers.size() + 1);
    
    // Clear previous positions
    neuronPositions.clear();
    neuronPositions.resize(layers.size());
    
    // Initialize activations
    activations.clear();
    activations.resize(layers.size());
    
    // Calculate positions for each layer
    for (size_t i = 0; i < layers.size(); ++i) {
        const std::vector<Neuron>& neurons = layers[i].getNeurons();
        size_t neuronCount = neurons.size();
        
        // For the input layer, we show fewer neurons as representative
        if (i == 0) {
            // Show at most 15 neurons to avoid overcrowding
            neuronCount = std::min(size_t(15), neuronCount);
        }
        
        // Calculate spacing between neurons in this layer
        neuronSpacing = size.y / (neuronCount + 1);
        
        neuronPositions[i].resize(neuronCount);
        activations[i].resize(neuronCount, 0.0);
        
        for (size_t j = 0; j < neuronCount; ++j) {
            float x = position.x + (i + 1) * layerSpacing;
            float y = position.y + (j + 1) * neuronSpacing;
            neuronPositions[i][j] = sf::Vector2f(x, y);
        }
    }
    
    std::cout << "Neuron positions calculated. Layers: " << neuronPositions.size() << std::endl;
    for (size_t i = 0; i < neuronPositions.size(); ++i) {
        std::cout << "  Layer " << i << ": " << neuronPositions[i].size() << " neurons" << std::endl;
    }
}

void NetworkVisualizer::update(const std::vector<double>& input) {
    if (!network || network->getLayerCount() == 0) {
        return;
    }
    
    // Get activations for all layers
    std::vector<std::vector<double>> allActivations = network->getAllActivations(input);
    
    // Map these to our visualization layers
    // First, make sure our activations vector is the right size
    activations.resize(neuronPositions.size());
    
    for (size_t i = 0; i < neuronPositions.size(); ++i) {
        activations[i].resize(neuronPositions[i].size());
        
        // For each neuron in this layer of our visualization
        for (size_t j = 0; j < neuronPositions[i].size(); ++j) {
            // If this is within the actual network's layer size
            if (i < allActivations.size() && j < allActivations[i].size()) {
                activations[i][j] = allActivations[i][j];
            } else {
                // Otherwise, use a default value
                activations[i][j] = 0.0;
            }
        }
    }
}

void NetworkVisualizer::draw(sf::RenderWindow& window) const {
    if (!network || network->getLayerCount() == 0 || neuronPositions.empty()) {
        // Draw empty visualization placeholder with improved styling
        sf::RectangleShape placeholder(size);
        placeholder.setPosition(position);
        placeholder.setFillColor(sf::Color(240, 240, 250));
        placeholder.setOutlineColor(sf::Color(200, 200, 220));
        placeholder.setOutlineThickness(2.0f);
        
        window.draw(placeholder);
        
        // Draw text indicating no network with better formatting
        sf::Text noNetworkText;
        noNetworkText.setFont(font);
        noNetworkText.setString("No network layers created yet\n\nUse 'Add Hidden Layer' and\n'Add Output Layer' buttons");
        noNetworkText.setCharacterSize(16);
        noNetworkText.setFillColor(sf::Color(100, 100, 120));
        noNetworkText.setPosition(position.x + size.x / 2.0f - 120, position.y + size.y / 2.0f - 40);
        
        window.draw(noNetworkText);
        return;
    }
    
    const std::vector<Layer>& layers = network->getLayers();
    
    // Draw layer backgrounds
    for (size_t i = 0; i < layers.size(); ++i) {
        sf::Vector2f layerPos(position.x + i * layerSpacing, position.y);
        sf::Vector2f layerSize(layerSpacing, size.y);
        drawLayerBackground(window, i, layerPos, layerSize);
    }
    
    // Draw connections between layers
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        drawConnections(window, i, i + 1);
    }
    
    // Draw neurons for each layer
    for (size_t i = 0; i < neuronPositions.size(); ++i) {
        for (size_t j = 0; j < neuronPositions[i].size(); ++j) {
            float activation = 0.0f;
            if (i < activations.size() && j < activations[i].size()) {
                activation = activations[i][j];
            }
            
            bool isOutput = (i == neuronPositions.size() - 1);
            drawNeuron(window, neuronPositions[i][j], activation, isOutput);
            
            // Add labels for output neurons (digits 0-9)
            if (isOutput) {
                sf::Text label;
                label.setFont(font);
                label.setString(std::to_string(j));
                label.setCharacterSize(14);
                label.setFillColor(sf::Color::Black);
                
                // Center the text on the neuron
                sf::FloatRect textBounds = label.getLocalBounds();
                label.setOrigin(textBounds.left + textBounds.width / 2.0f,
                               textBounds.top + textBounds.height / 2.0f);
                label.setPosition(neuronPositions[i][j]);
                
                window.draw(label);
            }
        }
    }
    
    // After drawing all neurons, highlight the winning output neuron
    if (!activations.empty() && !activations.back().empty()) {
        auto& outputActivations = activations.back();
        auto maxIt = std::max_element(outputActivations.begin(), outputActivations.end());
        size_t winningIdx = std::distance(outputActivations.begin(), maxIt);
        
        if (winningIdx < neuronPositions.back().size()) {
            // Draw a larger highlight circle around the winning neuron
            sf::CircleShape highlight(neuronRadius * 1.5f);
            highlight.setOrigin(neuronRadius * 1.5f, neuronRadius * 1.5f);
            highlight.setPosition(neuronPositions.back()[winningIdx]);
            highlight.setFillColor(sf::Color::Transparent);
            highlight.setOutlineThickness(2.0f);
            highlight.setOutlineColor(sf::Color::Yellow);
            
            window.draw(highlight);
        }
    }
}

void NetworkVisualizer::drawLayerBackground(sf::RenderWindow& window, size_t layerIdx, 
                                          const sf::Vector2f& pos, const sf::Vector2f& layerSize) const {
    sf::RectangleShape background;
    background.setPosition(pos);
    background.setSize(layerSize);
    
    // Choose color based on layer type
    if (layerIdx == 0) {
        background.setFillColor(inputLayerColor);
    } else if (layerIdx == neuronPositions.size() - 1) {
        background.setFillColor(outputLayerColor);
    } else {
        background.setFillColor(hiddenLayerColor);
    }
    
    window.draw(background);
}

void NetworkVisualizer::drawNeuron(sf::RenderWindow& window, const sf::Vector2f& pos, 
                                  float activation, bool isOutput) const {
    sf::CircleShape neuron(neuronRadius);
    neuron.setOrigin(neuronRadius, neuronRadius);
    neuron.setPosition(pos);
    
    // Color based on activation
    sf::Color color = neuronColor;
    if (activation > 0.5f) {
        // Lerp between neuronColor and activeNeuronColor based on activation
        float t = (activation - 0.5f) * 2.0f; // Map 0.5-1.0 to 0.0-1.0
        color.r = static_cast<sf::Uint8>((1 - t) * neuronColor.r + t * activeNeuronColor.r);
        color.g = static_cast<sf::Uint8>((1 - t) * neuronColor.g + t * activeNeuronColor.g);
        color.b = static_cast<sf::Uint8>((1 - t) * neuronColor.b + t * activeNeuronColor.b);
    }
    
    neuron.setFillColor(color);
    
    // Add outline for output neurons
    if (isOutput) {
        neuron.setOutlineThickness(2.0f);
        neuron.setOutlineColor(sf::Color::Red);
    }
    
    window.draw(neuron);
    
    // Add activation value for output neurons
    if (isOutput) {
        sf::Text valueText;
        valueText.setFont(font);
        
        // Format the activation to 2 decimal places
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << activation;
        valueText.setString(ss.str());
        
        valueText.setCharacterSize(12);
        valueText.setFillColor(sf::Color::Black);
        
        // Position the text to the right of the neuron
        valueText.setPosition(pos.x + neuronRadius + 5, pos.y - 6);
        
        window.draw(valueText);
    }
}

void NetworkVisualizer::drawConnections(sf::RenderWindow& window, size_t fromLayer, size_t toLayer) const {
    if (!connectionsVisible || fromLayer >= neuronPositions.size() || toLayer >= neuronPositions.size()) {
        return;
    }
    
    const auto& fromNeurons = neuronPositions[fromLayer];
    const auto& toNeurons = neuronPositions[toLayer];
    
    // Draw lines between each neuron in the from layer to each neuron in the to layer
    for (const auto& fromPos : fromNeurons) {
        for (const auto& toPos : toNeurons) {
            sf::Vertex line[] = {
                sf::Vertex(fromPos, connectionColor),
                sf::Vertex(toPos, connectionColor)
            };
            
            window.draw(line, 2, sf::Lines);
        }
    }
}

// Add a public method to force recalculation
void NetworkVisualizer::updateNetworkStructure() {
    calculateNeuronPositions();
    
    // Initialize with zero activations
    if (!network || network->getLayerCount() == 0) {
        return;
    }
    
    // Create empty input vector with appropriate size (784 for MNIST)
    std::vector<double> emptyInput(784, 0.0);
    update(emptyInput);
    
    std::cout << "Network visualization updated with " << network->getLayerCount() 
              << " layers" << std::endl;
}

void NetworkVisualizer::setConnectionsVisible(bool visible) {
    connectionsVisible = visible;
}

// Implement the new method
void NetworkVisualizer::updateWithActivations(const std::vector<std::vector<double>>& allActivations) {
    if (!network || network->getLayerCount() == 0) {
        return;
    }
    
    // Map these to our visualization layers
    activations.resize(neuronPositions.size());
    
    for (size_t i = 0; i < neuronPositions.size(); ++i) {
        activations[i].resize(neuronPositions[i].size());
        
        // For each neuron in this layer of our visualization
        for (size_t j = 0; j < neuronPositions[i].size(); ++j) {
            // If this is within the actual network's layer size
            if (i < allActivations.size() && j < allActivations[i].size()) {
                activations[i][j] = allActivations[i][j];
            } else {
                // Otherwise, use a default value
                activations[i][j] = 0.0;
            }
        }
    }
    
    // Make sure connections are visible
    connectionsVisible = true;
    
    std::cout << "Visualization updated with direct activations" << std::endl;
    if (!allActivations.empty() && !allActivations.back().empty()) {
        std::cout << "Output layer activations: ";
        for (size_t i = 0; i < std::min(size_t(10), allActivations.back().size()); ++i) {
            std::cout << allActivations.back()[i] << " ";
        }
        std::cout << std::endl;
    }
} 