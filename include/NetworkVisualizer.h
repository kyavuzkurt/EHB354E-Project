#ifndef NETWORK_VISUALIZER_H
#define NETWORK_VISUALIZER_H

#include <SFML/Graphics.hpp>
#include <vector>
#include <memory>
#include "Network.h"

class NetworkVisualizer {
private:
    const Network* network;
    sf::Vector2f position;
    sf::Vector2f size;
    
    // Visual properties
    float neuronRadius;
    float layerSpacing;
    float neuronSpacing;
    
    // Colors
    sf::Color inputLayerColor;
    sf::Color hiddenLayerColor;
    sf::Color outputLayerColor;
    sf::Color neuronColor;
    sf::Color activeNeuronColor;
    sf::Color connectionColor;
    sf::Color activeConnectionColor;
    
    // Font for labels
    sf::Font font;
    
    // Keep track of neuron positions for drawing connections
    std::vector<std::vector<sf::Vector2f>> neuronPositions;
    
    // Current activations
    std::vector<std::vector<double>> activations;
    
    // Add to the private members
    bool connectionsVisible;
    
public:
    NetworkVisualizer(const Network* networkPtr, const sf::Vector2f& pos, 
                     const sf::Vector2f& visualizerSize, const sf::Font& fontRef);
    
    // Update the visualization based on new activations
    void update(const std::vector<double>& input);
    
    // Draw the network visualization
    void draw(sf::RenderWindow& window) const;
    
    // Calculate positions for all neurons
    void calculateNeuronPositions();
    
    // Draw a single neuron
    void drawNeuron(sf::RenderWindow& window, const sf::Vector2f& pos, 
                   float activation, bool isOutput = false) const;
    
    // Draw connections between layers
    void drawConnections(sf::RenderWindow& window, size_t fromLayer, size_t toLayer) const;
    
    // Draw layer backgrounds
    void drawLayerBackground(sf::RenderWindow& window, size_t layerIdx, 
                            const sf::Vector2f& pos, const sf::Vector2f& size) const;
    
    // Add to the public methods
    void updateNetworkStructure();
    
    // Add to the public methods
    void setConnectionsVisible(bool visible);
    
    // Add this new method to the public section
    void updateWithActivations(const std::vector<std::vector<double>>& allActivations);
};

#endif // NETWORK_VISUALIZER_H 