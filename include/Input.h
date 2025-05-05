#ifndef INPUT_H
#define INPUT_H

#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

class Input {
private:
    std::vector<std::vector<double>> images; // Store all loaded images
    std::vector<int> labels;                 // Store labels for all images
    sf::RectangleShape imageDisplay;         // For displaying the current image
    sf::Texture imageTexture;                // Texture for the image
    sf::Image sfImage;                       // SFML Image object
    
    size_t currentIndex;                     // Index of currently displayed image
    bool dataLoaded;                         // Flag indicating if data is loaded
    
    // For random selection
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;
    
public:
    // Constructor
    Input(const sf::Vector2f& position, const sf::Vector2f& size);
    
    // Load MNIST data
    bool loadData(const std::string& filename, int maxSamples = -1);
    
    // Get vector representation of current image (for neural network input)
    std::vector<double> getCurrentImageVector() const;
    
    // Get label of current image
    int getCurrentLabel() const;
    
    // Convert image data to SFML displayable format
    void updateImageDisplay();
    
    // Select next image
    void nextImage();
    
    // Select previous image
    void prevImage();
    
    // Select random image
    void randomImage();
    
    // Draw the image
    void draw(sf::RenderWindow& window) const;
    
    // Get number of loaded images
    size_t getImageCount() const;
    
    // Check if data is loaded
    bool isDataLoaded() const;
};

#endif // INPUT_H 