#include "../include/Input.h"

Input::Input(const sf::Vector2f& position, const sf::Vector2f& size) 
    : currentIndex(0), dataLoaded(false), gen(rd()) {
    
    // Setup the image display rectangle
    imageDisplay.setPosition(position);
    imageDisplay.setSize(size);
    imageDisplay.setFillColor(sf::Color::White);
    imageDisplay.setOutlineThickness(2);
    imageDisplay.setOutlineColor(sf::Color::Black);
    
    // Create an empty SFML image (28x28 pixels for MNIST)
    sfImage.create(28, 28, sf::Color::Black);
    
    // Create and apply texture
    imageTexture.loadFromImage(sfImage);
    imageDisplay.setTexture(&imageTexture);
}

bool Input::loadData(const std::string& filename, int maxSamples) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }
    
    images.clear();
    labels.clear();
    
    std::string line;
    int count = 0;
    
    while (std::getline(file, line) && (maxSamples == -1 || count < maxSamples)) {
        std::stringstream ss(line);
        std::string value;
        
        // Read the label (first value in the row)
        std::getline(ss, value, ',');
        int label = std::stoi(value);
        labels.push_back(label);
        
        // Read pixel values (remaining values in the row)
        std::vector<double> pixels;
        while (std::getline(ss, value, ',')) {
            double pixelValue = std::stod(value) / 255.0;
            pixels.push_back(pixelValue);
        }
        
        images.push_back(pixels);
        count++;
    }
    
    if (images.empty()) {
        std::cerr << "No images loaded from file" << std::endl;
        return false;
    }
    
    // Setup the distribution for random selection
    dis = std::uniform_int_distribution<>(0, images.size() - 1);
    
    dataLoaded = true;
    currentIndex = 0;
    updateImageDisplay();
    
    std::cout << "Loaded " << images.size() << " images from " << filename << std::endl;
    return true;
}

std::vector<double> Input::getCurrentImageVector() const {
    if (!dataLoaded || currentIndex >= images.size()) {
        return std::vector<double>(784, 0.0); // Return empty image if no data
    }
    
    return images[currentIndex];
}

int Input::getCurrentLabel() const {
    if (!dataLoaded || currentIndex >= labels.size()) {
        return -1; // Return invalid label if no data
    }
    
    return labels[currentIndex];
}

void Input::updateImageDisplay() {
    if (!dataLoaded || currentIndex >= images.size()) {
        std::cerr << "Cannot update image display: data not loaded or invalid index" << std::endl;
        return;
    }
    
    try {
        // Update the SFML image with current MNIST data
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int idx = y * 28 + x;
                
                // Make sure we don't go out of bounds
                if (idx < images[currentIndex].size()) {
                    // Convert grayscale value to color
                    unsigned char grayValue = static_cast<unsigned char>(images[currentIndex][idx] * 255);
                    sf::Color pixelColor(grayValue, grayValue, grayValue);
                    
                    sfImage.setPixel(x, y, pixelColor);
                }
            }
        }
        
        // Update the texture from the image
        if (!imageTexture.loadFromImage(sfImage)) {
            std::cerr << "Failed to load texture from image" << std::endl;
        }
        imageDisplay.setTexture(&imageTexture);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in updateImageDisplay: " << e.what() << std::endl;
    }
}

void Input::nextImage() {
    if (!dataLoaded || images.empty()) {
        return;
    }
    
    currentIndex = (currentIndex + 1) % images.size();
    updateImageDisplay();
}

void Input::prevImage() {
    if (!dataLoaded || images.empty()) {
        return;
    }
    
    currentIndex = (currentIndex == 0) ? images.size() - 1 : currentIndex - 1;
    updateImageDisplay();
}

void Input::randomImage() {
    if (!dataLoaded || images.empty()) {
        return;
    }
    
    currentIndex = dis(gen);
    updateImageDisplay();
}

void Input::draw(sf::RenderWindow& window) const {
    window.draw(imageDisplay);
}

size_t Input::getImageCount() const {
    return images.size();
}

bool Input::isDataLoaded() const {
    return dataLoaded;
} 