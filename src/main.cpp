#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include "../include/Neuron.h"
#include "../include/Layer.h"
#include "../include/Network.h"
#include "../include/Input.h"
#include "../include/Button.h"
#include "../include/NetworkVisualizer.h"

int main() {
    std::cout << "Starting application..." << std::endl;
    
    // Create the main window
    sf::RenderWindow window(sf::VideoMode(1024, 768), "Neural Network MNIST");
    window.setFramerateLimit(60);
    
    std::cout << "Window created" << std::endl;
    
    // Load font
    sf::Font font;
    if (!font.loadFromFile("resources/font.ttf")) {
        std::cerr << "Could not load font" << std::endl;
        return 1;
    }
    
    std::cout << "Font loaded" << std::endl;
    
    // Create network with learning rate 0.01
    Network network(0.01);
    std::cout << "Network created" << std::endl;
    
    // Create input display for MNIST data
    Input inputDisplay(sf::Vector2f(50, 50), sf::Vector2f(280, 280));
    std::cout << "Input display created" << std::endl;
    
    // Try to load the MNIST data with error handling
    try {
        bool loaded = inputDisplay.loadData("data/mnist_data_test.csv", 100);
        if (!loaded) {
            std::cerr << "Failed to load MNIST data" << std::endl;
        } else {
            std::cout << "MNIST data loaded successfully" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception while loading data: " << e.what() << std::endl;
    }
    
    // Create status text
    sf::Text statusText;
    statusText.setFont(font);
    statusText.setCharacterSize(18);
    statusText.setFillColor(sf::Color::Black);
    statusText.setPosition(50, 350);
    statusText.setString("Status: Ready to build network");
    
    // Create prediction text
    sf::Text predictionText;
    predictionText.setFont(font);
    predictionText.setCharacterSize(24);
    predictionText.setFillColor(sf::Color::Black);
    predictionText.setPosition(350, 50);
    predictionText.setString("Prediction: None");
    
    // First, create the visualizer before any buttons that use it
    NetworkVisualizer visualizer(&network, sf::Vector2f(450, 200), sf::Vector2f(400, 300), font);
    visualizer.updateNetworkStructure();  // Initialize visualization

    // Create buttons for neural network operations
    std::vector<Button> buttons;
    
    // Control panel buttons (left side)
    // Add hidden layer button
    buttons.emplace_back(
        sf::Vector2f(50, 400), sf::Vector2f(200, 40), 
        "Add Hidden Layer", &font, 
        [&]() {
            try {
                // Small layer for demo purposes
                network.addLayer(16, ActivationType::RELU);
                statusText.setString("Status: Added hidden layer with 16 neurons");
                std::cout << "Added hidden layer" << std::endl;
                
                // Update the visualization to reflect the new network structure
                visualizer.updateNetworkStructure();
            } catch (const std::exception& e) {
                std::cerr << "Error adding layer: " << e.what() << std::endl;
                statusText.setString("Status: Error adding layer: " + std::string(e.what()));
            }
        }
    );
    
    // Add output layer button
    buttons.emplace_back(
        sf::Vector2f(50, 450), sf::Vector2f(200, 40), 
        "Add Output Layer", &font, 
        [&]() {
            try {
                // Output layer has 10 neurons (one for each digit 0-9)
                network.addLayer(10, ActivationType::SOFTMAX);
                statusText.setString("Status: Added output layer with 10 neurons");
                std::cout << "Added output layer" << std::endl;
                
                // Update the visualization
                visualizer.updateNetworkStructure();
            } catch (const std::exception& e) {
                std::cerr << "Error adding output layer: " << e.what() << std::endl;
                statusText.setString("Status: Error adding output layer: " + std::string(e.what()));
            }
        }
    );
    
    // Add Build Network button
    buttons.emplace_back(
        sf::Vector2f(50, 500), sf::Vector2f(200, 40), 
        "Build Network", &font, 
        [&]() {
            try {
                if (network.getLayerCount() < 2) {
                    statusText.setString("Status: Add at least one hidden layer and output layer first");
                    return;
                }
                
                // Finalize the network structure
                statusText.setString("Status: Building network connections...");
                window.draw(statusText);
                window.display();  // Force update the display
                
                // Update the visualization with connections highlighted
                visualizer.updateNetworkStructure();
                visualizer.setConnectionsVisible(true);
                
                // Create a sample input to visualize the network structure
                std::vector<double> sampleInput(784, 0.1);  // Low activation for all inputs
                visualizer.update(sampleInput);
                
                statusText.setString("Status: Network built successfully!");
                std::cout << "Network structure finalized" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error building network: " << e.what() << std::endl;
                statusText.setString("Status: Error building network: " + std::string(e.what()));
            }
        }
    );
    
    // Train button
    buttons.emplace_back(
        sf::Vector2f(50, 550), sf::Vector2f(200, 40), 
        "Train (1 Epoch)", &font, 
        [&]() {
            try {
                if (network.getLayerCount() < 2) {
                    statusText.setString("Status: Add at least one hidden layer and output layer first");
                    return;
                }
                
                statusText.setString("Status: Training network (1 epoch), please wait...");
                window.draw(statusText);
                window.display();  // Force update the display
                
                // Train for just 1 epoch on a very small subset (50 samples)
                std::string trainFile = "data/mnist_data_train.csv";
                network.train(trainFile, 1, 10);  // 1 epoch, batch size 10
                
                statusText.setString("Status: Training complete!");
                std::cout << "Training completed successfully" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error during training: " << e.what() << std::endl;
                statusText.setString("Status: Error during training: " + std::string(e.what()));
            }
        }
    );
    
    // Test button
    buttons.emplace_back(
        sf::Vector2f(50, 600), sf::Vector2f(200, 40), 
        "Test (100 samples)", &font, 
        [&]() {
            try {
                if (network.getLayerCount() < 2) {
                    statusText.setString("Status: Add at least one hidden layer and output layer first");
                    return;
                }
                
                statusText.setString("Status: Testing network, please wait...");
                window.draw(statusText);
                window.display();  // Force update the display
                
                // Test on just 100 samples for quick results
                std::string testFile = "data/mnist_data_test.csv";
                double accuracy = network.test(testFile, 100);
                
                // Display the results
                std::string accuracyStr = std::to_string(accuracy * 100.0);
                // Truncate to 2 decimal places
                accuracyStr = accuracyStr.substr(0, accuracyStr.find(".") + 3);
                
                statusText.setString("Status: Test accuracy: " + accuracyStr + "%");
                std::cout << "Test completed with accuracy: " << accuracyStr << "%" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error during testing: " << e.what() << std::endl;
                statusText.setString("Status: Error during testing: " + std::string(e.what()));
            }
        }
    );
    
    // Image navigation buttons (center, below visualization)
    buttons.emplace_back(
        sf::Vector2f(500, 550), sf::Vector2f(80, 40), 
        "Prev", &font, 
        [&]() { inputDisplay.prevImage(); }
    );

    buttons.emplace_back(
        sf::Vector2f(590, 550), sf::Vector2f(80, 40), 
        "Next", &font, 
        [&]() { inputDisplay.nextImage(); }
    );

    // Random image button
    buttons.emplace_back(
        sf::Vector2f(680, 550), sf::Vector2f(120, 40), 
        "Random", &font, 
        [&]() { inputDisplay.randomImage(); }
    );

    // Predict button (prominent position)
    buttons.emplace_back(
        sf::Vector2f(500, 100), sf::Vector2f(120, 50), 
        "Predict", &font, 
        [&]() {
            try {
                if (network.getLayerCount() < 2) {
                    statusText.setString("Status: Add at least one hidden layer and output layer first");
                    return;
                }
                
                // Get current image and predict
                std::vector<double> input = inputDisplay.getCurrentImageVector();
                
                // First, get all activations for visualization
                std::vector<std::vector<double>> allActivations = network.getAllActivations(input);
                
                // Update the visualizer with these exact activations
                visualizer.updateWithActivations(allActivations);
                
                // Get the prediction (should match what's shown in visualization)
                int prediction = network.getMaxOutputIndex(allActivations.back());
                
                // Update prediction text
                predictionText.setString("Prediction: " + std::to_string(prediction));
                
                // Show actual label
                int actualLabel = inputDisplay.getCurrentLabel();
                statusText.setString("Status: Actual label: " + std::to_string(actualLabel));
                
                std::cout << "Predicted: " << prediction << ", Actual: " << actualLabel << std::endl;
                std::cout << "Output activations: ";
                for (double val : allActivations.back()) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "Error in prediction: " << e.what() << std::endl;
                statusText.setString("Status: Error in prediction: " + std::string(e.what()));
            }
        }
    );
    
    std::cout << "Entering main loop" << std::endl;
    
    // Set application color scheme
    sf::Color bgColor(240, 245, 255);           // Light blue-gray background
    sf::Color primaryColor(70, 130, 180);       // Steel blue for primary elements
    sf::Color secondaryColor(95, 158, 160);     // Cadet blue for secondary elements
    sf::Color accentColor(255, 140, 0);         // Dark orange for accent/highlights
    sf::Color textColor(50, 50, 50);            // Dark gray for text
    sf::Color panelColor(225, 235, 245);        // Slightly darker than bg for panels

    // Create a background panel for the control section
    sf::RectangleShape controlPanel;
    controlPanel.setPosition(30, 380);
    controlPanel.setSize(sf::Vector2f(240, 320));
    controlPanel.setFillColor(panelColor);
    controlPanel.setOutlineColor(sf::Color(200, 210, 220));
    controlPanel.setOutlineThickness(2);

    // Create a background panel for the visualization
    sf::RectangleShape visualizationPanel;
    visualizationPanel.setPosition(430, 160);
    visualizationPanel.setSize(sf::Vector2f(440, 380));
    visualizationPanel.setFillColor(panelColor);
    visualizationPanel.setOutlineColor(sf::Color(200, 210, 220));
    visualizationPanel.setOutlineThickness(2);

    // Create a background panel for the image display
    sf::RectangleShape imagePanel;
    imagePanel.setPosition(40, 40);
    imagePanel.setSize(sf::Vector2f(300, 300));
    imagePanel.setFillColor(sf::Color::White);
    imagePanel.setOutlineColor(primaryColor);
    imagePanel.setOutlineThickness(3);
    
    // Add a title at the top of the window
    sf::Text appTitle;
    appTitle.setFont(font);
    appTitle.setString("Neural Network MNIST Classifier");
    appTitle.setCharacterSize(24);
    appTitle.setStyle(sf::Text::Bold);
    appTitle.setFillColor(primaryColor);
    appTitle.setPosition(30, 10);

    // Add a separator line
    sf::RectangleShape headerSeparator;
    headerSeparator.setPosition(30, 45);
    headerSeparator.setSize(sf::Vector2f(964, 2));
    headerSeparator.setFillColor(primaryColor);

    // Update the network title position
    sf::Text networkTitle;
    networkTitle.setFont(font);
    networkTitle.setString("Neural Network Visualization");
    networkTitle.setCharacterSize(18);
    networkTitle.setFillColor(sf::Color::Black);
    networkTitle.setPosition(550, 170);

    // Update the layer label positions
    sf::Text inputLabel;
    inputLabel.setFont(font);
    inputLabel.setString("Input\nLayer");
    inputLabel.setCharacterSize(14);
    inputLabel.setFillColor(sf::Color::Black);
    inputLabel.setPosition(470, 510);

    sf::Text hiddenLabel;
    hiddenLabel.setFont(font);
    hiddenLabel.setString("Hidden\nLayers");
    hiddenLabel.setCharacterSize(14);
    hiddenLabel.setFillColor(sf::Color::Black);
    hiddenLabel.setPosition(550, 510);

    sf::Text outputLabel;
    outputLabel.setFont(font);
    outputLabel.setString("Output\nLayer");
    outputLabel.setCharacterSize(14);
    outputLabel.setFillColor(sf::Color::Black);
    outputLabel.setPosition(750, 510);
    
    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::MouseMoved) {
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                
                // Update all buttons
                for (auto& button : buttons) {
                    button.update(mousePos);
                }
            }
            else if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    // Handle button presses
                    for (auto& button : buttons) {
                        button.handleMousePress();
                    }
                }
            }
            else if (event.type == sf::Event::MouseButtonReleased) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                    
                    // Handle button releases
                    for (auto& button : buttons) {
                        button.handleMouseRelease(mousePos);
                    }
                }
            }
        }
        
        // Clear the window
        window.clear(bgColor);
        
        // Draw panels first
        window.draw(imagePanel);
        window.draw(controlPanel);
        window.draw(visualizationPanel);

        // Draw input display
        inputDisplay.draw(window);

        // Draw neural network visualization
        visualizer.draw(window);

        // Draw texts with updated colors
        statusText.setFillColor(textColor);
        predictionText.setFillColor(textColor);
        networkTitle.setFillColor(textColor);
        inputLabel.setFillColor(textColor);
        hiddenLabel.setFillColor(textColor);
        outputLabel.setFillColor(textColor);

        window.draw(statusText);
        window.draw(predictionText);
        window.draw(networkTitle);
        window.draw(inputLabel);
        window.draw(hiddenLabel);
        window.draw(outputLabel);

        // Draw all buttons
        for (const auto& button : buttons) {
            button.draw(window);
        }
        
        // Draw title and separator
        window.draw(appTitle);
        window.draw(headerSeparator);
        
        // Display the window contents
        window.display();
    }
    
    return 0;
}