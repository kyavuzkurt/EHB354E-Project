#ifndef BUTTON_H
#define BUTTON_H

#include <SFML/Graphics.hpp>
#include <string>
#include <functional>

class Button {
private:
    sf::RectangleShape shape;        // Button shape
    sf::Text text;                   // Button text
    const sf::Font* fontPtr;          // Font for text (pointer)
    
    bool isPressed;                  // State of the button
    bool isHovered;                  // Is mouse hovering over button
    
    sf::Color idleColor;             // Color when not interacting
    sf::Color hoverColor;            // Color when hovering
    sf::Color pressedColor;          // Color when pressed
    
    std::function<void()> callback;  // Function to call when clicked
    
public:
    // Constructor
    Button(const sf::Vector2f& position, const sf::Vector2f& size, 
           const std::string& text, const sf::Font* font, 
           std::function<void()> callback = nullptr);
    
    // Update the button state
    void update(const sf::Vector2i& mousePosition);
    
    // Check if mouse is over button
    bool isMouseOver(const sf::Vector2i& mousePosition) const;
    
    // Handle mouse press
    void handleMousePress();
    
    // Handle mouse release and trigger callback if appropriate
    void handleMouseRelease(const sf::Vector2i& mousePosition);
    
    // Draw the button
    void draw(sf::RenderWindow& window) const;
    
    // Set the callback function
    void setCallback(std::function<void()> func);
    
    // Set button text
    void setText(const std::string& newText);
    
    // Set button colors
    void setColors(const sf::Color& idle, const sf::Color& hover, const sf::Color& pressed);
    
    // Set button position
    void setPosition(const sf::Vector2f& position);
};

#endif // BUTTON_H 