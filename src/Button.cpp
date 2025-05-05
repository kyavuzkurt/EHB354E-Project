#include "../include/Button.h"

Button::Button(const sf::Vector2f& position, const sf::Vector2f& size, 
               const std::string& buttonText, const sf::Font* buttonFont, 
               std::function<void()> buttonCallback)
    : shape(), 
      text(),
      fontPtr(buttonFont),      
      isPressed(false), 
      isHovered(false), 
      idleColor(sf::Color(70, 130, 180)),        // Steel blue
      hoverColor(sf::Color(100, 149, 237)),      // Cornflower blue
      pressedColor(sf::Color(65, 105, 225)),     // Royal blue
      callback(buttonCallback) 
{
    // Setup the button shape
    shape.setPosition(position);
    shape.setSize(size);
    
    shape.setFillColor(idleColor);
    shape.setOutlineThickness(2);
    shape.setOutlineColor(sf::Color(240, 248, 255)); // Alice blue
    
    // Setup the text
    if (fontPtr) {
        text.setFont(*fontPtr);
        text.setString(buttonText);
        text.setCharacterSize(18);  // Slightly larger text
        text.setFillColor(sf::Color::White);
        
        // Center the text in the button
        sf::FloatRect textBounds = text.getLocalBounds();
        text.setOrigin(textBounds.left + textBounds.width / 2.0f,
                      textBounds.top + textBounds.height / 2.0f);
        text.setPosition(position.x + size.x / 2.0f, position.y + size.y / 2.0f);
    }
}

void Button::update(const sf::Vector2i& mousePosition) {
    // Check if mouse is over button
    isHovered = isMouseOver(mousePosition);
    
    // Update button color based on state
    if (isPressed) {
        shape.setFillColor(pressedColor);
    }
    else if (isHovered) {
        shape.setFillColor(hoverColor);
    }
    else {
        shape.setFillColor(idleColor);
    }
}

bool Button::isMouseOver(const sf::Vector2i& mousePosition) const {
    sf::Vector2f buttonPos = shape.getPosition();
    sf::Vector2f buttonSize = shape.getSize();
    
    float mouseX = static_cast<float>(mousePosition.x);
    float mouseY = static_cast<float>(mousePosition.y);
    
    // Check if mouse is within button boundaries
    return (mouseX >= buttonPos.x && mouseX < buttonPos.x + buttonSize.x &&
            mouseY >= buttonPos.y && mouseY < buttonPos.y + buttonSize.y);
}

void Button::handleMousePress() {
    if (isHovered) {
        isPressed = true;
    }
}

void Button::handleMouseRelease(const sf::Vector2i& mousePosition) {
    // If button was pressed and mouse is still over it, execute callback
    if (isPressed && isMouseOver(mousePosition) && callback) {
        callback();
    }
    
    isPressed = false;
}

void Button::draw(sf::RenderWindow& window) const {
    window.draw(shape);
    window.draw(text);
}

void Button::setCallback(std::function<void()> func) {
    callback = func;
}

void Button::setText(const std::string& newText) {
    text.setString(newText);
    
    // Recenter the text
    sf::FloatRect textBounds = text.getLocalBounds();
    text.setOrigin(textBounds.left + textBounds.width / 2.0f,
                  textBounds.top + textBounds.height / 2.0f);
    
    sf::Vector2f buttonPos = shape.getPosition();
    sf::Vector2f buttonSize = shape.getSize();
    text.setPosition(buttonPos.x + buttonSize.x / 2.0f, buttonPos.y + buttonSize.y / 2.0f);
}

void Button::setColors(const sf::Color& idle, const sf::Color& hover, const sf::Color& pressed) {
    idleColor = idle;
    hoverColor = hover;
    pressedColor = pressed;
    
    // Update current color
    if (isPressed) {
        shape.setFillColor(pressedColor);
    }
    else if (isHovered) {
        shape.setFillColor(hoverColor);
    }
    else {
        shape.setFillColor(idleColor);
    }
}

void Button::setPosition(const sf::Vector2f& position) {
    shape.setPosition(position);
    
    // Update text position
    sf::Vector2f buttonSize = shape.getSize();
    text.setPosition(position.x + buttonSize.x / 2.0f, position.y + buttonSize.y / 2.0f);
} 