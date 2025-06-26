#!/bin/bash

# Automated IBKR Gateway Login Script
# Uses xdotool to automate the login process

echo "Starting automated IBKR Gateway login..."

# Wait for Gateway to start
sleep 10

# Install xdotool for automation
sudo apt-get update && sudo apt-get install -y xdotool

# Wait for Gateway window to appear
echo "Waiting for Gateway window..."
sleep 5

# Find Gateway window
WINDOW_ID=$(xdotool search --name "IB Gateway" | head -1)

if [ -z "$WINDOW_ID" ]; then
    echo "Gateway window not found, trying alternative names..."
    WINDOW_ID=$(xdotool search --name "IBGateway" | head -1)
fi

if [ -z "$WINDOW_ID" ]; then
    echo "Gateway window not found, trying class name..."
    WINDOW_ID=$(xdotool search --class "ibgateway" | head -1)
fi

if [ -n "$WINDOW_ID" ]; then
    echo "Found Gateway window: $WINDOW_ID"
    
    # Activate window
    xdotool windowactivate $WINDOW_ID
    sleep 2
    
    # Enter username
    echo "Entering username..."
    xdotool type --delay 100 "$IBKR_USERNAME"
    
    # Tab to password field
    xdotool key Tab
    sleep 1
    
    # Enter password
    echo "Entering password..."
    xdotool type --delay 100 "$IBKR_PASSWORD"
    
    # Press Enter to login
    sleep 1
    xdotool key Return
    
    echo "Login credentials submitted"
    
    # Wait for login to complete
    sleep 10
    
    # Try to enable API
    echo "Attempting to enable API..."
    
    # Look for API configuration
    xdotool key ctrl+alt+f
    sleep 2
    
    # Navigate to API settings (this may vary based on Gateway version)
    xdotool key Tab Tab Tab
    sleep 1
    xdotool key space  # Enable API checkbox
    sleep 1
    
    # Set port to 4002
    xdotool key Tab
    xdotool key ctrl+a
    xdotool type "4002"
    
    # Apply settings
    xdotool key Return
    
    echo "API configuration attempted"
    
else
    echo "Could not find Gateway window for automation"
    exit 1
fi

