#!/bin/bash

# Chimera Trading System v2.0 - Complete System Startup
# This script starts both the trading system and dashboard

echo "🚀 Starting Chimera Trading System v2.0..."
echo "================================================"

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $name to start..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo "✅ $name is ready!"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    
    echo "❌ $name failed to start within 30 seconds"
    return 1
}

# Check required ports
echo "🔍 Checking ports..."
if ! check_port 5000; then
    echo "❌ Backend port 5000 is in use. Please stop the service or use a different port."
    exit 1
fi

if ! check_port 5173; then
    echo "❌ Frontend port 5173 is in use. Please stop the service or use a different port."
    exit 1
fi

# Start the backend API
echo "🔧 Starting Dashboard API Backend..."
cd dashboard_api
source venv/bin/activate
python src/main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
if ! wait_for_service "http://localhost:5000/api/system/status" "Backend API"; then
    echo "❌ Failed to start backend. Cleaning up..."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start the frontend dashboard
echo "🎨 Starting Dashboard Frontend..."
cd dashboard_frontend
pnpm run dev --host &
FRONTEND_PID=$!
cd ..

# Wait for frontend to be ready
if ! wait_for_service "http://localhost:5173" "Frontend Dashboard"; then
    echo "❌ Failed to start frontend. Cleaning up..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 Chimera Trading System v2.0 is now running!"
echo "================================================"
echo "📊 Dashboard: http://localhost:5173"
echo "🔌 API: http://localhost:5000"
echo ""
echo "📋 Available Features:"
echo "   • Real-time trading dashboard with dark theme"
echo "   • Order book heatmap visualization (Bookmap-style)"
echo "   • Performance gauges and analytics"
echo "   • Risk management monitoring"
echo "   • ML-powered signal detection"
echo "   • Mock Level 2 data generation"
echo "   • WebSocket real-time updates"
echo ""
echo "🎮 Controls:"
echo "   • Use the 'Start System' button to begin trading"
echo "   • Navigate between different dashboard sections"
echo "   • Monitor real-time performance metrics"
echo ""
echo "⚠️  To stop the system, press Ctrl+C"
echo ""

# Create a cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down Chimera Trading System..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ System stopped successfully"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep the script running
echo "💡 System is running. Press Ctrl+C to stop."
wait

