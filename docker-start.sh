#!/bin/bash

# Chimera Trading System - Docker Deployment Script
# Supports both mock data and IBKR live data

set -e

echo "🚀 Chimera Trading System v2.0 - Docker Deployment"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p external_data data/db logs config ibkr_settings

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before running with IBKR"
fi

# Function to start with mock data
start_mock() {
    echo "🎭 Starting Chimera Trading System with MOCK data..."
    echo "   - Backend API: http://localhost:5000"
    echo "   - Frontend Dashboard: http://localhost:3000"
    echo "   - Mock data will be generated externally"
    
    # Set mock data environment
    export DATA_SOURCE=mock
    
    # Start with mock profile
    docker-compose --profile mock up -d
    
    echo "✅ System started with mock data!"
    echo "   Access dashboard at: http://localhost:3000"
}

# Function to start with IBKR
start_ibkr() {
    echo "📈 Starting Chimera Trading System with IBKR integration..."
    
    # Check if IBKR configuration is set
    if ! grep -q "TWS_USERID=your_ibkr_username" .env; then
        echo "⚠️  IBKR credentials detected in .env file"
    else
        echo "❌ Please configure IBKR credentials in .env file first"
        echo "   Edit the following variables:"
        echo "   - TWS_USERID"
        echo "   - TWS_PASSWORD"
        echo "   - IBKR_HOST (if not using containerized gateway)"
        echo "   - IBKR_PORT"
        exit 1
    fi
    
    echo "   - Backend API: http://localhost:5000"
    echo "   - Frontend Dashboard: http://localhost:3000"
    echo "   - IBKR Gateway: http://localhost:4001 (VNC)"
    
    # Set IBKR environment
    export DATA_SOURCE=ibkr
    
    # Start with IBKR profile
    docker-compose --profile ibkr up -d
    
    echo "✅ System started with IBKR integration!"
    echo "   Access dashboard at: http://localhost:3000"
    echo "   IBKR Gateway VNC: http://localhost:4001 (password: chimera123)"
}

# Function to stop the system
stop_system() {
    echo "🛑 Stopping Chimera Trading System..."
    docker-compose --profile mock --profile ibkr down
    echo "✅ System stopped!"
}

# Function to show logs
show_logs() {
    echo "📋 Showing system logs..."
    docker-compose logs -f
}

# Function to show status
show_status() {
    echo "📊 System Status:"
    echo "================"
    docker-compose ps
    echo ""
    echo "🔗 Service URLs:"
    echo "   - Dashboard: http://localhost:3000"
    echo "   - API: http://localhost:5000"
    echo "   - Health Check: http://localhost:5000/api/system/status"
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up Docker resources..."
    docker-compose --profile mock --profile ibkr down -v
    docker system prune -f
    echo "✅ Cleanup completed!"
}

# Function to update system
update_system() {
    echo "🔄 Updating Chimera Trading System..."
    git pull origin main
    docker-compose build --no-cache
    echo "✅ System updated!"
}

# Main menu
case "${1:-}" in
    "mock")
        start_mock
        ;;
    "ibkr")
        start_ibkr
        ;;
    "stop")
        stop_system
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup
        ;;
    "update")
        update_system
        ;;
    "restart")
        stop_system
        sleep 2
        if [ "${2:-}" = "ibkr" ]; then
            start_ibkr
        else
            start_mock
        fi
        ;;
    *)
        echo "Usage: $0 {mock|ibkr|stop|logs|status|cleanup|update|restart}"
        echo ""
        echo "Commands:"
        echo "  mock     - Start with mock data (for testing)"
        echo "  ibkr     - Start with IBKR integration (requires configuration)"
        echo "  stop     - Stop all services"
        echo "  logs     - Show system logs"
        echo "  status   - Show system status"
        echo "  cleanup  - Clean up Docker resources"
        echo "  update   - Update system from Git"
        echo "  restart  - Restart system (add 'ibkr' for IBKR mode)"
        echo ""
        echo "Examples:"
        echo "  $0 mock          # Start with mock data"
        echo "  $0 ibkr          # Start with IBKR"
        echo "  $0 restart ibkr  # Restart with IBKR"
        echo ""
        echo "🔗 Quick Links:"
        echo "   Dashboard: http://localhost:3000"
        echo "   API: http://localhost:5000"
        exit 1
        ;;
esac

