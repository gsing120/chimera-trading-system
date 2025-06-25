import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import io from 'socket.io-client'
import './App.css'

// Import components
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import OverviewDashboard from './components/OverviewDashboard'
import TradingMonitor from './components/TradingMonitor'
import PerformanceAnalytics from './components/PerformanceAnalytics'
import RiskManagement from './components/RiskManagement'
import SystemConfiguration from './components/SystemConfiguration'

// Socket.IO connection
const socket = io('http://localhost:5000', {
  autoConnect: false
})

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [systemStatus, setSystemStatus] = useState({
    status: 'stopped',
    uptime: 0,
    signals_generated: 0,
    trades_executed: 0,
    ml_predictions: 0,
    performance: {
      total_pnl: 0,
      daily_pnl: 0,
      sharpe_ratio: 0,
      max_drawdown: 0,
      win_rate: 0
    },
    risk: {
      current_exposure: 0,
      var_95: 0,
      position_count: 0,
      leverage: 1.0
    }
  })
  const [tradingData, setTradingData] = useState({
    signals: [],
    trades: [],
    order_books: {},
    timestamp: Date.now()
  })
  const [performanceMetrics, setPerformanceMetrics] = useState({})
  const [riskMetrics, setRiskMetrics] = useState({})
  const [currentPage, setCurrentPage] = useState('overview')

  useEffect(() => {
    // Connect to WebSocket
    socket.connect()

    // Socket event listeners
    socket.on('connect', () => {
      console.log('Connected to dashboard API')
      setIsConnected(true)
    })

    socket.on('disconnect', () => {
      console.log('Disconnected from dashboard API')
      setIsConnected(false)
    })

    socket.on('system:status', (data) => {
      setSystemStatus(data)
    })

    socket.on('trading:data', (data) => {
      setTradingData(data)
    })

    socket.on('performance:metrics', (data) => {
      setPerformanceMetrics(data)
    })

    socket.on('risk:metrics', (data) => {
      setRiskMetrics(data)
    })

    // Cleanup on unmount
    return () => {
      socket.disconnect()
    }
  }, [])

  const startSystem = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/system/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      const result = await response.json()
      if (result.success) {
        console.log('System started successfully')
      }
    } catch (error) {
      console.error('Error starting system:', error)
    }
  }

  const stopSystem = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/system/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      const result = await response.json()
      if (result.success) {
        console.log('System stopped successfully')
      }
    } catch (error) {
      console.error('Error stopping system:', error)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Router>
        <div className="flex h-screen">
          {/* Sidebar */}
          <Sidebar 
            currentPage={currentPage} 
            setCurrentPage={setCurrentPage}
            systemStatus={systemStatus}
          />
          
          {/* Main Content */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Header */}
            <Header 
              isConnected={isConnected}
              systemStatus={systemStatus}
              onStartSystem={startSystem}
              onStopSystem={stopSystem}
            />
            
            {/* Page Content */}
            <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-900 p-6">
              <Routes>
                <Route 
                  path="/" 
                  element={
                    <OverviewDashboard 
                      systemStatus={systemStatus}
                      tradingData={tradingData}
                      performanceMetrics={performanceMetrics}
                      riskMetrics={riskMetrics}
                    />
                  } 
                />
                <Route 
                  path="/trading" 
                  element={
                    <TradingMonitor 
                      tradingData={tradingData}
                      systemStatus={systemStatus}
                    />
                  } 
                />
                <Route 
                  path="/performance" 
                  element={
                    <PerformanceAnalytics 
                      performanceMetrics={performanceMetrics}
                      systemStatus={systemStatus}
                    />
                  } 
                />
                <Route 
                  path="/risk" 
                  element={
                    <RiskManagement 
                      riskMetrics={riskMetrics}
                      systemStatus={systemStatus}
                    />
                  } 
                />
                <Route 
                  path="/config" 
                  element={
                    <SystemConfiguration 
                      systemStatus={systemStatus}
                    />
                  } 
                />
              </Routes>
              
              {/* Show content based on current page when not using routing */}
              {currentPage === 'overview' && (
                <OverviewDashboard 
                  systemStatus={systemStatus}
                  tradingData={tradingData}
                  performanceMetrics={performanceMetrics}
                  riskMetrics={riskMetrics}
                />
              )}
              {currentPage === 'trading' && (
                <TradingMonitor 
                  tradingData={tradingData}
                  systemStatus={systemStatus}
                />
              )}
              {currentPage === 'performance' && (
                <PerformanceAnalytics 
                  performanceMetrics={performanceMetrics}
                  systemStatus={systemStatus}
                />
              )}
              {currentPage === 'risk' && (
                <RiskManagement 
                  riskMetrics={riskMetrics}
                  systemStatus={systemStatus}
                />
              )}
              {currentPage === 'config' && (
                <SystemConfiguration 
                  systemStatus={systemStatus}
                />
              )}
            </main>
          </div>
        </div>
      </Router>
    </div>
  )
}

export default App

