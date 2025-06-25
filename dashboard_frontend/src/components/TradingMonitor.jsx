import React, { useState, useEffect } from 'react'
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Volume2,
  Target,
  Clock,
  DollarSign
} from 'lucide-react'

// Order Book Heatmap Component (Bookmap-style)
const OrderBookHeatmap = ({ symbol = 'AAPL' }) => {
  const [orderBook, setOrderBook] = useState({
    bids: [],
    asks: [],
    timestamp: Date.now()
  })

  // Mock order book data
  useEffect(() => {
    const generateOrderBook = () => {
      const basePrice = 150.0
      const bids = []
      const asks = []
      
      // Generate bid levels
      for (let i = 0; i < 10; i++) {
        bids.push({
          price: basePrice - (i * 0.01),
          size: Math.floor(Math.random() * 2000) + 500,
          orders: Math.floor(Math.random() * 10) + 1,
          intensity: Math.random()
        })
      }
      
      // Generate ask levels
      for (let i = 0; i < 10; i++) {
        asks.push({
          price: basePrice + ((i + 1) * 0.01),
          size: Math.floor(Math.random() * 2000) + 500,
          orders: Math.floor(Math.random() * 10) + 1,
          intensity: Math.random()
        })
      }
      
      setOrderBook({ bids, asks, timestamp: Date.now() })
    }

    generateOrderBook()
    const interval = setInterval(generateOrderBook, 1000)
    return () => clearInterval(interval)
  }, [])

  const getIntensityColor = (intensity, side) => {
    const alpha = Math.min(intensity * 0.8 + 0.2, 1)
    if (side === 'bid') {
      return `rgba(16, 185, 129, ${alpha})` // Green for bids
    } else {
      return `rgba(239, 68, 68, ${alpha})` // Red for asks
    }
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Order Book - {symbol}</h3>
        <div className="text-sm text-gray-400">
          Last Update: {new Date(orderBook.timestamp).toLocaleTimeString()}
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-4">
        {/* Bids */}
        <div>
          <div className="text-sm font-medium text-green-400 mb-2">Bids</div>
          <div className="space-y-1">
            {orderBook.bids.map((bid, index) => (
              <div 
                key={index}
                className="flex justify-between items-center p-2 rounded text-sm"
                style={{ backgroundColor: getIntensityColor(bid.intensity, 'bid') }}
              >
                <span className="text-white font-mono">{bid.price.toFixed(2)}</span>
                <span className="text-white">{bid.size.toLocaleString()}</span>
                <span className="text-gray-300 text-xs">{bid.orders}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Spread */}
        <div className="flex flex-col items-center justify-center">
          <div className="text-center">
            <div className="text-2xl font-bold text-white">
              {orderBook.asks.length > 0 && orderBook.bids.length > 0 
                ? (orderBook.asks[0].price - orderBook.bids[0].price).toFixed(3)
                : '0.000'
              }
            </div>
            <div className="text-sm text-gray-400">Spread</div>
          </div>
          
          <div className="mt-4 text-center">
            <div className="text-lg font-bold text-blue-400">
              {orderBook.asks.length > 0 && orderBook.bids.length > 0 
                ? ((orderBook.asks[0].price + orderBook.bids[0].price) / 2).toFixed(2)
                : '150.00'
              }
            </div>
            <div className="text-sm text-gray-400">Mid Price</div>
          </div>
        </div>

        {/* Asks */}
        <div>
          <div className="text-sm font-medium text-red-400 mb-2">Asks</div>
          <div className="space-y-1">
            {orderBook.asks.map((ask, index) => (
              <div 
                key={index}
                className="flex justify-between items-center p-2 rounded text-sm"
                style={{ backgroundColor: getIntensityColor(ask.intensity, 'ask') }}
              >
                <span className="text-white font-mono">{ask.price.toFixed(2)}</span>
                <span className="text-white">{ask.size.toLocaleString()}</span>
                <span className="text-gray-300 text-xs">{ask.orders}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// Signal Card Component
const SignalCard = ({ signal, index }) => {
  const getSignalColor = (type) => {
    switch (type) {
      case 'buy': return 'border-green-500 bg-green-900'
      case 'sell': return 'border-red-500 bg-red-900'
      default: return 'border-blue-500 bg-blue-900'
    }
  }

  const getSignalIcon = (type) => {
    switch (type) {
      case 'buy': return <TrendingUp className="w-5 h-5 text-green-400" />
      case 'sell': return <TrendingDown className="w-5 h-5 text-red-400" />
      default: return <Activity className="w-5 h-5 text-blue-400" />
    }
  }

  return (
    <div className={`border rounded-lg p-4 ${getSignalColor(signal.type)}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          {getSignalIcon(signal.type)}
          <span className="font-medium text-white capitalize">{signal.type} Signal</span>
        </div>
        <span className="text-xs text-gray-400">
          {new Date(signal.timestamp).toLocaleTimeString()}
        </span>
      </div>
      
      <div className="space-y-1">
        <div className="flex justify-between text-sm">
          <span className="text-gray-300">Symbol:</span>
          <span className="text-white font-mono">{signal.symbol}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-300">Price:</span>
          <span className="text-white font-mono">${signal.price}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-300">Confidence:</span>
          <span className="text-white">{(signal.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-300">Strategy:</span>
          <span className="text-white">{signal.strategy}</span>
        </div>
      </div>
    </div>
  )
}

// Trade Feed Component
const TradeFeed = ({ trades }) => {
  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h3 className="text-lg font-semibold text-white mb-4">Live Trade Feed</h3>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {trades.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            No trades executed yet
          </div>
        ) : (
          trades.slice(-20).reverse().map((trade, index) => (
            <div key={index} className="flex items-center justify-between p-3 bg-gray-900 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className={`w-2 h-2 rounded-full ${
                  trade.side === 'buy' ? 'bg-green-400' : 'bg-red-400'
                }`} />
                <span className="font-mono text-white">{trade.symbol}</span>
                <span className={`text-sm ${
                  trade.side === 'buy' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {trade.side.toUpperCase()}
                </span>
              </div>
              
              <div className="flex items-center space-x-4 text-sm">
                <span className="text-white font-mono">${trade.price}</span>
                <span className="text-gray-300">{trade.size}</span>
                <span className="text-gray-400">
                  {new Date(trade.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

const TradingMonitor = ({ tradingData, systemStatus }) => {
  // Mock signals data
  const [signals, setSignals] = useState([])
  
  useEffect(() => {
    // Generate mock signals
    const generateSignal = () => {
      const types = ['buy', 'sell', 'hold']
      const symbols = ['AAPL', 'NVDA', 'TSLA', 'SPY']
      const strategies = ['Liquidity Sweep', 'Absorption Reversal', 'Mean Reversion', 'Momentum']
      
      return {
        type: types[Math.floor(Math.random() * types.length)],
        symbol: symbols[Math.floor(Math.random() * symbols.length)],
        price: (150 + Math.random() * 100).toFixed(2),
        confidence: Math.random(),
        strategy: strategies[Math.floor(Math.random() * strategies.length)],
        timestamp: Date.now()
      }
    }

    const interval = setInterval(() => {
      if (systemStatus.status === 'running' && Math.random() > 0.7) {
        setSignals(prev => [generateSignal(), ...prev.slice(0, 9)])
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [systemStatus.status])

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Trading Monitor</h1>
          <p className="text-gray-400 mt-1">Real-time order book, signals, and trade execution</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">{signals.length}</div>
            <div className="text-sm text-gray-400">Active Signals</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {tradingData.trades?.length || 0}
            </div>
            <div className="text-sm text-gray-400">Trades Today</div>
          </div>
        </div>
      </div>

      {/* Order Book Heatmap */}
      <OrderBookHeatmap />

      {/* Signals and Trades */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Signals */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Active Signals</h3>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {signals.length === 0 ? (
              <div className="text-center text-gray-400 py-8">
                {systemStatus.status === 'running' 
                  ? 'Waiting for signals...' 
                  : 'Start the system to see signals'
                }
              </div>
            ) : (
              signals.map((signal, index) => (
                <SignalCard key={index} signal={signal} index={index} />
              ))
            )}
          </div>
        </div>

        {/* Trade Feed */}
        <TradeFeed trades={tradingData.trades || []} />
      </div>

      {/* Market Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center space-x-3 mb-4">
            <Volume2 className="w-6 h-6 text-blue-400" />
            <h3 className="text-lg font-semibold text-white">Volume Analysis</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Volume</span>
              <span className="text-white font-mono">2.5M</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Volume</span>
              <span className="text-white font-mono">1.8M</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Volume Ratio</span>
              <span className="text-green-400 font-mono">1.39x</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center space-x-3 mb-4">
            <Target className="w-6 h-6 text-purple-400" />
            <h3 className="text-lg font-semibold text-white">Signal Accuracy</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Today</span>
              <span className="text-green-400 font-mono">72.5%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">This Week</span>
              <span className="text-green-400 font-mono">68.2%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">This Month</span>
              <span className="text-yellow-400 font-mono">65.8%</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center space-x-3 mb-4">
            <Clock className="w-6 h-6 text-yellow-400" />
            <h3 className="text-lg font-semibold text-white">Execution Stats</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Latency</span>
              <span className="text-green-400 font-mono">2.3ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Fill Rate</span>
              <span className="text-green-400 font-mono">98.7%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Slippage</span>
              <span className="text-yellow-400 font-mono">0.02%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TradingMonitor

