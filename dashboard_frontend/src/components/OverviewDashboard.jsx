import React from 'react'
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  DollarSign,
  Target,
  Shield,
  Zap,
  Brain
} from 'lucide-react'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar
} from 'recharts'

// Gauge component
const CircularGauge = ({ value, max, label, color = '#3b82f6', size = 120 }) => {
  const percentage = Math.min((value / max) * 100, 100)
  const strokeDasharray = `${percentage * 2.51} 251.2`
  
  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r="40"
            stroke="#374151"
            strokeWidth="8"
            fill="transparent"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r="40"
            stroke={color}
            strokeWidth="8"
            fill="transparent"
            strokeDasharray={strokeDasharray}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-lg font-bold text-white">{value}</div>
            <div className="text-xs text-gray-400">{label}</div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Linear gauge component
const LinearGauge = ({ value, max, label, color = '#3b82f6' }) => {
  const percentage = Math.min((value / max) * 100, 100)
  
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-gray-300">{label}</span>
        <span className="text-white font-medium">{value}</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div 
          className="h-2 rounded-full transition-all duration-1000 ease-out"
          style={{ 
            width: `${percentage}%`,
            backgroundColor: color
          }}
        />
      </div>
    </div>
  )
}

// Metric card component
const MetricCard = ({ title, value, change, icon: Icon, color = 'blue' }) => {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    red: 'bg-red-500',
    purple: 'bg-purple-500',
    yellow: 'bg-yellow-500'
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm font-medium">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {change !== undefined && (
            <div className={`flex items-center mt-2 text-sm ${
              change >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {change >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              <span>{Math.abs(change).toFixed(2)}%</span>
            </div>
          )}
        </div>
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  )
}

const OverviewDashboard = ({ systemStatus, tradingData, performanceMetrics, riskMetrics }) => {
  // Mock data for charts
  const equityCurveData = Array.from({ length: 24 }, (_, i) => ({
    time: `${i}:00`,
    value: 10000 + Math.random() * 2000 - 1000 + i * 50
  }))

  const signalFrequencyData = Array.from({ length: 12 }, (_, i) => ({
    hour: `${i * 2}:00`,
    signals: Math.floor(Math.random() * 20) + 5
  }))

  const strategyPerformanceData = [
    { name: 'Liquidity Sweep', value: 35, color: '#3b82f6' },
    { name: 'Absorption', value: 28, color: '#10b981' },
    { name: 'Mean Reversion', value: 22, color: '#f59e0b' },
    { name: 'Momentum', value: 15, color: '#ef4444' }
  ]

  const riskExposureData = [
    { category: 'Equity', exposure: 65 },
    { category: 'Options', exposure: 25 },
    { category: 'Futures', exposure: 10 }
  ]

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Overview Dashboard</h1>
          <p className="text-gray-400 mt-1">Real-time system monitoring and key performance indicators</p>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-400">Last Updated</div>
          <div className="text-white font-medium">{new Date().toLocaleTimeString()}</div>
        </div>
      </div>

      {/* Key Performance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total P&L"
          value={`$${systemStatus.performance?.total_pnl?.toFixed(2) || '0.00'}`}
          change={2.5}
          icon={DollarSign}
          color="green"
        />
        <MetricCard
          title="Daily P&L"
          value={`$${systemStatus.performance?.daily_pnl?.toFixed(2) || '0.00'}`}
          change={systemStatus.performance?.daily_pnl >= 0 ? 1.2 : -1.2}
          icon={TrendingUp}
          color={systemStatus.performance?.daily_pnl >= 0 ? 'green' : 'red'}
        />
        <MetricCard
          title="Active Signals"
          value={systemStatus.signals_generated || 0}
          change={5.8}
          icon={Activity}
          color="blue"
        />
        <MetricCard
          title="ML Predictions"
          value={systemStatus.ml_predictions || 0}
          change={3.2}
          icon={Brain}
          color="purple"
        />
      </div>

      {/* Gauges Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">System Health</h3>
          <CircularGauge
            value={systemStatus.status === 'running' ? 95 : 0}
            max={100}
            label="Health %"
            color="#10b981"
          />
        </div>
        
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Risk Level</h3>
          <CircularGauge
            value={Math.min(systemStatus.risk?.current_exposure * 100 || 0, 100)}
            max={100}
            label="Risk %"
            color="#f59e0b"
          />
        </div>
        
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Win Rate</h3>
          <CircularGauge
            value={Math.round((systemStatus.performance?.win_rate || 0) * 100)}
            max={100}
            label="Win %"
            color="#3b82f6"
          />
        </div>
        
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Performance</h3>
          <CircularGauge
            value={Math.min(Math.max((systemStatus.performance?.sharpe_ratio || 0) * 50, 0), 100)}
            max={100}
            label="Sharpe"
            color="#8b5cf6"
          />
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Equity Curve */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Equity Curve</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={equityCurveData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1f2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Signal Frequency */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Signal Frequency</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={signalFrequencyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hour" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1f2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="signals" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Strategy Performance and Risk Exposure */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Strategy Performance */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Strategy Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={strategyPerformanceData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}%`}
              >
                {strategyPerformanceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Metrics */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Risk Exposure</h3>
          <div className="space-y-4">
            {riskExposureData.map((item, index) => (
              <LinearGauge
                key={index}
                value={`${item.exposure}%`}
                max={100}
                label={item.category}
                color={index === 0 ? '#3b82f6' : index === 1 ? '#10b981' : '#f59e0b'}
              />
            ))}
          </div>
          
          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-red-400">
                ${riskMetrics.var_95?.toFixed(0) || '2,500'}
              </div>
              <div className="text-sm text-gray-400">VaR (95%)</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400">
                {((systemStatus.performance?.max_drawdown || 0) * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-400">Max Drawdown</div>
            </div>
          </div>
        </div>
      </div>

      {/* System Alerts */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">System Alerts</h3>
        <div className="space-y-3">
          {systemStatus.status === 'running' ? (
            <>
              <div className="flex items-center space-x-3 p-3 bg-green-900 border border-green-700 rounded-lg">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-green-200">System operating normally</span>
                <span className="text-xs text-green-400 ml-auto">
                  {new Date().toLocaleTimeString()}
                </span>
              </div>
              {systemStatus.performance?.daily_pnl < -500 && (
                <div className="flex items-center space-x-3 p-3 bg-yellow-900 border border-yellow-700 rounded-lg">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full" />
                  <span className="text-yellow-200">Daily loss approaching threshold</span>
                  <span className="text-xs text-yellow-400 ml-auto">
                    {new Date().toLocaleTimeString()}
                  </span>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center space-x-3 p-3 bg-red-900 border border-red-700 rounded-lg">
              <div className="w-2 h-2 bg-red-400 rounded-full" />
              <span className="text-red-200">Trading system is stopped</span>
              <span className="text-xs text-red-400 ml-auto">
                {new Date().toLocaleTimeString()}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default OverviewDashboard

