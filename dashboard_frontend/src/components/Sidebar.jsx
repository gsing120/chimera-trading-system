import React from 'react'
import { 
  BarChart3, 
  TrendingUp, 
  Shield, 
  Settings, 
  Activity,
  Zap,
  DollarSign,
  AlertTriangle
} from 'lucide-react'

const Sidebar = ({ currentPage, setCurrentPage, systemStatus }) => {
  const menuItems = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'trading', label: 'Trading Monitor', icon: Activity },
    { id: 'performance', label: 'Performance', icon: TrendingUp },
    { id: 'risk', label: 'Risk Management', icon: Shield },
    { id: 'config', label: 'Configuration', icon: Settings }
  ]

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'text-green-400'
      case 'stopped': return 'text-red-400'
      case 'starting': return 'text-yellow-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return <Zap className="w-4 h-4" />
      case 'stopped': return <AlertTriangle className="w-4 h-4" />
      default: return <Activity className="w-4 h-4" />
    }
  }

  return (
    <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <BarChart3 className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Chimera</h1>
            <p className="text-xs text-gray-400">Trading Dashboard</p>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="p-4 border-b border-gray-700">
        <div className="bg-gray-900 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-300">System Status</span>
            <div className={`flex items-center space-x-1 ${getStatusColor(systemStatus.status)}`}>
              {getStatusIcon(systemStatus.status)}
              <span className="text-xs font-medium capitalize">{systemStatus.status}</span>
            </div>
          </div>
          
          {systemStatus.status === 'running' && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Uptime</span>
                <span className="text-white">{Math.floor(systemStatus.uptime / 60)}m</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Signals</span>
                <span className="text-green-400">{systemStatus.signals_generated}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">P&L</span>
                <span className={systemStatus.performance?.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                  ${systemStatus.performance?.daily_pnl?.toFixed(2) || '0.00'}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon
            const isActive = currentPage === item.id
            
            return (
              <li key={item.id}>
                <button
                  onClick={() => setCurrentPage(item.id)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{item.label}</span>
                </button>
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Quick Stats */}
      <div className="p-4 border-t border-gray-700">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-900 rounded-lg p-2 text-center">
            <div className="text-lg font-bold text-green-400">
              {systemStatus.performance?.win_rate ? (systemStatus.performance.win_rate * 100).toFixed(0) : '0'}%
            </div>
            <div className="text-xs text-gray-400">Win Rate</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-2 text-center">
            <div className="text-lg font-bold text-blue-400">
              {systemStatus.risk?.position_count || 0}
            </div>
            <div className="text-xs text-gray-400">Positions</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Sidebar

