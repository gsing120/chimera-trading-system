import React from 'react'
import { 
  Play, 
  Square, 
  Wifi, 
  WifiOff, 
  Bell, 
  User,
  Settings
} from 'lucide-react'
import { Button } from '@/components/ui/button'

const Header = ({ isConnected, systemStatus, onStartSystem, onStopSystem }) => {
  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`
    } else {
      return `${secs}s`
    }
  }

  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left side - System controls */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            {systemStatus.status === 'running' ? (
              <Button
                onClick={onStopSystem}
                variant="destructive"
                size="sm"
                className="flex items-center space-x-2"
              >
                <Square className="w-4 h-4" />
                <span>Stop System</span>
              </Button>
            ) : (
              <Button
                onClick={onStartSystem}
                variant="default"
                size="sm"
                className="flex items-center space-x-2 bg-green-600 hover:bg-green-700"
              >
                <Play className="w-4 h-4" />
                <span>Start System</span>
              </Button>
            )}
          </div>

          {/* System Status Indicator */}
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              systemStatus.status === 'running' ? 'bg-green-400 animate-pulse' : 'bg-red-400'
            }`} />
            <span className="text-sm text-gray-300 capitalize">
              {systemStatus.status}
            </span>
            {systemStatus.status === 'running' && (
              <span className="text-xs text-gray-400">
                ({formatUptime(systemStatus.uptime)})
              </span>
            )}
          </div>
        </div>

        {/* Center - Key Metrics */}
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-4">
            <div className="text-center">
              <div className="text-lg font-bold text-white">
                ${systemStatus.performance?.total_pnl?.toFixed(2) || '0.00'}
              </div>
              <div className="text-xs text-gray-400">Total P&L</div>
            </div>
            
            <div className="text-center">
              <div className={`text-lg font-bold ${
                (systemStatus.performance?.daily_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                ${systemStatus.performance?.daily_pnl?.toFixed(2) || '0.00'}
              </div>
              <div className="text-xs text-gray-400">Daily P&L</div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-bold text-blue-400">
                {systemStatus.signals_generated || 0}
              </div>
              <div className="text-xs text-gray-400">Signals</div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-bold text-purple-400">
                {systemStatus.ml_predictions || 0}
              </div>
              <div className="text-xs text-gray-400">ML Predictions</div>
            </div>
          </div>
        </div>

        {/* Right side - Connection status and user menu */}
        <div className="flex items-center space-x-4">
          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <>
                <Wifi className="w-4 h-4 text-green-400" />
                <span className="text-sm text-green-400">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-red-400" />
                <span className="text-sm text-red-400">Disconnected</span>
              </>
            )}
          </div>

          {/* Notifications */}
          <Button variant="ghost" size="sm" className="relative">
            <Bell className="w-4 h-4" />
            <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full"></span>
          </Button>

          {/* Settings */}
          <Button variant="ghost" size="sm">
            <Settings className="w-4 h-4" />
          </Button>

          {/* User Menu */}
          <Button variant="ghost" size="sm" className="flex items-center space-x-2">
            <User className="w-4 h-4" />
            <span className="text-sm">Admin</span>
          </Button>
        </div>
      </div>

      {/* Alert Bar */}
      {systemStatus.status === 'running' && systemStatus.performance?.daily_pnl < -1000 && (
        <div className="mt-3 bg-red-900 border border-red-700 rounded-lg px-4 py-2">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse" />
            <span className="text-sm text-red-200">
              Daily loss exceeds $1,000 threshold. Consider reviewing risk parameters.
            </span>
          </div>
        </div>
      )}
    </header>
  )
}

export default Header

