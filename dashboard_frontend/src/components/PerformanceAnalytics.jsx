import React from 'react'
import { TrendingUp, BarChart3, PieChart, Target } from 'lucide-react'

const PerformanceAnalytics = ({ performanceMetrics, systemStatus }) => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Performance Analytics</h1>
        <p className="text-gray-400 mt-1">Detailed performance metrics and strategy analysis</p>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Performance Overview</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              ${systemStatus.performance?.total_pnl?.toFixed(2) || '0.00'}
            </div>
            <div className="text-sm text-gray-400">Total P&L</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">
              {((systemStatus.performance?.sharpe_ratio || 0) * 100).toFixed(1)}
            </div>
            <div className="text-sm text-gray-400">Sharpe Ratio</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-400">
              {((systemStatus.performance?.max_drawdown || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">Max Drawdown</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-400">
              {((systemStatus.performance?.win_rate || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">Win Rate</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PerformanceAnalytics

