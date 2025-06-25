import React from 'react'
import { Shield, AlertTriangle, TrendingDown, Activity } from 'lucide-react'

const RiskManagement = ({ riskMetrics, systemStatus }) => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Risk Management</h1>
        <p className="text-gray-400 mt-1">Portfolio risk monitoring and control</p>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Risk Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">
              ${systemStatus.risk?.var_95?.toFixed(0) || '2,500'}
            </div>
            <div className="text-sm text-gray-400">VaR (95%)</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-400">
              {((systemStatus.risk?.current_exposure || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400">Current Exposure</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">
              {systemStatus.risk?.position_count || 0}
            </div>
            <div className="text-sm text-gray-400">Active Positions</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-400">
              {systemStatus.risk?.leverage?.toFixed(1) || '1.0'}x
            </div>
            <div className="text-sm text-gray-400">Leverage</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default RiskManagement

