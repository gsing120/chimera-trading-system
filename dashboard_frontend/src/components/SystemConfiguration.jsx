import React from 'react'
import { Settings, Database, Cpu, Wifi } from 'lucide-react'

const SystemConfiguration = ({ systemStatus }) => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">System Configuration</h1>
        <p className="text-gray-400 mt-1">Trading system parameters and settings</p>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Current Configuration</h3>
        <div className="space-y-4">
          <div className="flex justify-between">
            <span className="text-gray-400">System Status</span>
            <span className={`font-medium capitalize ${
              systemStatus.status === 'running' ? 'text-green-400' : 'text-red-400'
            }`}>
              {systemStatus.status}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Active Symbols</span>
            <span className="text-white">{systemStatus.active_symbols?.length || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">ML Models</span>
            <span className="text-green-400">Active</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Data Source</span>
            <span className="text-blue-400">Mock Data</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SystemConfiguration

