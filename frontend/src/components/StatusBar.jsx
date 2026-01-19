import React from 'react'
import './StatusBar.css'

function StatusBar({ status }) {
  return (
    <div className="status-bar">
      <div className="status-bar-content">
        <span className="status-label">Status:</span>
        <span className="status-value">{status}</span>
      </div>
    </div>
  )
}

export default StatusBar

