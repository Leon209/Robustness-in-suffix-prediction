import React from 'react'
import './GenerateButton.css'

function GenerateButton({ onClick, disabled, loading, children }) {
  return (
    <button
      className={`generate-button ${disabled ? 'disabled' : ''}`}
      onClick={onClick}
      disabled={disabled || loading}
    >
      {loading ? 'Generating...' : children || 'Generate Dataset!'}
    </button>
  )
}

export default GenerateButton
