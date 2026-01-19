import React from 'react'
import './TextInput.css'

function TextInput({ type = 'text', value, onChange, placeholder, min, max, step, disabled }) {
  return (
    <input
      type={type}
      className="form-text-input"
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
    />
  )
}

export default TextInput

