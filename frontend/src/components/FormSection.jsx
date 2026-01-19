import React from 'react'
import './FormSection.css'

function FormSection({ title, description, children }) {
  return (
    <div className="form-section">
      <h3 className="form-section-title">{title}</h3>
      <p className="form-section-description">{description}</p>
      <div className="form-section-input">
        {children}
      </div>
    </div>
  )
}

export default FormSection

