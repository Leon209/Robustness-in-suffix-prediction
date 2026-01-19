import React, { useState, useEffect } from 'react'
import StatusBar from './components/StatusBar'
import FormSection from './components/FormSection'
import MultiSelectDropdown from './components/MultiSelectDropdown'
import TextInput from './components/TextInput'
import GenerateButton from './components/GenerateButton'
import { generateDataset } from './services/api'
import './App.css'

const PERTURBABLE_FEATURES = [
  'Resource',
  'Variant index',
  'seriousness',
  'customer',
  'responsible_section',
  'product',
  'seriousness_2',
  'service_level',
  'service_type',
  'support_section',
  'workgroup'
]

function App() {
  const [dataset, setDataset] = useState('')
  const [perturbableFeatures, setPerturbableFeatures] = useState([])
  const [numPerturbations, setNumPerturbations] = useState('')
  const [perturbationRate, setPerturbationRate] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [status, setStatus] = useState('Ready')

  const maxPerturbations = perturbableFeatures.length

  useEffect(() => {
    // Reset numPerturbations if it exceeds max when features change
    if (numPerturbations && parseInt(numPerturbations) > maxPerturbations) {
      setNumPerturbations('')
    }
  }, [perturbableFeatures, numPerturbations, maxPerturbations])

  const validateForm = () => {
    if (!dataset) {
      setError('Please select a dataset')
      return false
    }
    if (perturbableFeatures.length === 0) {
      setError('Please select at least one perturbable feature')
      return false
    }
    if (!numPerturbations) {
      setError('Please enter number of perturbations per event')
      return false
    }
    const numPert = parseInt(numPerturbations)
    if (isNaN(numPert) || numPert < 1 || numPert > maxPerturbations) {
      setError(`Number of perturbations must be between 1 and ${maxPerturbations}`)
      return false
    }
    if (!perturbationRate) {
      setError('Please enter perturbation rate')
      return false
    }
    const rate = parseFloat(perturbationRate)
    if (isNaN(rate) || rate <= 0 || rate > 1) {
      setError('Perturbation rate must be between 0 and 1 (exclusive of 0)')
      return false
    }
    return true
  }

  const handleGenerate = async () => {
    if (!validateForm()) return

    setLoading(true)
    setError(null)
    setStatus('Generating dataset...')

    try {
      await generateDataset({
        dataset,
        perturbable_features: perturbableFeatures,
        num_perturbations: parseInt(numPerturbations),
        perturbation_rate: parseFloat(perturbationRate)
      })
      setStatus('Dataset generated successfully!')
      setTimeout(() => setStatus('Ready'), 3000)
    } catch (err) {
      setError(err.message || 'Failed to generate dataset')
      setStatus('Error generating dataset')
      setTimeout(() => setStatus('Ready'), 3000)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <StatusBar status={status} />
      
      <div className="app-content">
        <h1 className="app-title">Dataset Perturbator</h1>
        
        <FormSection
          title="Dataset"
          description="Choose the dataset, you want to apply perturbations towards"
        >
          <select
            className="form-select"
            value={dataset}
            onChange={(e) => {
              setDataset(e.target.value)
              setError(null)
            }}
          >
            <option value="">-- Select dataset --</option>
            <option value="Helpdesk">Helpdesk</option>
          </select>
        </FormSection>

        <FormSection
          title="Perturbable Features"
          description="Choose the features you want to perturb"
        >
          <MultiSelectDropdown
            options={PERTURBABLE_FEATURES}
            selected={perturbableFeatures}
            onChange={setPerturbableFeatures}
          />
        </FormSection>

        <FormSection
          title="Number of Perturbations per Event"
          description={`Select how many features of the list above should be perturbed per Event [1, ${maxPerturbations}]`}
        >
          <TextInput
            type="number"
            value={numPerturbations}
            onChange={(e) => {
              const val = e.target.value
              if (val === '' || (parseInt(val) >= 1 && parseInt(val) <= maxPerturbations)) {
                setNumPerturbations(val)
                setError(null)
              }
            }}
            placeholder={`Enter number (1-${maxPerturbations})`}
            min="1"
            max={maxPerturbations}
            disabled={perturbableFeatures.length === 0}
          />
        </FormSection>

        <FormSection
          title="Perturbation Rate"
          description="Chance an event will be perturbed (0,1]"
        >
          <TextInput
            type="number"
            value={perturbationRate}
            onChange={(e) => {
              const val = e.target.value
              if (val === '' || (parseFloat(val) > 0 && parseFloat(val) <= 1)) {
                setPerturbationRate(val)
                setError(null)
              }
            }}
            placeholder="Enter rate (0,1]"
            step="0.01"
            min="0.01"
            max="1"
          />
        </FormSection>

        <div className="button-container">
          <GenerateButton
            onClick={handleGenerate}
            disabled={loading || !dataset || perturbableFeatures.length === 0 || !numPerturbations || !perturbationRate}
            loading={loading}
          >
            Generate Dataset!
          </GenerateButton>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
