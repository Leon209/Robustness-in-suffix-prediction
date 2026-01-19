const API_BASE_URL = '/api'

export const generateDataset = async (data) => {
  const response = await fetch(`${API_BASE_URL}/generate-dataset`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      dataset: data.dataset,
      perturbable_features: data.perturbable_features,
      num_perturbations: data.num_perturbations,
      perturbation_rate: data.perturbation_rate
    }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || 'Failed to generate dataset')
  }

  // Get the blob from response
  const blob = await response.blob()
  
  // Create download link
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  
  // Get filename from Content-Disposition header or use default
  const contentDisposition = response.headers.get('Content-Disposition')
  let filename = 'dataset.csv'
  if (contentDisposition) {
    const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i)
    if (filenameMatch) {
      filename = filenameMatch[1]
    }
  }
  
  a.download = filename
  document.body.appendChild(a)
  a.click()
  
  // Cleanup
  window.URL.revokeObjectURL(url)
  document.body.removeChild(a)
}
