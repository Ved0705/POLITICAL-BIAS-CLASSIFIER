import { useState } from 'react'
import './App.css'

function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handlePredict = async () => {
    if (!text.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Send POST request to backend
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`)
      }

      const data = await response.json()
      
      // Expected backend response: { "bias": "Left" } or similar
      // If backend returns a different key, we'll adapt. Fallback to data.prediction or data.result if needed.
      const predictedBias = data.bias || data.prediction || data.result

      if (predictedBias) {
        setResult(predictedBias)
      } else {
        throw new Error('Unexpected response format from server')
      }
    } catch (err) {
      console.error(err)
      setError(err.message === 'Failed to fetch' 
        ? 'Cannot connect to server. Is the backend running on port 8000?' 
        : err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="header">
        <h1>Political Bias Classifier</h1>
        <p className="subtitle">Paste text below to detect its political leaning</p>
      </div>

      <div className="input-container">
        <div className="textarea-wrapper">
          <textarea
            placeholder="Enter news article, speech, or social media post here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={loading}
          />
        </div>

        <button 
          className="predict-btn" 
          onClick={handlePredict}
          disabled={loading || !text.trim()}
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              Analyzing...
            </>
          ) : (
            'Predict Bias'
          )}
        </button>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {result && !loading && !error && (
          <div className="result-container">
            <div className="result-label">Predicted Bias</div>
            <div className={`result-value result-${result}`}>
              {result}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
