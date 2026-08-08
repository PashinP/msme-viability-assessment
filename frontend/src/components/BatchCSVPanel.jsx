import { useState } from "react"
import axios from "axios"
import { UploadCloud, File, AlertCircle, CheckCircle } from "lucide-react"
import { Button } from "@/components/ui/button"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"
const API_KEY = import.meta.env.VITE_API_KEY || "msme-dev-key-2024"

export function BatchCSVPanel() {
  const [file, setFile] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setError(null)
      setResults(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsUploading(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const { data } = await axios.post(`${API_URL}/predict/batch`, formData, {
        headers: { 
          "X-API-Key": API_KEY,
          "Content-Type": "multipart/form-data"
        }
      })
      setResults(data)
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || "An error occurred during upload. Please ensure your CSV has the correct 11 columns.")
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <div className="text-center mb-6">
        <h3 className="text-lg font-bold">Portfolio Scoring</h3>
        <p className="text-sm text-muted-foreground">Upload a CSV to process multiple applications at once.</p>
      </div>

      <div className="border-dashed border-2 border-border p-6 rounded-2xl text-center bg-secondary/30">
        <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
          <UploadCloud className="w-6 h-6 text-primary" />
        </div>
        
        <input 
          type="file" 
          id="file-upload" 
          accept=".csv" 
          className="hidden" 
          onChange={handleFileChange}
        />
        
        <div className="flex flex-col items-center gap-4">
          <Button asChild variant="outline" size="sm" className="cursor-pointer">
            <label htmlFor="file-upload">Select CSV File</label>
          </Button>
          
          {file && (
            <div className="flex items-center gap-2 text-xs text-foreground bg-secondary px-3 py-1.5 rounded-lg border border-border">
              <File className="w-3 h-3 text-muted-foreground" />
              <span className="truncate max-w-[150px]">{file.name}</span>
            </div>
          )}

          {file && (
            <Button onClick={handleUpload} disabled={isUploading} className="w-full">
              {isUploading ? "Processing..." : "Process Batch"}
            </Button>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-3 rounded-xl flex items-start gap-2 text-sm">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {results && (
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-green-500/5 border border-green-500/20 flex flex-col items-center text-center gap-2">
            <CheckCircle className="w-6 h-6 text-green-500" />
            <div>
              <h4 className="text-sm font-semibold text-green-400">Batch Success</h4>
              <p className="text-xs text-muted-foreground">Processed {results.total_processed} records.</p>
            </div>
          </div>

          <div className="rounded-xl overflow-hidden border border-border">
            <div className="p-3 bg-secondary text-xs font-semibold grid grid-cols-3">
              <div>ID</div>
              <div>Grade</div>
              <div className="text-right">Conf</div>
            </div>
            <div className="divide-y divide-border max-h-48 overflow-y-auto">
              {results.results.map((res, i) => (
                <div key={i} className="p-3 grid grid-cols-3 text-xs hover:bg-accent/50 transition-colors">
                  <div className="font-mono text-muted-foreground truncate pr-2">#{res.prediction_id.split("-")[0]}</div>
                  <div className="font-bold" style={{ color: `var(--grade-${res.predicted_label.toLowerCase().replace('-', '')})` }}>
                    {res.predicted_label}
                  </div>
                  <div className="text-right">{(res.confidence * 100).toFixed(0)}%</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
