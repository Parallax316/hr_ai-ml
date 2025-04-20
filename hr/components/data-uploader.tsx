"use client"

import type React from "react"

import { useState } from "react"
import { Upload, FileText, AlertCircle, CheckCircle2, Loader2 } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useDataStore } from "@/lib/stores/data-store"

export function DataUploader() {
  const { uploadData, dataFile, dataStatus, dataError } = useDataStore()
  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files.length > 0 && files[0].type === "text/csv") {
      handleFileUpload(files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0])
    }
  }

  const handleFileUpload = async (file: File) => {
    uploadData(file)
  }

  return (
    <div className="space-y-4">
      <div
        className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
          isDragging
            ? "border-primary bg-primary/10"
            : "border-muted-foreground/20 hover:border-primary/50 hover:bg-primary/5"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => document.getElementById("data-upload")?.click()}
      >
        <input type="file" id="data-upload" className="hidden" accept=".csv" onChange={handleFileChange} />
        <div className="flex flex-col items-center gap-2 py-4">
          <Upload className="w-8 h-8 text-primary/80" />
          <p className="text-sm font-medium">Upload Employee Data</p>
          <p className="text-xs text-muted-foreground">Drag & drop or click to browse</p>
          <p className="text-xs text-muted-foreground">CSV files only</p>
        </div>
      </div>

      {dataFile && (
        <div className="p-3 border rounded-lg bg-card/50 backdrop-blur-sm">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium truncate">{dataFile.name}</span>
          </div>

          {dataStatus === "processing" && (
            <div className="mt-2 space-y-1">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="w-3 h-3 animate-spin" />
                <span>Processing data...</span>
              </div>
              <Progress value={33} className="h-1" />
            </div>
          )}

          {dataStatus === "ready" && (
            <div className="flex items-center gap-2 mt-2 text-xs text-green-600">
              <CheckCircle2 className="w-3 h-3" />
              <span>Data ready for analysis</span>
            </div>
          )}
        </div>
      )}

      {dataError && (
        <Alert variant="destructive" className="py-2">
          <AlertCircle className="w-4 h-4" />
          <AlertDescription className="text-xs">{dataError}</AlertDescription>
        </Alert>
      )}
    </div>
  )
}
