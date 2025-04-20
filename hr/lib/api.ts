// API client for interacting with the FastAPI backend

const API_BASE_URL = "http://localhost:8000" // Change this if your backend runs on a different port

export async function uploadPolicyFile(file: File): Promise<{
  message: string
  filename: string
  status: string
}> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(`${API_BASE_URL}/upload/policy`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || "Failed to upload policy file")
  }

  return response.json()
}

export async function uploadDataFile(file: File): Promise<{
  message: string
  filename: string
  status: string
}> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(`${API_BASE_URL}/upload/data`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || "Failed to upload data file")
  }

  return response.json()
}

export async function runAnalysis(): Promise<{
  message: string
  status: string
  summary: Record<string, string>
}> {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || "Failed to run analysis")
  }

  return response.json()
}

export async function* streamPolicyChat(prompt: string): AsyncGenerator<string, void, unknown> {
  const response = await fetch(`${API_BASE_URL}/chat/policy`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prompt }),
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || "Failed to chat with policy assistant")
  }

  if (!response.body) {
    throw new Error("Response body is null")
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      yield chunk
    }
  } finally {
    reader.releaseLock()
  }
}

export async function* streamAnalysisChat(prompt: string): AsyncGenerator<string, void, unknown> {
  const response = await fetch(`${API_BASE_URL}/chat/analysis`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prompt }),
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || "Failed to chat with analysis assistant")
  }

  if (!response.body) {
    throw new Error("Response body is null")
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      yield chunk
    }
  } finally {
    reader.releaseLock()
  }
}

export async function getBasicDashboard(): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/dashboard/basic`, {
    method: "GET",
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || "Failed to get basic dashboard")
  }

  return response.blob()
}

export async function getExtendedDashboard(): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/dashboard/extended`, {
    method: "GET",
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || "Failed to get extended dashboard")
  }

  return response.blob()
}
