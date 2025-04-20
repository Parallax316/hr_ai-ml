import { create } from "zustand"
import { uploadDataFile } from "@/lib/api"

type DataStatus = "idle" | "processing" | "ready" | "error"

interface DataState {
  dataFile: File | null
  dataStatus: DataStatus
  dataError: string | null
  uploadData: (file: File) => Promise<void>
}

export const useDataStore = create<DataState>((set) => ({
  dataFile: null,
  dataStatus: "idle",
  dataError: null,
  uploadData: async (file: File) => {
    set({ dataFile: file, dataStatus: "processing", dataError: null })

    try {
      const response = await uploadDataFile(file)

      if (response.status === "uploaded") {
        set({ dataStatus: "ready" })
      } else {
        set({
          dataStatus: "error",
          dataError: "Failed to process data file. Please try again.",
        })
      }
    } catch (error) {
      console.error("Error uploading data:", error)
      set({
        dataStatus: "error",
        dataError: error instanceof Error ? error.message : "Failed to process data file. Please try again.",
      })
    }
  },
}))
