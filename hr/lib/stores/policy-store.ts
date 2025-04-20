import { create } from "zustand"
import { uploadPolicyFile } from "@/lib/api"

type PolicyStatus = "idle" | "processing" | "ready" | "error"

interface PolicyState {
  policyFile: File | null
  policyStatus: PolicyStatus
  policyError: string | null
  uploadPolicy: (file: File) => Promise<void>
}

export const usePolicyStore = create<PolicyState>((set) => ({
  policyFile: null,
  policyStatus: "idle",
  policyError: null,
  uploadPolicy: async (file: File) => {
    set({ policyFile: file, policyStatus: "processing", policyError: null })

    try {
      const response = await uploadPolicyFile(file)

      if (response.status === "ready") {
        set({ policyStatus: "ready" })
      } else {
        set({
          policyStatus: "error",
          policyError: "Failed to process policy document. Please try again.",
        })
      }
    } catch (error) {
      console.error("Error uploading policy:", error)
      set({
        policyStatus: "error",
        policyError: error instanceof Error ? error.message : "Failed to process policy document. Please try again.",
      })
    }
  },
}))
