"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { BarChart3, LineChart, Send, Loader2, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { useDataStore } from "@/lib/stores/data-store"
// Make sure runAnalysis is typed correctly in api.ts if possible
// For now, we assume it might return a less specific type for result.summary
// Ensure streamAnalysisChat is implemented correctly as shown in the example provided previously
import { runAnalysis, streamAnalysisChat, getBasicDashboard, getExtendedDashboard } from "@/lib/api"
import { cn } from "@/lib/utils"
// Temporarily remove ReactMarkdown import if testing without it
// import ReactMarkdown from "react-markdown"

// --- Type Definitions ---
type Message = {
  id: string
  role: "user" | "assistant"
  content: string
}

// Define the structure of the summary object returned by the API
type AnalysisSummary = {
  attrition: string;
  clustering: string;
  // Add other potential summary fields here if needed
}

type AnalysisState = {
  status: "idle" | "loading" | "success" | "error"
  summary: AnalysisSummary | null // Use the defined type here
  error: string | null
}

type DashboardState = {
  status: "idle" | "loading" | "success" | "error"
  imageUrl: string | null
  error: string | null
}

// --- Component ---
export function DataAnalysis() {
  const { dataFile, dataStatus } = useDataStore()
  const [analysisState, setAnalysisState] = useState<AnalysisState>({
    status: "idle",
    summary: null, // Initial state is null
    error: null,
  })
  const [basicDashboard, setBasicDashboard] = useState<DashboardState>({
    status: "idle",
    imageUrl: null,
    error: null,
  })
  const [extendedDashboard, setExtendedDashboard] = useState<DashboardState>({
    status: "idle",
    imageUrl: null,
    error: null,
  })
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false) // Tracks chat loading state
  const [error, setError] = useState<string | null>(null) // Tracks chat error
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Effect to display initial analysis summary as the first assistant message
  useEffect(() => {
    // Check status and if summary object exists
    if (analysisState.status === "success" && analysisState.summary) { // Check truthiness first

        // --- Add Logging ---
        // console.log('[Effect] Analysis Summary Type:', typeof analysisState.summary);
        // console.log('[Effect] Analysis Summary Value:', analysisState.summary);
        // --- End Logging ---

        // --- Explicit Type Check ---
        // Verify it's truly an object before accessing properties
        if (typeof analysisState.summary === 'object' && analysisState.summary !== null) {
             // Check if this is the first message setup (avoid adding summary repeatedly)
             if (messages.length === 0 || !messages.some(msg => msg.id.startsWith('analysis-summary-'))) {
                 let formattedSummary = "## Employee Data Analysis Summary\n\n";
                 // Access properties *inside* the confirmed object check
                 if (analysisState.summary.attrition) { // Should be safe now
                     formattedSummary += analysisState.summary.attrition + "\n\n";
                 } else {
                     console.warn("analysisState.summary.attrition is missing or falsy");
                 }
                 if (analysisState.summary.clustering) { // Should be safe now
                     formattedSummary += analysisState.summary.clustering;
                 } else {
                      console.warn("analysisState.summary.clustering is missing or falsy");
                 }

                 // Add the summary message only if it hasn't been added before
                 setMessages((prevMessages) => {
                    // Prevent adding duplicate summary if effect runs multiple times
                    if (!prevMessages.some(msg => msg.id.startsWith('analysis-summary-'))) {
                        // Add summary to the start of the message list
                        return [
                            {
                                id: `analysis-summary-${Date.now()}`,
                                role: "assistant",
                                content: formattedSummary,
                            },
                            ...prevMessages
                        ];
                    }
                    return prevMessages; // Otherwise return existing messages
                 });

             }
        } else {
            // Log if summary is not the expected object type
            console.error("[Effect] analysisState.summary is not an object or is null, despite status being 'success'. Value:", analysisState.summary);
        }
    }
     // Only run when analysisState changes
  }, [analysisState]); // Removed messages.length dependency

  const runDataAnalysis = async () => {
    setAnalysisState({ status: "loading", summary: null, error: null })
    setError(null) // Clear chat error
    setMessages([]) // Clear previous chat messages

    try {
      // Assume runAnalysis might return a less specific type for summary, e.g., Record<string, string>
      const result = await runAnalysis()
      // console.log("API runAnalysis result:", result); // Log API result (Optional)

      // Basic validation before assertion
      if (result && typeof result.summary === 'object' && result.summary !== null && 'attrition' in result.summary && 'clustering' in result.summary) {
          // Store the summary object, using type assertion to satisfy TypeScript
          setAnalysisState({
            status: "success",
            // Add 'as AnalysisSummary' to assert the type
            summary: result.summary as AnalysisSummary,
            error: null,
          })
      } else {
           console.error("API result.summary is not structured as expected:", result?.summary);
           throw new Error("Received invalid summary structure from analysis API.");
      }

    } catch (error) {
      console.error("Error running analysis:", error)
      const errorMessage = error instanceof Error ? error.message : "Failed to run analysis. Please try again."
      setAnalysisState({ status: "error", summary: null, error: errorMessage })
      setError(errorMessage) // Also show error in chat area if needed
    }
  }

  const loadBasicDashboard = async () => {
    setBasicDashboard({ status: "loading", imageUrl: null, error: null })
    setError(null)

    try {
      const blob = await getBasicDashboard()
      const imageUrl = URL.createObjectURL(blob)
      setBasicDashboard({ status: "success", imageUrl, error: null })
    } catch (error) {
      console.error("Error loading basic dashboard:", error)
      setBasicDashboard({ status: "error", imageUrl: null, error: error instanceof Error ? error.message : "Failed to load basic dashboard." })
    }
  }

   // Cleanup function for Object URLs
   useEffect(() => {
    const basicUrl = basicDashboard.imageUrl;
    const extendedUrl = extendedDashboard.imageUrl;
    return () => {
      if (basicUrl) URL.revokeObjectURL(basicUrl);
      if (extendedUrl) URL.revokeObjectURL(extendedUrl);
    };
  }, [basicDashboard.imageUrl, extendedDashboard.imageUrl]);


  const loadExtendedDashboard = async () => {
    setExtendedDashboard({ status: "loading", imageUrl: null, error: null })
    setError(null)

    try {
      const blob = await getExtendedDashboard()
      const imageUrl = URL.createObjectURL(blob)
      setExtendedDashboard({ status: "success", imageUrl, error: null })
    } catch (error) {
      console.error("Error loading extended dashboard:", error)
      setExtendedDashboard({ status: "error", imageUrl: null, error: error instanceof Error ? error.message : "Failed to load extended dashboard." })
    }
  }

  // --- Start: handleSubmit function based on user's corrected logic ---
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    setError(null);

    const userMessage: Message = { id: `user-${Date.now()}`, role: "user", content: input };
    const assistantMessageId = `assistant-${Date.now()}`;
    // Add user message and placeholder for assistant
    setMessages((prev) => [...prev, userMessage, { id: assistantMessageId, role: "assistant", content: "" }]);
    setInput("");
    setIsLoading(true);

    let accumulatedResponse = ""; // Accumulator for the streaming text

    try {
      // Call the async generator function from the API library
      const stream = streamAnalysisChat(input);
      let buffer = ""; // Buffer for incomplete chunks/lines

      // Process the stream chunk by chunk
      for await (const rawChunk of stream) {
        // console.log("Raw chunk received:", JSON.stringify(rawChunk)); // Log raw chunk data
        buffer += rawChunk; // Append new data to buffer

        // Process complete lines (ending with newline) from the buffer
        let newlineIndex;
        while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
          const line = buffer.substring(0, newlineIndex).trim(); // Extract line
          buffer = buffer.substring(newlineIndex + 1); // Update buffer

          if (line) { // Process non-empty lines
            // Try parsing the line as a JSON marker object
            try {
              const jsonMarker = JSON.parse(line);
              // console.log("Parsed JSON marker:", jsonMarker); // Log parsed marker

              // Handle specific marker types
              if (jsonMarker.type === "response_start") {
                console.log("Stream started (marker identified in loop).");
                // Reset accumulator if needed (already done at start)
              } else if (jsonMarker.type === "response_end") {
                console.log("Stream ended (response_end marker identified in loop).");
                // This line is a marker, do not append it to the response text
              } else if (jsonMarker.error) {
                // Handle error markers
                console.error("Stream error marker:", jsonMarker.error);
                accumulatedResponse += `\n[Error: ${jsonMarker.error}]`; // Append error info
                setError(jsonMarker.error);
                // Update state immediately to show error
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId ? { ...msg, content: accumulatedResponse } : msg
                  )
                );
              } else {
                // Handle unknown JSON structures if necessary
                console.warn("Received unknown JSON object in stream:", jsonMarker);
              }
            } catch (e) {
              // If JSON parsing fails, assume it's a regular text chunk
              // console.log("Processing text chunk:", line); // Log text chunk processing
              accumulatedResponse += line; // Append the text content
              // Update the corresponding message in the state
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId ? { ...msg, content: accumulatedResponse } : msg
                )
              );
            }
          }
        } // End while loop for processing lines
      } // End for await loop for processing stream chunks

      // Handle any remaining content in the buffer after the stream ends
      // This might catch text not ending with a newline, or a marker without a newline
      const remainingBufferContent = buffer.trim();
      if (remainingBufferContent) {
        // console.log("Processing remaining buffer:", JSON.stringify(remainingBufferContent));
        let isFinalMarker = false;
        // Check if the remaining content is a JSON marker
        try {
          const jsonMarker = JSON.parse(remainingBufferContent);
          // console.log("Final buffer contains JSON:", jsonMarker);
          // Check if it's a known marker type we should ignore/handle
          if (jsonMarker.type === "response_start" || jsonMarker.type === "response_end") {
             isFinalMarker = true;
          } else if (jsonMarker.error) {
             // Handle final error marker
             isFinalMarker = true;
             accumulatedResponse += `\n[Error: ${jsonMarker.error}]`;
             setError(jsonMarker.error);
             // Final update needed
             setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessageId ? { ...msg, content: accumulatedResponse } : msg
                )
              );
          }
          // If it's some other JSON, we might ignore it or log it
        } catch (e) {
          // Not JSON, assume it's regular text to be appended
          isFinalMarker = false;
        }

        // Append remaining buffer only if it wasn't identified as a marker
        if (!isFinalMarker) {
            accumulatedResponse += remainingBufferContent;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId ? { ...msg, content: accumulatedResponse } : msg
              )
            );
        }
      }

    } catch (error) {
      // Handle errors during the fetch/stream processing
      console.error("Error in chat stream processing:", error);
      const errorMsg = error instanceof Error ? error.message : String(error);
      setError(errorMsg);
      // Update the assistant message placeholder with an error message
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, content: `${accumulatedResponse}\n[Error: ${errorMsg}]` } // Show accumulated + error
            : msg
        )
      );
    } finally {
      setIsLoading(false); // Set loading state to false

      // --- Final Safety Cleanup (Regex Workaround) ---
      // This runs after all stream processing and attempts to remove
      // any stray markers that might have slipped through.
      setMessages((prev) => {
        return prev.map(msg => {
          if (msg.id === assistantMessageId) {
            // Regex to find markers possibly surrounded by whitespace at the end
            const cleanedContent = msg.content
              .replace(/\s*\{"type":\s*"response_end"\}\s*$/g, '')
              .replace(/\s*\{"type":\s*"response_start"\}\s*$/g, '') // Less likely but safe to include
              .trimEnd(); // Trim potential trailing whitespace after removal

            // Only update if content actually changed
            if (cleanedContent !== msg.content) {
                 console.warn("Applying final regex cleanup for marker.");
                 return { ...msg, content: cleanedContent };
            }
          }
          return msg; // Return other messages unchanged
        });
      });
      // --- End Final Cleanup ---
    }
  };
  // --- End: handleSubmit function ---


  const isDataReady = dataStatus === "ready"
  const isAnalysisReady = analysisState.status === "success"

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Buttons Card */}
      <Card className="bg-card/50 backdrop-blur-sm border-primary/10">
        <CardContent className="p-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Button
              className="flex items-center gap-2 bg-primary/90 hover:bg-primary"
              onClick={runDataAnalysis}
              disabled={!isDataReady || analysisState.status === "loading"}
            >
              {analysisState.status === "loading" ? (
                 <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                 <BarChart3 className="w-4 h-4" />
              )}
              {analysisState.status === "loading" ? "Analyzing..." : "Run ML Analysis"}
            </Button>
            <Button
              className="flex items-center gap-2"
              onClick={loadBasicDashboard}
              disabled={!isDataReady || basicDashboard.status === "loading"}
              variant="outline"
            >
               {basicDashboard.status === "loading" ? (
                 <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                 <BarChart3 className="w-4 h-4" />
              )}
              {basicDashboard.status === "loading" ? "Loading..." : "Show Basic Dashboard"}
            </Button>
            <Button
              className="flex items-center gap-2"
              onClick={loadExtendedDashboard}
              disabled={!isDataReady || extendedDashboard.status === "loading"}
              variant="outline"
            >
              {extendedDashboard.status === "loading" ? (
                 <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                 <LineChart className="w-4 h-4" />
              )}
              {extendedDashboard.status === "loading" ? "Loading..." : "Show Extended Dashboard"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Status - Only show error here, loading is on button */}
      {analysisState.status === "error" && (
        <Alert variant="destructive">
          <AlertCircle className="w-4 h-4" />
          <AlertDescription>{analysisState.error}</AlertDescription>
        </Alert>
      )}

      {/* Dashboard Display */}
      {(basicDashboard.status === "success" || extendedDashboard.status === "success" || basicDashboard.status === "loading" || extendedDashboard.status === "loading") && (
        <Card className="bg-card/50 backdrop-blur-sm border-primary/10 overflow-hidden">
          <Tabs defaultValue={basicDashboard.status !== "idle" ? "basic" : "extended"}>
            <div className="px-4 pt-4">
              <TabsList className="w-full grid grid-cols-2">
                <TabsTrigger value="basic" disabled={basicDashboard.status === "idle" && extendedDashboard.status !== 'success'}>
                  Basic Dashboard
                </TabsTrigger>
                <TabsTrigger value="extended" disabled={extendedDashboard.status === "idle" && basicDashboard.status !== 'success'}>
                  Extended Dashboard
                </TabsTrigger>
              </TabsList>
            </div>
            <TabsContent value="basic">
              {basicDashboard.status === "loading" && <div className="p-6 text-center"><Loader2 className="w-6 h-6 animate-spin inline-block"/></div>}
              {basicDashboard.status === "success" && basicDashboard.imageUrl && (
                <div>
                  <CardHeader>
                    <CardTitle>Basic Employee Data Dashboard</CardTitle>
                    <CardDescription>Overview of key employee metrics</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-auto rounded-lg border border-border/30">
                      <img
                        src={basicDashboard.imageUrl} // Removed placeholder fallback
                        alt="Basic Dashboard"
                        className="w-full h-auto"
                        onError={(e) => e.currentTarget.src = "/placeholder.svg"} // Add placeholder on error
                      />
                    </div>
                  </CardContent>
                </div>
              )}
              {basicDashboard.status === "error" && (
                <Alert variant="destructive" className="m-4">
                  <AlertCircle className="w-4 h-4" />
                  <AlertDescription>{basicDashboard.error}</AlertDescription>
                </Alert>
              )}
            </TabsContent>
            <TabsContent value="extended">
             {extendedDashboard.status === "loading" && <div className="p-6 text-center"><Loader2 className="w-6 h-6 animate-spin inline-block"/></div>}
              {extendedDashboard.status === "success" && extendedDashboard.imageUrl && (
                <div>
                  <CardHeader>
                    <CardTitle>Extended Employee Data Dashboard</CardTitle>
                    <CardDescription>Detailed analysis with advanced metrics</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-auto rounded-lg border border-border/30">
                      <img
                        src={extendedDashboard.imageUrl} // Removed placeholder fallback
                        alt="Extended Dashboard"
                        className="w-full h-auto"
                        onError={(e) => e.currentTarget.src = "/placeholder.svg"} // Add placeholder on error
                      />
                    </div>
                  </CardContent>
                </div>
              )}
              {extendedDashboard.status === "error" && (
                <Alert variant="destructive" className="m-4">
                  <AlertCircle className="w-4 h-4" />
                  <AlertDescription>{extendedDashboard.error}</AlertDescription>
                </Alert>
              )}
            </TabsContent>
          </Tabs>
        </Card>
      )}

      {/* Analysis Chat */}
      {isAnalysisReady && (
        <Card className="bg-card/50 backdrop-blur-sm border-primary/10">
          <CardHeader className="pb-2">
            <CardTitle>Analysis Insights & Chat</CardTitle>
            <CardDescription>Ask follow-up questions about the analysis</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[400px] flex flex-col">
              {/* Chat Messages Area */}
              <div className="flex-1 overflow-y-auto pr-4 -mr-4 mb-4"> {/* Added margin-bottom */}
                <div className="space-y-4 py-4">
                  {messages.map((message) => (
                    <div key={message.id} className={cn("flex gap-3", message.role === "user" ? "justify-end" : "items-start")}>
                      {message.role === "assistant" && (
                        <Avatar className="mt-1 flex-shrink-0">
                          <AvatarFallback className="bg-primary/20 text-primary">AI</AvatarFallback>
                        </Avatar>
                      )}
                      <Card
                        className={cn(
                          "max-w-[80%] rounded-xl", // Softer corners
                          message.role === "user"
                            ? "bg-primary text-primary-foreground border-primary/50"
                            : "bg-background/70 backdrop-blur-sm border", // Use background for AI
                        )}
                      >
                        <CardContent className="p-3 text-sm"> {/* Ensure consistent text size */}
                          {/* --- TEMP: Render directly to test --- */}
                           <p style={{ whiteSpace: "pre-wrap" }}>{message.content || (isLoading && message.id.startsWith('assistant-') ? "..." : "")}</p>
                          {/* --- Original ReactMarkdown (Uncomment after testing) ---
                           <div className="prose prose-sm dark:prose-invert max-w-none">
                             <ReactMarkdown>
                               {message.content || (isLoading && message.id.startsWith('assistant-') ? "..." : "")}
                             </ReactMarkdown>
                           </div>
                           */}
                        </CardContent>
                      </Card>
                      {message.role === "user" && (
                        <Avatar className="mt-1 flex-shrink-0">
                          <AvatarFallback>U</AvatarFallback>
                        </Avatar>
                      )}
                    </div>
                  ))}
                  {/* Separate Loading indicator if needed, handled by empty message content now */}
                  {/* Chat Error Display */}
                   {error && !isLoading && ( // Show error only when not loading
                     <Alert variant="destructive" className="max-w-3xl mt-4">
                       <AlertCircle className="h-4 w-4" />
                       <AlertDescription>{error}</AlertDescription>
                     </Alert>
                   )}
                  <div ref={messagesEndRef} />
                </div>
              </div>

              {/* Input Area */}
              <div className="sticky bottom-0 pt-2 border-t border-border/20"> {/* Added border */}
                <form onSubmit={handleSubmit} className="flex gap-2 items-center">
                  <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask follow-up questions..." // Shorter placeholder
                    className="min-h-[40px] max-h-[100px] resize-none bg-background/50 backdrop-blur-sm flex-1 rounded-lg" // Rounded
                    disabled={isLoading}
                    rows={1} // Start with 1 row
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault()
                        handleSubmit(e)
                      }
                    }}
                  />
                  <Button
                    type="submit"
                    size="icon"
                    className="bg-primary/90 hover:bg-primary rounded-lg" // Rounded
                    disabled={isLoading || !input.trim()}
                  >
                    {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                    <span className="sr-only">Send message</span>
                  </Button>
                </form>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Initial State Message */}
      {!isDataReady && (
        <div className="flex flex-col items-center justify-center p-12 text-center">
          <div className="flex items-center justify-center w-16 h-16 mb-4 rounded-full bg-primary/10">
            <BarChart3 className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-xl font-semibold">Data Analysis</h2>
          <p className="max-w-md mt-2 text-sm text-muted-foreground">
            Upload employee data using the sidebar to start analyzing and visualizing insights.
          </p>
        </div>
      )}
    </div>
  )
}
