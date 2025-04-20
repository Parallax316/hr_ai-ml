"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Send, Loader2, FileText, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent } from "@/components/ui/card"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { usePolicyStore } from "@/lib/stores/policy-store"
// Ensure streamPolicyChat is implemented correctly, similar to streamAnalysisChat
import { streamPolicyChat } from "@/lib/api"
import { cn } from "@/lib/utils"
import ReactMarkdown from "react-markdown" // Keep ReactMarkdown

// --- Type Definition ---
type Message = {
  id: string
  role: "user" | "assistant"
  content: string
}

// --- Component ---
export function PolicyChat() {
  const { policyFile, policyStatus } = usePolicyStore()
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

  // --- Start: handleSubmit function with robust streaming logic ---
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    setError(null); // Clear previous errors

    const userMessage: Message = { id: `user-${Date.now()}`, role: "user", content: input };
    const assistantMessageId = `assistant-${Date.now()}`;
    // Add user message and placeholder for assistant
    setMessages((prev) => [...prev, userMessage, { id: assistantMessageId, role: "assistant", content: "" }]);
    setInput("");
    setIsLoading(true);

    let accumulatedResponse = ""; // Accumulator for the streaming text

    try {
      // Call the async generator function for policy chat
      const stream = streamPolicyChat(input);
      let buffer = ""; // Buffer for incomplete chunks/lines

      // Process the stream chunk by chunk
      for await (const rawChunk of stream) {
        // console.log("Raw policy chunk received:", JSON.stringify(rawChunk)); // Optional: Log raw chunk data
        buffer += rawChunk; // Append new data to buffer

        // Process complete lines (ending with newline) from the buffer
        let newlineIndex;
        while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
          const line = buffer.substring(0, newlineIndex).trim(); // Extract line
          buffer = buffer.substring(newlineIndex + 1); // Update buffer

          if (line) { // Process non-empty lines
            let isTextToAppend = true; // Assume line is text unless proven otherwise
            // Try parsing the line as a JSON marker object
            try {
              const jsonMarker = JSON.parse(line);
              // console.log("Parsed policy JSON marker:", jsonMarker); // Optional: Log parsed marker

              // Handle specific marker types
              if (jsonMarker.type === "response_start") {
                console.log("Policy stream started (marker identified in loop).");
                accumulatedResponse = ""; // Reset accumulator on start
                isTextToAppend = false;
              } else if (jsonMarker.type === "response_end") {
                console.log("Policy stream ended (response_end marker identified in loop).");
                isTextToAppend = false; // This line is a marker, do not append
              } else if (jsonMarker.error) {
                // Handle error markers
                console.error("Policy stream error marker:", jsonMarker.error);
                accumulatedResponse += `\n[Error: ${jsonMarker.error}]`; // Append error info
                setError(jsonMarker.error);
                isTextToAppend = false; // Error marker handled, don't append raw line
                // Update state immediately to show error
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessageId ? { ...msg, content: accumulatedResponse } : msg
                  )
                );
              } else {
                // Handle unknown JSON structures if necessary
                console.warn("Received unknown JSON object in policy stream:", jsonMarker);
                isTextToAppend = false; // Don't append unknown JSON structure
              }
            } catch (e) {
              // If JSON parsing fails, assume it's a regular text chunk
              isTextToAppend = true;
            }

             // Append ONLY if it was determined to be text
            if (isTextToAppend) {
              // console.log("Processing policy text chunk:", line); // Optional: Log text chunk processing
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
      const remainingBufferContent = buffer.trim();
      if (remainingBufferContent) {
        // console.log("Processing remaining policy buffer:", JSON.stringify(remainingBufferContent));
        let isFinalMarker = false;
        // Check if the remaining content is a JSON marker
        try {
          const jsonMarker = JSON.parse(remainingBufferContent);
          // console.log("Final policy buffer contains JSON:", jsonMarker);
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
      console.error("Error in policy chat stream processing:", error);
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
                 console.warn("Applying final regex cleanup for policy marker.");
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


  const isPolicyReady = policyStatus === "ready"

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] max-w-4xl mx-auto">
      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto pr-4 -mr-4 mb-4"> {/* Added margin-bottom */}
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="flex items-center justify-center w-16 h-16 mb-4 rounded-full bg-primary/10">
              <FileText className="w-8 h-8 text-primary" />
            </div>
            <h2 className="text-xl font-semibold">Policy Chat Assistant</h2>
            <p className="max-w-md mt-2 text-sm text-muted-foreground">
              {isPolicyReady
                ? "Your policy document is ready. Ask questions about the policy to get started."
                : "Upload a policy document using the sidebar to start chatting about it."}
            </p>
          </div>
        ) : (
          <div className="space-y-4 py-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex gap-3 max-w-3xl",
                  message.role === "user" ? "ml-auto justify-end" : "items-start" // Ensure AI messages align left
                )}
              >
                {message.role === "assistant" && (
                  <Avatar className="mt-1 flex-shrink-0">
                    <AvatarFallback className="bg-primary/20 text-primary">AI</AvatarFallback>
                  </Avatar>
                )}
                <Card
                  className={cn(
                    "max-w-[85%] rounded-xl", // Softer corners
                    message.role === "user"
                      ? "bg-primary text-primary-foreground border-primary/50"
                      : "bg-background/70 backdrop-blur-sm border", // Use background for AI
                  )}
                >
                  <CardContent className="p-3 text-sm"> {/* Consistent text size */}
                     {/* Use ReactMarkdown for assistant messages */}
                     {message.role === "assistant" ? (
                       <div className="prose prose-sm dark:prose-invert max-w-none">
                          {/* Add remark-gfm if needed for tables, etc. `npm install remark-gfm` */}
                         <ReactMarkdown>
                           {message.content || (isLoading && message.id.startsWith('assistant-') ? "..." : "")}
                         </ReactMarkdown>
                       </div>
                     ) : (
                       <p>{message.content}</p> // Render user message as plain text
                     )}
                  </CardContent>
                </Card>
                {message.role === "user" && (
                  <Avatar className="mt-1 flex-shrink-0">
                    <AvatarFallback>U</AvatarFallback>
                  </Avatar>
                )}
              </div>
            ))}
            {/* Loading indicator logic might need adjustment if placeholder isn't used */}
            {isLoading && messages[messages.length -1]?.role !== 'assistant' && ( // Show thinking only if last msg isn't empty assistant placeholder
               <div className="flex items-start gap-3 max-w-3xl">
                 <Avatar className="mt-1">
                   <AvatarFallback className="bg-primary/20 text-primary">AI</AvatarFallback>
                 </Avatar>
                 <Card className="bg-card/50 backdrop-blur-sm">
                   <CardContent className="p-3">
                     <div className="flex items-center gap-2">
                       <Loader2 className="w-4 h-4 animate-spin" />
                       <span className="text-sm text-muted-foreground">Thinking...</span>
                     </div>
                   </CardContent>
                 </Card>
               </div>
             )}
             {/* Chat Error Display */}
             {error && !isLoading && ( // Show error only when not loading
               <Alert variant="destructive" className="max-w-3xl mt-4">
                 <AlertCircle className="h-4 w-4" />
                 <AlertDescription>{error}</AlertDescription>
               </Alert>
             )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="sticky bottom-0 pt-4 border-t border-border/20 bg-background/80 backdrop-blur-sm"> {/* Added border & background */}
        <form onSubmit={handleSubmit} className="flex gap-2 items-center">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isPolicyReady ? "Ask about the uploaded policy..." : "Upload a policy document first..."}
            className="min-h-[40px] max-h-[100px] resize-none bg-background/50 backdrop-blur-sm flex-1 rounded-lg" // Rounded
            disabled={!isPolicyReady || isLoading}
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
            disabled={!isPolicyReady || isLoading || !input.trim()}
          >
             {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            <span className="sr-only">Send message</span>
          </Button>
        </form>
      </div>
    </div>
  )
}
