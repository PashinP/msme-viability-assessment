import { useState, useRef, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import { Send, Bot, User, Mic } from "lucide-react"
import axios from "axios"
import { Button } from "@/components/ui/button"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"
const API_KEY = import.meta.env.VITE_API_KEY || "msme-dev-key-2024"

export default function ChatCoach() {
  const navigate = useNavigate()
  const [messages, setMessages] = useState([
    { role: "assistant", content: "👋 नमस्ते! Welcome to the MSME Loan Coach. Please tell me a bit about your business — what do you do, how many employees do you have, and how much loan are you looking for?" }
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e?.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { role: "user", content: input.trim() }
    setMessages(prev => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const historyForApi = messages.map(m => ({
        role: m.role === "assistant" ? "model" : "user", // Google/FastAPI expects 'model'
        content: m.content
      })).concat([{ role: "user", content: userMessage.content }])

      const response = await axios.post(`${API_URL}/chat`, { messages: historyForApi }, {
        headers: { "X-API-Key": API_KEY }
      })

      const { response: aiResponse, extraction_complete, features_extracted } = response.data

      setMessages(prev => [...prev, { role: "assistant", content: aiResponse }])

      if (extraction_complete && features_extracted) {
        setTimeout(() => {
          navigate("/result", { state: { features: features_extracted } })
        }, 2000) // Brief delay so they can read the final message
      }
    } catch (error) {
      console.error("Chat error:", error)
      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: "⚠️ I'm having trouble connecting to the server. Please try again or use Expert Mode." 
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container max-w-4xl mx-auto h-[calc(100vh-4rem)] flex flex-col py-6">
      
      {/* Header */}
      <div className="flex items-center gap-4 pb-6 border-b border-border">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-purple to-accent-blue flex items-center justify-center shadow-lg">
          <Bot className="w-7 h-7 text-white" />
        </div>
        <div>
          <h2 className="text-2xl font-bold">MSME Loan Readiness Coach</h2>
          <p className="text-muted-foreground">Conversational assessment powered by Llama 3.3 70B</p>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto py-6 space-y-6 scrollbar-hide pr-2">
        <AnimatePresence>
          {messages.map((msg, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex items-start gap-4 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
            >
              <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
                msg.role === "user" ? "bg-accent-blue/20" : "bg-accent-purple/20"
              }`}>
                {msg.role === "user" ? <User className="w-5 h-5 text-accent-blue" /> : <Bot className="w-5 h-5 text-accent-purple" />}
              </div>
              <div className={`px-5 py-3.5 rounded-2xl max-w-[80%] text-sm md:text-base leading-relaxed ${
                msg.role === "user" 
                  ? "bg-primary text-primary-foreground rounded-tr-sm" 
                  : "glass-panel rounded-tl-sm text-foreground"
              }`}>
                {msg.content.split('\n').map((line, i) => <span key={i}>{line}<br/></span>)}
              </div>
            </motion.div>
          ))}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-start gap-4"
            >
              <div className="w-10 h-10 rounded-full bg-accent-purple/20 flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-accent-purple" />
              </div>
              <div className="glass-panel px-5 py-4 rounded-2xl rounded-tl-sm flex items-center gap-1.5">
                <span className="w-2 h-2 bg-accent-purple rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-2 h-2 bg-accent-purple rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-2 h-2 bg-accent-purple rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="pt-4 mt-auto">
        <form onSubmit={handleSubmit} className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message in Hindi or English..."
            className="w-full bg-secondary border border-border rounded-full pl-6 pr-24 py-4 text-base focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all shadow-sm"
            disabled={isLoading}
          />
          <div className="absolute right-2 flex items-center gap-1">
            <Button type="button" size="icon" variant="ghost" className="rounded-full text-muted-foreground hover:text-foreground hover:bg-background">
              <Mic className="w-5 h-5" />
            </Button>
            <Button type="submit" size="icon" className="rounded-full bg-primary hover:bg-primary/90 shadow-md" disabled={!input.trim() || isLoading}>
              <Send className="w-5 h-5 ml-0.5" />
            </Button>
          </div>
        </form>
      </div>

    </div>
  )
}
