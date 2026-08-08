import { BrowserRouter, Routes, Route } from "react-router-dom"
import Dashboard from "./pages/Dashboard"
import Navbar from "./components/Navbar"
import Footer from "./components/Footer"

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background text-foreground flex flex-col font-sans">
        <Navbar />
        <main className="flex-1 overflow-x-hidden">
          <Routes>
            <Route path="/" element={<Dashboard />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  )
}

export default App
