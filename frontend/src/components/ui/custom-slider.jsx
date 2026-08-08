import React from "react"
import { cn } from "@/lib/utils"

export function CustomSlider({ min, max, step, value, onChange, label, displayValue, className }) {
  // Calculate percentage for the filled track
  const percentage = ((value - min) / (max - min)) * 100

  return (
    <div className={cn("flex flex-col space-y-3", className)}>
      <div className="flex justify-between items-end">
        <label className="text-sm font-medium text-foreground">{label}</label>
        <span className="text-sm font-bold text-primary">{displayValue || value}</span>
      </div>
      <div className="relative w-full h-2 bg-secondary rounded-full">
        {/* Filled Track */}
        <div 
          className="absolute top-0 left-0 h-full bg-primary rounded-full"
          style={{ width: `${percentage}%` }}
        />
        {/* Native range input overlay (invisible but interactive) */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={onChange}
          className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
        />
        {/* Custom Thumb */}
        <div 
          className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white border-2 border-primary rounded-full shadow pointer-events-none"
          style={{ left: `calc(${percentage}% - 8px)` }}
        />
      </div>
    </div>
  )
}
