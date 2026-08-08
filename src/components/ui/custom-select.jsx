import React from "react"
import { ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

export function CustomSelect({ value, onChange, label, options, className }) {
  return (
    <div className={cn("flex flex-col space-y-2", className)}>
      <label className="text-sm font-medium text-muted-foreground">{label}</label>
      <div className="relative">
        <select
          value={value}
          onChange={onChange}
          className="w-full appearance-none bg-secondary border border-border text-foreground text-sm rounded-xl px-4 py-3 pr-10 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-colors cursor-pointer"
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none text-muted-foreground">
          <ChevronDown className="w-4 h-4" />
        </div>
      </div>
    </div>
  )
}
