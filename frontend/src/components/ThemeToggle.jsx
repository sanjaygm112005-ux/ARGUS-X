import React from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "../context/ThemeContext";

export function ThemeToggle({ className = "" }) {
  const { theme, toggleTheme, isDark } = useTheme();

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={`Switch to ${isDark ? "light" : "dark"} mode`}
      title={`Switch to ${isDark ? "light" : "dark"} mode`}
      className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl border transition-all duration-200 ${
        isDark
          ? "bg-slate-800/80 hover:bg-slate-700/80 text-amber-300 border-slate-700/80 shadow-sm"
          : "bg-slate-100 hover:bg-slate-200/90 text-amber-600 border-slate-300 shadow-sm"
      } ${className}`}
    >
      {isDark ? (
        <>
          <Sun className="w-4 h-4 text-amber-400 animate-spin-slow" />
          <span className="text-xs font-semibold text-slate-200 hidden sm:inline">Light</span>
        </>
      ) : (
        <>
          <Moon className="w-4 h-4 text-slate-700" />
          <span className="text-xs font-semibold text-slate-700 hidden sm:inline">Dark</span>
        </>
      )}
    </button>
  );
}
