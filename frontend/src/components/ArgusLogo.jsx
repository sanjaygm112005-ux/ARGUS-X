import React from "react";
import { ShieldCheck } from "lucide-react";

export function ArgusLogo({ size = "md" }) {
  const iconCls = size === "lg" ? "w-8 h-8" : "w-6 h-6";
  const textCls = size === "lg" ? "text-2xl" : "text-lg";
  return (
    <div className="inline-flex items-center gap-2.5 select-none">
      <div className="relative flex items-center justify-center rounded-xl p-2 bg-gradient-to-br from-sky-500/20 to-violet-600/30 border border-sky-400/30 shadow-[0_0_20px_rgba(56,189,248,0.25)]">
        <ShieldCheck className={`${iconCls} text-sky-400`} />
      </div>
      <div className="leading-tight">
        <span className={`font-display font-bold tracking-wider text-slate-900 dark:text-white ${textCls}`}>
          ARGUS<span className="text-sky-500 dark:text-sky-400">-X</span>
        </span>
        <div className="text-[10px] uppercase tracking-[0.22em] text-slate-500 dark:text-slate-400">
          Track the Untrackable
        </div>
      </div>
    </div>
  );
}
