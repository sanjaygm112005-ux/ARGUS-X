import React from "react";
import { ArrowRight } from "lucide-react";

const ARCH_NODES = [
  { name: "Camera Streams", sub: "camera/camera_manager.py" },
  { name: "Detection", sub: "YOLO11n" },
  { name: "Tracking", sub: "BoT-SORT" },
  { name: "Re-identification", sub: "OSNet x0.25" },
  { name: "Identity Memory", sub: "FAISS + SQLite" },
  { name: "Association", sub: "Hungarian Cost Matrix" },
  { name: "Global Identity Manager", sub: "Conflict & Lifecycle" },
  { name: "Dashboard", sub: "React + FastAPI + WS" },
];

export function ArchitectureDiagram() {
  return (
    <div className="glass-card rounded-2xl p-6 overflow-x-auto">
      <div className="flex flex-wrap lg:flex-nowrap items-center justify-between gap-3 min-w-[680px]">
        {ARCH_NODES.map((node, i) => (
          <React.Fragment key={node.name}>
            <div className="flex-1 min-w-[130px] rounded-xl bg-sky-50/80 dark:bg-[#0a122e] border border-sky-200 dark:border-sky-400/25 p-3.5 text-center shadow-sm dark:shadow-lg dark:shadow-sky-950/30 transition-colors">
              <div className="text-xs font-display font-semibold text-slate-900 dark:text-white">{node.name}</div>
              <div className="mt-1 text-[11px] font-mono text-sky-700 dark:text-sky-300/80">{node.sub}</div>
            </div>
            {i < ARCH_NODES.length - 1 && (
              <ArrowRight className="w-4 h-4 text-violet-600 dark:text-violet-400 shrink-0" />
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
