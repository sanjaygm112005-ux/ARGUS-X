import React, { useState } from "react";
import {
  Video,
  ScanSearch,
  Activity,
  Fingerprint,
  Database,
  GitMerge,
  ShieldCheck,
  MonitorCheck,
  ArrowDown,
} from "lucide-react";

const STAGES = [
  {
    title: "VIDEO INPUT",
    detail: "Multi-camera CCTV / RTSP / local video streams managed by CameraManager.",
    icon: Video,
  },
  {
    title: "YOLO11 OBJECT DETECTION",
    detail: "Real-time person bounding-box localization per frame (detection/detector.py).",
    icon: ScanSearch,
  },
  {
    title: "BoT-SORT LOCAL TRACKING",
    detail: "Independent per-camera short-horizon track IDs (tracking/tracker.py).",
    icon: Activity,
  },
  {
    title: "OSNet FEATURE EXTRACTION",
    detail: "512-dimensional L2-normalized appearance embeddings (osnet_x0_25).",
    icon: Fingerprint,
  },
  {
    title: "FAISS IDENTITY MEMORY",
    detail: "Cosine similarity vector gallery with SQLite metadata persistence.",
    icon: Database,
  },
  {
    title: "HUNGARIAN DATA ASSOCIATION",
    detail: "Confidence-aware assignment combining appearance, time, camera, and motion.",
    icon: GitMerge,
  },
  {
    title: "GLOBAL IDENTITY RECOVERY",
    detail: "Lifecycle states (ACTIVE, TEMPORARILY_LOST, RECOVERED, UNCERTAIN, EXPIRED).",
    icon: ShieldCheck,
  },
  {
    title: "OPERATOR REVIEW & DASHBOARD",
    detail: "Conflict prevention, human-in-the-loop review, and audit logging.",
    icon: MonitorCheck,
  },
];

export function PipelineVisualizer() {
  const [activeIdx, setActiveIdx] = useState(0);

  return (
    <div className="grid lg:grid-cols-12 gap-6 items-start">
      <div className="lg:col-span-7 space-y-2">
        {STAGES.map((stage, idx) => {
          const Icon = stage.icon;
          const isActive = idx === activeIdx;
          return (
            <React.Fragment key={stage.title}>
              <button
                type="button"
                onClick={() => setActiveIdx(idx)}
                className={`w-full text-left glass-card rounded-xl p-4 flex items-center gap-4 transition shadow-sm ${
                  isActive
                    ? "ring-2 ring-sky-500 dark:ring-sky-400/70 bg-sky-50 dark:bg-sky-500/10"
                    : "hover:bg-slate-50 dark:hover:bg-slate-800/40"
                }`}
              >
                <div
                  className={`p-2.5 rounded-lg ${
                    isActive
                      ? "bg-sky-100 dark:bg-sky-500/25 text-sky-700 dark:text-sky-300"
                      : "bg-slate-100 dark:bg-slate-800/80 text-slate-600 dark:text-slate-400"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <div className="text-xs font-mono text-sky-600 dark:text-sky-400 font-semibold">STAGE 0{idx + 1}</div>
                  <div className="font-display font-semibold text-slate-900 dark:text-white">{stage.title}</div>
                </div>
              </button>
              {idx < STAGES.length - 1 && (
                <div className="flex justify-center py-0.5">
                  <ArrowDown className="w-4 h-4 text-violet-500 dark:text-violet-400/70 animate-pulse" />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      <div className="lg:col-span-5 glass-card rounded-2xl p-6 sticky top-24 shadow-sm">
        <div className="text-xs font-mono uppercase tracking-widest text-violet-600 dark:text-violet-400 font-semibold">
          Selected Pipeline Stage
        </div>
        <h3 className="mt-2 font-display text-2xl font-bold text-slate-900 dark:text-white">
          {STAGES[activeIdx].title}
        </h3>
        <p className="mt-3 text-slate-700 dark:text-slate-300 leading-relaxed">{STAGES[activeIdx].detail}</p>
        <div className="mt-6 pt-4 border-t border-slate-200 dark:border-slate-800 text-xs text-slate-500 dark:text-slate-400">
          Click any stage in the pipeline to inspect how ARGUS-X preserves identity continuity without exposing raw biometric vectors.
        </div>
      </div>
    </div>
  );
}
