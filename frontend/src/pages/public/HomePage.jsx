import React from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  Camera,
  Cpu,
  Database,
  EyeOff,
  Fingerprint,
  GitBranch,
  Layers,
  Lock,
  Radio,
  RefreshCw,
  ShieldAlert,
  Sparkles,
  UserCheck,
} from "lucide-react";
import { PipelineVisualizer } from "../../components/PipelineVisualizer";
import { ArchitectureDiagram } from "../../components/ArchitectureDiagram";

const PROBLEMS = [
  { title: "Occlusion Dropouts", desc: "People disappearing behind pillars, crowds, or doorways break continuous bounding boxes.", icon: EyeOff },
  { title: "Camera Transitions", desc: "Non-overlapping camera fields reset local tracker IDs whenever a subject moves between zones.", icon: Camera },
  { title: "Appearance Drift", desc: "Lighting shifts, viewing angle changes, and posture differences degrade naive matching.", icon: Sparkles },
  { title: "Long-Term Failures", desc: "Short-horizon trackers forget identities within seconds once a track buffer expires.", icon: Activity },
  { title: "Conflicting Identities", desc: "Similar-looking people trigger false merges or duplicate global IDs without conflict guards.", icon: ShieldAlert },
];

const TECH_STACK = [
  { name: "Python", role: "Core AI orchestration & async services" },
  { name: "YOLO11", role: "Real-time person object detection" },
  { name: "BoT-SORT", role: "Per-camera short-horizon local tracking" },
  { name: "OSNet", role: "512-d L2-normalized person Re-ID embeddings" },
  { name: "FAISS", role: "IndexFlatIP high-speed vector similarity memory" },
  { name: "SciPy Hungarian Algorithm", role: "Optimal confidence-aware track-to-identity assignment" },
  { name: "OpenCV", role: "Video capture, frame decoding & overlay rendering" },
  { name: "FastAPI", role: "Authenticated REST & WebSocket backend" },
  { name: "Streamlit", role: "Annotation & diagnostic tooling" },
  { name: "SQLite / PostgreSQL", role: "Isolated biometric metadata & Neon app DB" },
];

const FEATURES = [
  { title: "Multi-Camera Tracking", desc: "Independent local trackers per camera feeding one shared identity space.", icon: Camera },
  { title: "Long-Term Identity Memory", desc: "FAISS gallery + SQLite persistence retain identities across extended gaps.", icon: Database },
  { title: "Cross-Camera Recovery", desc: "Transition-aware matching reconnects subjects across distinct camera views.", icon: GitBranch },
  { title: "Occlusion Recovery", desc: "Re-associates fragmented tracklets after temporary visual obstructions.", icon: RefreshCw },
  { title: "Appearance-Drift Handling", desc: "Multi-observation galleries adapt to viewpoint and lighting variations.", icon: Fingerprint },
  { title: "Conflict Prevention", desc: "Blocks incompatible simultaneous assignments and unsafe auto-merges.", icon: Lock },
  { title: "Manual Operator Review", desc: "Uncertain matches route to a human-in-the-loop review queue.", icon: UserCheck },
  { title: "Audit Logging", desc: "Every administrative action and identity decision is recorded.", icon: Layers },
];

export function HomePage() {
  const dashboardUrl = import.meta.env.VITE_DASHBOARD_URL || "/login";
  const isExternalDash = dashboardUrl.startsWith("http");

  return (
    <div className="space-y-24 pb-12">
      {/* 1. HERO SECTION */}
      <section className="pt-16 sm:pt-24 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto text-center">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-sky-500/10 border border-sky-400/30 text-sky-300 text-xs font-mono tracking-wider mb-6">
          <Radio className="w-3.5 h-3.5 animate-pulse" />
          LONG-HORIZON MULTI-OBJECT IDENTITY RECOVERY
        </div>
        <h1 className="font-display text-5xl sm:text-7xl font-bold tracking-tight text-slate-900 dark:text-white">
          ARGUS<span className="text-sky-500 dark:text-sky-400">-X</span>
        </h1>
        <p className="mt-3 font-display text-xl sm:text-2xl font-semibold tracking-[0.18em] text-transparent bg-clip-text bg-gradient-to-r from-sky-500 to-violet-600 dark:from-sky-400 dark:to-violet-400">
          TRACK THE UNTRACKABLE.
        </p>
        <p className="mt-4 text-sm sm:text-base text-slate-600 dark:text-slate-400 font-medium">
          Long-Horizon Multi-Object Identity Recovery Under Camera, Occlusion &amp; Appearance Drift.
        </p>
        <p className="mt-5 max-w-3xl mx-auto text-base sm:text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
          “An intelligent multi-camera tracking and identity recovery system designed to recognize and reconnect identities across cameras, occlusions and changing appearances.”
        </p>

        <div className="mt-9 flex flex-wrap items-center justify-center gap-4">
          <Link
            to="/technology"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white font-semibold shadow-lg shadow-sky-500/25 hover:opacity-95 transition"
          >
            Explore Technology <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            to="/architecture"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl glass-card text-slate-800 dark:text-slate-100 font-semibold hover:border-sky-400/50 transition shadow-sm"
          >
            View Architecture
          </Link>
          <a
            href="#live-demo"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl border border-violet-300 dark:border-violet-500/40 bg-violet-50 dark:bg-violet-500/10 text-violet-700 dark:text-violet-200 font-semibold hover:bg-violet-100 dark:hover:bg-violet-500/20 transition"
          >
            Explore Demo
          </a>
        </div>
      </section>

      {/* 2. PROBLEM STATEMENT */}
      <section className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="mb-10">
          <span className="text-xs font-mono uppercase tracking-widest text-sky-600 dark:text-sky-400">Problem Statement</span>
          <h2 className="mt-2 font-display text-3xl font-bold text-slate-900 dark:text-white">Why Traditional Trackers Break</h2>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {PROBLEMS.map((p) => {
            const Icon = p.icon;
            return (
              <div key={p.title} className="glass-card rounded-2xl p-5 transition">
                <Icon className="w-6 h-6 text-rose-500 dark:text-rose-400 mb-3" />
                <h3 className="font-display font-semibold text-slate-900 dark:text-white">{p.title}</h3>
                <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 leading-relaxed">{p.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* 3. OUR SOLUTION */}
      <section className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="glass-card rounded-3xl p-8 sm:p-10 grid lg:grid-cols-2 gap-8 items-center">
          <div>
            <span className="text-xs font-mono uppercase tracking-widest text-violet-600 dark:text-violet-400">Our Solution</span>
            <h2 className="mt-2 font-display text-3xl font-bold text-slate-900 dark:text-white">
              Unified Detection, Re-ID &amp; Conflict-Aware Association
            </h2>
            <p className="mt-4 text-slate-700 dark:text-slate-300 leading-relaxed">
              ARGUS-X decouples short-horizon per-camera tracking from long-horizon global identity memory.
              Every tracklet is encoded into a 512-dimensional OSNet appearance vector, indexed in FAISS,
              and evaluated by a Hungarian data associator that fuses appearance, temporal feasibility,
              camera transition topology, and calibrated motion before assigning or recovering a global ID.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-2xl bg-sky-50 dark:bg-[#09112b] border border-sky-200 dark:border-sky-500/20 p-4">
              <div className="text-2xl font-display font-bold text-sky-600 dark:text-sky-400">512-d</div>
              <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">OSNet L2-Normalized Embeddings</div>
            </div>
            <div className="rounded-2xl bg-violet-50 dark:bg-[#09112b] border border-violet-200 dark:border-violet-500/20 p-4">
              <div className="text-2xl font-display font-bold text-violet-600 dark:text-violet-400">5 States</div>
              <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">Active, Lost, Recovered, Uncertain, Expired</div>
            </div>
            <div className="rounded-2xl bg-sky-50 dark:bg-[#09112b] border border-sky-200 dark:border-sky-500/20 p-4">
              <div className="text-2xl font-display font-bold text-sky-600 dark:text-sky-400">FAISS</div>
              <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">IndexFlatIP Cosine Gallery</div>
            </div>
            <div className="rounded-2xl bg-violet-50 dark:bg-[#09112b] border border-violet-200 dark:border-violet-500/20 p-4">
              <div className="text-2xl font-display font-bold text-violet-600 dark:text-violet-400">Zero Auto-Merge</div>
              <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">Operator Confirmation Policy</div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. HOW IT WORKS */}
      <section className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="mb-10">
          <span className="text-xs font-mono uppercase tracking-widest text-sky-600 dark:text-sky-400">How It Works</span>
          <h2 className="mt-2 font-display text-3xl font-bold text-slate-900 dark:text-white">End-to-End Recovery Pipeline</h2>
        </div>
        <PipelineVisualizer />
      </section>

      {/* 5. TECHNOLOGY STACK */}
      <section className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="mb-10">
          <span className="text-xs font-mono uppercase tracking-widest text-violet-600 dark:text-violet-400">Technology Stack</span>
          <h2 className="mt-2 font-display text-3xl font-bold text-slate-900 dark:text-white">Core Engines &amp; Infrastructure</h2>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {TECH_STACK.map((t) => (
            <div key={t.name} className="glass-card rounded-2xl p-5">
              <Cpu className="w-5 h-5 text-sky-600 dark:text-sky-400 mb-3" />
              <div className="font-display font-semibold text-slate-900 dark:text-white">{t.name}</div>
              <div className="mt-1.5 text-xs text-slate-600 dark:text-slate-400">{t.role}</div>
            </div>
          ))}
        </div>
      </section>

      {/* 6. KEY FEATURES */}
      <section className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="mb-10">
          <span className="text-xs font-mono uppercase tracking-widest text-sky-600 dark:text-sky-400">Capabilities</span>
          <h2 className="mt-2 font-display text-3xl font-bold text-slate-900 dark:text-white">Key Features</h2>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {FEATURES.map((f) => {
            const Icon = f.icon;
            return (
              <div key={f.title} className="glass-card rounded-2xl p-5">
                <Icon className="w-5 h-5 text-violet-600 dark:text-violet-400 mb-3" />
                <h3 className="font-display font-semibold text-slate-900 dark:text-white">{f.title}</h3>
                <p className="mt-1.5 text-sm text-slate-600 dark:text-slate-400">{f.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* 7. SYSTEM ARCHITECTURE */}
      <section className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="mb-8">
          <span className="text-xs font-mono uppercase tracking-widest text-violet-600 dark:text-violet-400">System Architecture</span>
          <h2 className="mt-2 font-display text-3xl font-bold text-slate-900 dark:text-white">Data Flow &amp; Isolation</h2>
        </div>
        <ArchitectureDiagram />
      </section>

      {/* 8. PERFORMANCE & EVALUATION */}
      <section className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="glass-card rounded-3xl p-8">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <span className="text-xs font-mono uppercase tracking-widest text-sky-600 dark:text-sky-400">Performance &amp; Evaluation</span>
              <h2 className="mt-1 font-display text-3xl font-bold text-slate-900 dark:text-white">Benchmark &amp; Validation Status</h2>
            </div>
            <span className="px-3 py-1 rounded-full text-xs font-medium bg-amber-50 dark:bg-amber-500/15 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-500/35">
              Real-World Annotated Validation: Pending
            </span>
          </div>
          <p className="text-sm text-slate-700 dark:text-slate-300 mb-6">
            Automated unit and synthetic pipeline tests (85/85 passing) verify detection, tracking, association,
            and API behavior under synthetic conditions. Real-world cross-camera metrics below remain pending until
            annotated CCTV evaluation is completed via <code className="text-sky-600 dark:text-sky-300 font-mono">evaluation/real_video_validation.py</code>.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
            {[
              "Detection FPS",
              "Tracking FPS",
              "Recovery Accuracy",
              "Precision",
              "Recall",
              "F1 Score",
              "Processing Latency",
            ].map((metric) => (
              <div key={metric} className="rounded-xl bg-slate-50 dark:bg-[#080f26] border border-slate-200 dark:border-slate-800 p-4 text-center">
                <div className="text-xs text-slate-600 dark:text-slate-400 font-medium">{metric}</div>
                <div className="mt-2 font-mono text-sm font-semibold text-amber-700 dark:text-amber-300">Pending Real Eval</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 9. LIVE DEMO & 10. FUTURE SCOPE */}
      <section id="live-demo" className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto grid lg:grid-cols-2 gap-6">
        <div className="glass-card rounded-3xl p-8 flex flex-col justify-between">
          <div>
            <span className="text-xs font-mono uppercase tracking-widest text-sky-600 dark:text-sky-400">Live Platform</span>
            <h2 className="mt-2 font-display text-3xl font-bold text-slate-900 dark:text-white">Interactive Monitoring Console</h2>
            <p className="mt-3 text-slate-700 dark:text-slate-300 text-sm leading-relaxed">
              Sign in as an Administrator or Operator to access live camera streams, real-time WebSocket tracking
              telemetry, identity recovery history, conflict guards, and the uncertain identity review queue.
            </p>
          </div>
          <div className="mt-6 flex flex-wrap gap-3">
            {isExternalDash ? (
              <a
                href={dashboardUrl}
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white font-semibold shadow-md"
              >
                Open Monitoring Dashboard <ArrowRight className="w-4 h-4" />
              </a>
            ) : (
              <Link
                to={dashboardUrl}
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white font-semibold shadow-md"
              >
                Open Monitoring Dashboard <ArrowRight className="w-4 h-4" />
              </Link>
            )}
          </div>
        </div>

        <div className="glass-card rounded-3xl p-8">
          <span className="text-xs font-mono uppercase tracking-widest text-violet-600 dark:text-violet-400">Roadmap</span>
          <h2 className="mt-2 font-display text-3xl font-bold text-slate-900 dark:text-white">Future Scope</h2>
          <ul className="mt-4 space-y-2.5 text-sm text-slate-700 dark:text-slate-300">
            <li>• Scaling to larger multi-site camera networks</li>
            <li>• Improved long-term identity recovery under severe appearance drift</li>
            <li>• Edge deployment &amp; hardware-accelerated inference</li>
            <li>• Real-time latency optimization for OSNet Re-ID extraction</li>
            <li>• Comprehensive benchmarking on annotated real-world CCTV datasets</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
