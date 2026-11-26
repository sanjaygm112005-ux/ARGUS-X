import React, { useState } from "react";
import { Link } from "react-router-dom";
import { ArchitectureDiagram } from "../../components/ArchitectureDiagram";
import { PipelineVisualizer } from "../../components/PipelineVisualizer";

export function AboutPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-16 space-y-6">
      <h1 className="font-display text-4xl font-bold text-slate-900 dark:text-white">About ARGUS-X</h1>
      <div className="glass-card rounded-2xl p-6 space-y-4 text-slate-700 dark:text-slate-300 leading-relaxed shadow-sm">
        <p>
          ARGUS-X is an AI-powered multi-camera identity recovery platform engineered to solve long-horizon
          person tracking across non-overlapping camera feeds, occlusions, and appearance drift.
        </p>
        <p>
          By combining YOLO11n detection, BoT-SORT short-horizon tracking, OSNet x0.25 appearance embeddings,
          FAISS vector similarity memory, and Hungarian data association with strict conflict resolution,
          ARGUS-X maintains global identity continuity while keeping human operators in control of uncertain matches.
        </p>
      </div>
    </div>
  );
}

export function TechnologyPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-16 space-y-10">
      <div>
        <h1 className="font-display text-4xl font-bold text-slate-900 dark:text-white">Technology Pipeline</h1>
        <p className="mt-2 text-slate-600 dark:text-slate-400">
          Inspect each stage from raw camera frames to operator-verified global identity recovery.
        </p>
      </div>
      <PipelineVisualizer />
    </div>
  );
}

export function ArchitecturePage() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-16 space-y-8">
      <div>
        <h1 className="font-display text-4xl font-bold text-slate-900 dark:text-white">System Architecture</h1>
        <p className="mt-2 text-slate-600 dark:text-slate-400">
          Shared single-instance AI engine (YOLO11n, BoT-SORT, OSNet, FAISS) separated cleanly from the Neon/PostgreSQL application database.
        </p>
      </div>
      <ArchitectureDiagram />
    </div>
  );
}

export function FeaturesPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-16 space-y-8">
      <h1 className="font-display text-4xl font-bold text-slate-900 dark:text-white">Platform Features</h1>
      <div className="grid sm:grid-cols-2 gap-5">
        {[
          ["Multi-Camera Tracking", "Independent per-camera BoT-SORT local trackers feeding shared global identity memory."],
          ["Long-Term Identity Memory", "FAISS IndexFlatIP gallery paired with SQLite observation persistence."],
          ["Conflict Prevention", "Prevents simultaneous incompatible assignments across cameras and forbids unsafe auto-merges."],
          ["Role-Based Security", "Argon2id passwords, HttpOnly JWT cookies, CSRF protection, and Admin/Operator RBAC."],
        ].map(([title, desc]) => (
          <div key={title} className="glass-card rounded-2xl p-6 shadow-sm">
            <h2 className="font-display text-xl font-semibold text-slate-900 dark:text-white">{title}</h2>
            <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">{desc}</p>
          </div>
        ))}
      </div>
      <Link to="/login" className="inline-block px-5 py-3 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white font-semibold shadow-md">
        Sign In to Platform
      </Link>
    </div>
  );
}

export function ContactPage() {
  const [sent, setSent] = useState(false);
  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-16 space-y-6">
      <h1 className="font-display text-4xl font-bold text-slate-900 dark:text-white">Contact &amp; Access</h1>
      <div className="glass-card rounded-2xl p-6 space-y-4 shadow-sm">
        <p className="text-sm text-slate-600 dark:text-slate-300">
          Need operator credentials or deployment assistance for ARGUS-X? Submit an inquiry below or sign in with your administrator-issued account.
        </p>
        {sent ? (
          <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-500/15 border border-emerald-300 dark:border-emerald-500/30 text-emerald-800 dark:text-emerald-300 text-sm">
            Inquiry noted locally. For immediate access, please contact your system administrator or sign in on the Login page.
          </div>
        ) : (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              setSent(true);
            }}
            className="space-y-4"
          >
            <input
              required
              placeholder="Your Name"
              className="w-full rounded-xl bg-white dark:bg-[#080f26] border border-slate-300 dark:border-slate-700 px-4 py-2.5 text-sm text-slate-900 dark:text-white shadow-sm"
            />
            <input
              required
              type="email"
              placeholder="Work Email"
              className="w-full rounded-xl bg-white dark:bg-[#080f26] border border-slate-300 dark:border-slate-700 px-4 py-2.5 text-sm text-slate-900 dark:text-white shadow-sm"
            />
            <textarea
              required
              rows={4}
              placeholder="Message"
              className="w-full rounded-xl bg-white dark:bg-[#080f26] border border-slate-300 dark:border-slate-700 px-4 py-2.5 text-sm text-slate-900 dark:text-white shadow-sm"
            />
            <button
              type="submit"
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white text-sm font-semibold shadow-md"
            >
              Send Message
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
