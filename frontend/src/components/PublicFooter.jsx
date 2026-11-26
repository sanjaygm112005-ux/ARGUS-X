import React from "react";
import { Link } from "react-router-dom";
import { ArgusLogo } from "./ArgusLogo";

export function PublicFooter() {
  return (
    <footer className="border-t border-slate-200 dark:border-slate-800/70 bg-white/80 dark:bg-[#040612] py-12 mt-20 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid gap-8 md:grid-cols-3 items-start">
        <div>
          <ArgusLogo />
          <p className="mt-3 text-sm text-slate-600 dark:text-slate-400 max-w-xs">
            AI-Powered Multi-Camera Identity Recovery System. Built for long-horizon tracking under occlusion, camera transitions, and appearance drift.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm text-slate-600 dark:text-slate-400">
          <Link to="/" className="hover:text-sky-600 dark:hover:text-sky-400">Home</Link>
          <Link to="/about" className="hover:text-sky-600 dark:hover:text-sky-400">About</Link>
          <Link to="/technology" className="hover:text-sky-600 dark:hover:text-sky-400">Technology</Link>
          <Link to="/architecture" className="hover:text-sky-600 dark:hover:text-sky-400">Architecture</Link>
          <Link to="/features" className="hover:text-sky-600 dark:hover:text-sky-400">Features</Link>
          <Link to="/contact" className="hover:text-sky-600 dark:hover:text-sky-400">Contact</Link>
        </div>
        <div className="text-sm text-slate-600 dark:text-slate-400 md:text-right">
          <p className="text-slate-900 dark:text-slate-200 font-semibold">Platform Access</p>
          <p className="mt-1">Role-based Admin &amp; Operator Monitoring</p>
          <div className="mt-3 flex md:justify-end gap-3 font-medium">
            <Link to="/login" className="text-sky-600 dark:text-sky-400 hover:underline">Sign In</Link>
            <Link to="/register" className="text-violet-600 dark:text-violet-400 hover:underline">Register</Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
