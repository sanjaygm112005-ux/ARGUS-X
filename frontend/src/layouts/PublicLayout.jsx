import React from "react";
import { Outlet } from "react-router-dom";
import { PublicNavbar } from "../components/PublicNavbar";
import { PublicFooter } from "../components/PublicFooter";

export function PublicLayout() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-[#050816] text-slate-900 dark:text-slate-100 argus-grid-bg transition-colors">
      <PublicNavbar />
      <main className="flex-1">
        <Outlet />
      </main>
      <PublicFooter />
    </div>
  );
}
