import React, { useState } from "react";
import { Link, NavLink } from "react-router-dom";
import { Menu, X, LogIn, LayoutDashboard } from "lucide-react";
import { ArgusLogo } from "./ArgusLogo";
import { ThemeToggle } from "./ThemeToggle";
import { useAuth } from "../context/AuthContext";

const NAV_LINKS = [
  { to: "/", label: "Home" },
  { to: "/about", label: "About" },
  { to: "/technology", label: "Technology" },
  { to: "/architecture", label: "Architecture" },
  { to: "/features", label: "Features" },
  { to: "/contact", label: "Contact" },
];

export function PublicNavbar() {
  const [open, setOpen] = useState(false);
  const { user } = useAuth();
  const dashPath = user?.role === "ADMIN" ? "/admin" : "/operator";

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200 dark:border-slate-800/70 bg-white/90 dark:bg-[#050816]/85 backdrop-blur-xl transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <Link to="/" onClick={() => setOpen(false)}>
          <ArgusLogo />
        </Link>

        <nav className="hidden md:flex items-center gap-6">
          {NAV_LINKS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `text-sm font-medium transition-colors ${
                  isActive
                    ? "text-sky-600 dark:text-sky-400 font-semibold"
                    : "text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="hidden md:flex items-center gap-3">
          <ThemeToggle />
          {user ? (
            <Link
              to={dashPath}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white text-sm font-semibold shadow-md shadow-sky-500/20 hover:opacity-95 transition"
            >
              <LayoutDashboard className="w-4 h-4" />
              Dashboard
            </Link>
          ) : (
            <Link
              to="/login"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-sky-50 hover:bg-sky-100 text-sky-700 border border-sky-300 dark:bg-sky-500/15 dark:border-sky-400/35 dark:text-sky-300 text-sm font-semibold dark:hover:bg-sky-500/25 transition shadow-sm"
            >
              <LogIn className="w-4 h-4" />
              Login
            </Link>
          )}
        </div>

        <div className="flex md:hidden items-center gap-2">
          <ThemeToggle />
          <button
            type="button"
            aria-label="Toggle Menu"
            onClick={() => setOpen((v) => !v)}
            className="p-2 rounded-lg text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/60"
          >
            {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {open && (
        <div className="md:hidden border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-[#070b1d] px-4 py-4 space-y-2">
          {NAV_LINKS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `block px-3 py-2 rounded-lg text-sm font-medium ${
                  isActive
                    ? "bg-sky-50 dark:bg-sky-500/15 text-sky-600 dark:text-sky-400 font-semibold"
                    : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/50"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
          <Link
            to={user ? dashPath : "/login"}
            onClick={() => setOpen(false)}
            className="block text-center mt-2 px-4 py-2.5 rounded-lg bg-gradient-to-r from-sky-500 to-violet-600 text-white text-sm font-semibold shadow-md"
          >
            {user ? "Open Dashboard" : "Login"}
          </Link>
        </div>
      )}
    </header>
  );
}
