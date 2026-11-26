import React, { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Camera,
  Database,
  FileText,
  Film,
  Gauge,
  GitCompare,
  History,
  LayoutDashboard,
  LogOut,
  Menu,
  Settings,
  ShieldAlert,
  UserCircle,
  Users,
  X,
} from "lucide-react";
import { ArgusLogo } from "../components/ArgusLogo";
import { ThemeToggle } from "../components/ThemeToggle";
import { useAuth } from "../context/AuthContext";
import { useMonitoringSocket } from "../hooks/useMonitoringSocket";

const ADMIN_NAV = [
  { to: "/admin", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/admin/mot17", label: "MOT17 Sequences", icon: Database },
  { to: "/admin/benchmarks", label: "Dataset Benchmarking", icon: Gauge },
  { to: "/admin/single-video", label: "Single-Video Analysis", icon: Film },
  { to: "/admin/multicamera", label: "Multi-Camera Analysis", icon: GitCompare },
  { to: "/admin/cameras", label: "Camera Management", icon: Camera },
  { to: "/admin/monitoring", label: "Live Monitoring", icon: Activity },
  { to: "/admin/identities", label: "Global Identities", icon: Users },
  { to: "/admin/recovery-history", label: "Recovery History", icon: History },
  { to: "/admin/uncertain", label: "Uncertain Review", icon: AlertTriangle },
  { to: "/admin/conflicts", label: "Conflict Management", icon: ShieldAlert },
  { to: "/admin/users", label: "User Management", icon: UserCircle },
  { to: "/admin/audit-logs", label: "Audit Logs", icon: FileText },
  { to: "/admin/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/admin/settings", label: "Settings", icon: Settings },
  { to: "/profile", label: "My Profile", icon: UserCircle },
];

const OPERATOR_NAV = [
  { to: "/operator", label: "Live Camera Feeds", icon: Camera, end: true },
  { to: "/operator/mot17", label: "MOT17 Sequences", icon: Database },
  { to: "/operator/benchmarks", label: "Dataset Benchmarking", icon: Gauge },
  { to: "/operator/single-video", label: "Single-Video Analysis", icon: Film },
  { to: "/operator/multicamera", label: "Multi-Camera Analysis", icon: GitCompare },
  { to: "/operator/tracking", label: "Active Tracking", icon: Activity },
  { to: "/operator/identities", label: "Global Identities", icon: Users },
  { to: "/operator/recovery-history", label: "Recovery History", icon: History },
  { to: "/operator/uncertain", label: "Uncertain Alerts", icon: AlertTriangle },
  { to: "/operator/review", label: "Review Interface", icon: ShieldAlert },
  { to: "/profile", label: "My Profile", icon: UserCircle },
];

export function DashboardLayout({ mode = "operator" }) {
  const { user, logout } = useAuth();
  const { snapshot, connectionState } = useMonitoringSocket(true);
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const navItems = mode === "admin" ? ADMIN_NAV : OPERATOR_NAV;

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  const connColor =
    connectionState === "connected"
      ? "bg-emerald-50 dark:bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-500/40"
      : connectionState === "polling"
      ? "bg-amber-50 dark:bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-300 dark:border-amber-500/40"
      : "bg-rose-50 dark:bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-300 dark:border-rose-500/40";

  return (
    <div className="min-h-screen bg-[#faf7f2] dark:bg-[#050816] text-[#0f1e3d] dark:text-slate-100 flex transition-colors">
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 bg-[#fffdf9] dark:bg-[#070c20] border-r border-[#e6dfd1] dark:border-slate-800/80 flex flex-col transition-transform lg:static lg:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="h-16 px-4 flex items-center justify-between border-b border-[#e6dfd1] dark:border-slate-800/80">
          <NavLink to="/">
            <ArgusLogo />
          </NavLink>
          <button
            type="button"
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden p-1.5 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                onClick={() => setSidebarOpen(false)}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition ${
                    isActive
                      ? "bg-blue-700 text-white dark:bg-gradient-to-r dark:from-sky-500/20 dark:to-violet-600/20 dark:text-sky-300 border border-blue-800 dark:border-sky-400/30 font-semibold shadow-sm"
                      : "text-slate-700 dark:text-slate-400 hover:text-blue-800 dark:hover:text-slate-100 hover:bg-[#f3ede2] dark:hover:bg-slate-800/50"
                  }`
                }
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-200 dark:border-slate-800/80 space-y-3">
          <div className="text-xs text-slate-500 dark:text-slate-400">
            Signed in as <span className="text-slate-900 dark:text-slate-200 font-semibold">{user?.username}</span>
            <span className="ml-2 px-1.5 py-0.5 rounded bg-violet-100 dark:bg-violet-500/20 text-violet-700 dark:text-violet-300 text-[10px] font-medium border border-violet-200 dark:border-transparent">
              {user?.role}
            </span>
          </div>
          <button
            type="button"
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-slate-100 hover:bg-rose-50 hover:text-rose-600 dark:bg-slate-800/80 dark:hover:bg-rose-500/20 dark:hover:text-rose-300 text-slate-700 dark:text-slate-300 text-sm transition font-medium border border-slate-200 dark:border-transparent shadow-sm"
          >
            <LogOut className="w-4 h-4" />
            Logout
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-16 border-b border-[#e6dfd1] dark:border-slate-800/80 bg-[#fffdf9]/90 dark:bg-[#070c20]/80 backdrop-blur px-4 sm:px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden p-2 rounded-lg text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              <Menu className="w-5 h-5" />
            </button>
            <h1 className="font-display font-semibold text-base sm:text-lg text-slate-900 dark:text-white">
              {mode === "admin" ? "ARGUS-X Admin Command" : "ARGUS-X Operator Console"}
            </h1>
          </div>

          <div className="flex items-center gap-3">
            <ThemeToggle />
            <span className={`px-2.5 py-1 rounded-full text-xs font-medium border ${connColor}`}>
              WS: {connectionState}
            </span>
          </div>
        </header>

        <main className="flex-1 p-4 sm:p-6 overflow-y-auto">
          <Outlet context={{ snapshot, connectionState }} />
        </main>
      </div>
    </div>
  );
}
