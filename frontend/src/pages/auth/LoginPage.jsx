import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff, Loader2, Lock } from "lucide-react";
import { ArgusLogo } from "../../components/ArgusLogo";
import { useAuth } from "../../context/AuthContext";

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const user = await login(identifier, password, rememberMe);
      navigate(user.role === "ADMIN" ? "/admin" : "/operator", { replace: true });
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Login failed. Check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md glass-card rounded-3xl p-8 shadow-xl shadow-slate-200/60 dark:shadow-2xl dark:shadow-sky-950/40">
        <div className="text-center mb-8">
          <ArgusLogo size="lg" />
          <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">Sign in to access ARGUS-X Command Console</p>
        </div>

        {error && (
          <div className="mb-5 p-3.5 rounded-xl bg-rose-50 dark:bg-rose-500/15 border border-rose-300 dark:border-rose-500/40 text-rose-700 dark:text-rose-300 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">
              Email or Username
            </label>
            <input
              type="text"
              required
              autoComplete="username"
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              placeholder="admin or operator (e.g. sanjay)"
              className="w-full rounded-xl bg-white dark:bg-[#080f26] border border-slate-300 dark:border-slate-700 focus:border-sky-500 dark:focus:border-sky-400 focus:outline-none px-4 py-2.5 text-sm text-slate-900 dark:text-white shadow-sm"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">Password</label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                required
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                className="w-full rounded-xl bg-white dark:bg-[#080f26] border border-slate-300 dark:border-slate-700 focus:border-sky-500 dark:focus:border-sky-400 focus:outline-none px-4 py-2.5 pr-11 text-sm text-slate-900 dark:text-white shadow-sm"
              />
              <button
                type="button"
                aria-label={showPassword ? "Hide password" : "Show password"}
                onClick={() => setShowPassword((v) => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-white"
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs">
            <label className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                className="rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-[#080f26] text-sky-600"
              />
              Remember me
            </label>
            <Link to="/forgot-password" className="text-sky-600 dark:text-sky-400 hover:underline font-medium">
              Forgot password?
            </Link>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-gradient-to-r from-blue-700 to-indigo-700 text-white font-semibold text-sm hover:opacity-95 disabled:opacity-60 transition"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Lock className="w-4 h-4" />}
            {loading ? "Authenticating..." : "Login"}
          </button>
        </form>

        <p className="mt-6 text-center text-xs text-slate-400">
          Need an operator account?{" "}
          <Link to="/register" className="text-sky-400 hover:underline">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
