import React, { useState } from "react";
import { Link } from "react-router-dom";
import api from "../../services/api";
import { ArgusLogo } from "../../components/ArgusLogo";

export function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    try {
      const { data } = await api.post("/api/auth/forgot-password", { email });
      setResult(data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Request failed.");
    }
  };

  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md glass-card rounded-3xl p-8 space-y-5">
        <div className="text-center">
          <ArgusLogo />
          <h1 className="mt-4 font-display text-2xl font-bold text-white">Forgot Password</h1>
        </div>
        {error && <div className="p-3 rounded-xl bg-rose-500/15 text-rose-300 text-sm">{error}</div>}
        {result && (
          <div className="p-3.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs space-y-2">
            <p>{result.message}</p>
            {result.dev_reset_token && (
              <p className="font-mono break-all text-sky-300">
                Dev Reset Token: {result.dev_reset_token}
              </p>
            )}
            <Link to="/reset-password" className="inline-block text-sky-400 underline">
              Proceed to Reset Password →
            </Link>
          </div>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="email"
            required
            placeholder="Registered Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-xl bg-[#080f26] border border-slate-700 px-4 py-2.5 text-sm text-white"
          />
          <button
            type="submit"
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white text-sm font-semibold"
          >
            Request Reset Token
          </button>
        </form>
        <div className="text-center text-xs">
          <Link to="/login" className="text-sky-400 hover:underline">Back to Login</Link>
        </div>
      </div>
    </div>
  );
}

export function ResetPasswordPage() {
  const [token, setToken] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErr("");
    try {
      const { data } = await api.post("/api/auth/reset-password", {
        token,
        new_password: newPassword,
      });
      setMsg(data.message);
    } catch (error) {
      setErr(error?.response?.data?.detail || "Reset failed.");
    }
  };

  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md glass-card rounded-3xl p-8 space-y-5">
        <div className="text-center">
          <ArgusLogo />
          <h1 className="mt-4 font-display text-2xl font-bold text-white">Reset Password</h1>
        </div>
        {err && <div className="p-3 rounded-xl bg-rose-500/15 text-rose-300 text-sm">{err}</div>}
        {msg && <div className="p-3 rounded-xl bg-emerald-500/15 text-emerald-300 text-sm">{msg}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            required
            placeholder="Reset Token"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            className="w-full rounded-xl bg-[#080f26] border border-slate-700 px-4 py-2.5 text-sm text-white"
          />
          <input
            type="password"
            required
            minLength={8}
            placeholder="New Password (min 8 chars)"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            className="w-full rounded-xl bg-[#080f26] border border-slate-700 px-4 py-2.5 text-sm text-white"
          />
          <button
            type="submit"
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white text-sm font-semibold"
          >
            Update Password
          </button>
        </form>
        <div className="text-center text-xs">
          <Link to="/login" className="text-sky-400 hover:underline">Back to Login</Link>
        </div>
      </div>
    </div>
  );
}
