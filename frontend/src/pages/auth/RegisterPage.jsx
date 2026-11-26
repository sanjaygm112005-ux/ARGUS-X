import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Loader2, UserPlus } from "lucide-react";
import { ArgusLogo } from "../../components/ArgusLogo";
import { useAuth } from "../../context/AuthContext";

export function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ email: "", username: "", full_name: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await register(form);
      setSuccess("Registration successful! Redirecting to login...");
      setTimeout(() => navigate("/login"), 1200);
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Registration failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md glass-card rounded-3xl p-8">
        <div className="text-center mb-6">
          <ArgusLogo size="lg" />
          <p className="mt-2 text-sm text-slate-400">Create an ARGUS-X Operator Account</p>
        </div>

        {error && (
          <div className="mb-4 p-3 rounded-xl bg-rose-500/15 border border-rose-500/40 text-rose-300 text-sm">
            {error}
          </div>
        )}
        {success && (
          <div className="mb-4 p-3 rounded-xl bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 text-sm">
            {success}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="email"
            required
            placeholder="Email address"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
            className="w-full rounded-xl bg-[#080f26] border border-slate-700 px-4 py-2.5 text-sm text-white"
          />
          <input
            type="text"
            required
            placeholder="Username"
            value={form.username}
            onChange={(e) => setForm({ ...form, username: e.target.value })}
            className="w-full rounded-xl bg-[#080f26] border border-slate-700 px-4 py-2.5 text-sm text-white"
          />
          <input
            type="text"
            placeholder="Full Name"
            value={form.full_name}
            onChange={(e) => setForm({ ...form, full_name: e.target.value })}
            className="w-full rounded-xl bg-[#080f26] border border-slate-700 px-4 py-2.5 text-sm text-white"
          />
          <input
            type="password"
            required
            minLength={8}
            placeholder="Password (min 8 characters)"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            className="w-full rounded-xl bg-[#080f26] border border-slate-700 px-4 py-2.5 text-sm text-white"
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white font-semibold text-sm"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <UserPlus className="w-4 h-4" />}
            Register
          </button>
        </form>

        <p className="mt-5 text-center text-xs text-slate-400">
          Already registered?{" "}
          <Link to="/login" className="text-sky-400 hover:underline">
            Back to Login
          </Link>
        </p>
      </div>
    </div>
  );
}
