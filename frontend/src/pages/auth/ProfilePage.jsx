import React, { useEffect, useState } from "react";
import api from "../../services/api";
import { useAuth } from "../../context/AuthContext";

export function ProfilePage() {
  const { user, refreshUser } = useAuth();
  const [fullName, setFullName] = useState(user?.full_name || "");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [sessions, setSessions] = useState([]);
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const loadSessions = async () => {
    try {
      const { data } = await api.get("/api/auth/sessions");
      setSessions(data.sessions || []);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    loadSessions();
  }, []);

  const handleSave = async (e) => {
    e.preventDefault();
    setStatusMsg("");
    setErrorMsg("");
    try {
      const payload = { full_name: fullName };
      if (newPassword) {
        payload.current_password = currentPassword;
        payload.new_password = newPassword;
      }
      await api.put("/api/auth/profile", payload);
      await refreshUser();
      setCurrentPassword("");
      setNewPassword("");
      setStatusMsg("Profile updated successfully.");
    } catch (err) {
      setErrorMsg(err?.response?.data?.detail || "Update failed.");
    }
  };

  const revokeSession = async (id) => {
    await api.delete(`/api/auth/sessions/${id}`);
    loadSessions();
  };

  return (
    <div className="max-w-4xl space-y-6">
      <div className="glass-card rounded-2xl p-6">
        <h2 className="font-display text-xl font-bold text-white mb-4">User Profile</h2>
        {statusMsg && <div className="mb-3 p-3 rounded-lg bg-emerald-500/15 text-emerald-300 text-sm">{statusMsg}</div>}
        {errorMsg && <div className="mb-3 p-3 rounded-lg bg-rose-500/15 text-rose-300 text-sm">{errorMsg}</div>}

        <form onSubmit={handleSave} className="grid sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Username</label>
            <input disabled value={user?.username || ""} className="w-full rounded-lg bg-slate-900 border border-slate-800 px-3 py-2 text-sm text-slate-400" />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Role</label>
            <input disabled value={user?.role || ""} className="w-full rounded-lg bg-slate-900 border border-slate-800 px-3 py-2 text-sm text-sky-300" />
          </div>
          <div className="sm:col-span-2">
            <label className="block text-xs text-slate-400 mb-1">Full Name</label>
            <input
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="w-full rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Current Password (for password change)</label>
            <input
              type="password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              className="w-full rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">New Password</label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="w-full rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
            />
          </div>
          <div className="sm:col-span-2">
            <button type="submit" className="px-5 py-2 rounded-lg bg-sky-500 text-white text-sm font-semibold">
              Save Changes
            </button>
          </div>
        </form>
      </div>

      <div className="glass-card rounded-2xl p-6">
        <h3 className="font-display text-lg font-semibold text-white mb-4">Session Management</h3>
        <div className="space-y-2">
          {sessions.map((s) => (
            <div key={s.id} className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-[#080f26] border border-slate-800 text-xs">
              <div>
                <div className="text-slate-200 font-medium">
                  IP: {s.ip_address || "unknown"} {s.is_current && <span className="ml-2 text-emerald-400">(Current)</span>}
                </div>
                <div className="text-slate-400 truncate max-w-md">{s.user_agent || "Browser session"}</div>
              </div>
              <div className="flex items-center gap-3">
                <span className={s.is_revoked ? "text-rose-400" : "text-emerald-400"}>
                  {s.is_revoked ? "REVOKED" : "ACTIVE"}
                </span>
                {!s.is_revoked && (
                  <button
                    type="button"
                    onClick={() => revokeSession(s.id)}
                    className="px-2.5 py-1 rounded bg-rose-500/20 text-rose-300 hover:bg-rose-500/30"
                  >
                    Revoke
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
