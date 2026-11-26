import React, { useEffect, useState } from "react";
import api from "../../services/api";

export function AuditLogsPage() {
  const [logs, setLogs] = useState([]);
  const [actionFilter, setActionFilter] = useState("");

  useEffect(() => {
    const params = actionFilter ? { action: actionFilter } : {};
    api.get("/api/audit-logs", { params }).then(({ data }) => {
      setLogs(data.audit_logs || []);
    });
  }, [actionFilter]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="font-display text-xl font-bold text-white">Security &amp; Administrative Audit Logs</h2>
        <input
          placeholder="Filter by action (e.g. LOGIN_SUCCESS)"
          value={actionFilter}
          onChange={(e) => setActionFilter(e.target.value)}
          className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-xs text-white"
        />
      </div>

      <div className="glass-card rounded-2xl p-5 overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-slate-400 border-b border-slate-800">
            <tr>
              <th className="py-2 pr-4">Time</th>
              <th className="py-2 pr-4">Actor</th>
              <th className="py-2 pr-4">Action</th>
              <th className="py-2 pr-4">Resource</th>
              <th className="py-2 pr-4">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {logs.map((l) => (
              <tr key={l.id}>
                <td className="py-2.5 pr-4 font-mono text-slate-400">{l.created_at}</td>
                <td className="py-2.5 pr-4 text-white">{l.actor_username}</td>
                <td className="py-2.5 pr-4 font-mono text-sky-300">{l.action}</td>
                <td className="py-2.5 pr-4">{l.resource_type} #{l.resource_id}</td>
                <td className="py-2.5 pr-4 text-slate-300">{l.details}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function AnalyticsPage() {
  const [overview, setOverview] = useState(null);
  const [perf, setPerf] = useState(null);

  useEffect(() => {
    Promise.all([api.get("/api/analytics/overview"), api.get("/api/analytics/performance")]).then(
      ([o, p]) => {
        setOverview(o.data);
        setPerf(p.data);
      }
    );
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="font-display text-xl font-bold text-white">Platform Analytics</h2>
      <div className="grid sm:grid-cols-3 gap-4">
        <div className="glass-card rounded-2xl p-5">
          <div className="text-xs text-slate-400">Tracking Events</div>
          <div className="mt-1 font-display text-2xl font-bold text-white">
            {overview?.total_tracking_events ?? 0}
          </div>
        </div>
        <div className="glass-card rounded-2xl p-5">
          <div className="text-xs text-slate-400">Recovery Events</div>
          <div className="mt-1 font-display text-2xl font-bold text-white">
            {overview?.total_recovery_events ?? 0}
          </div>
        </div>
        <div className="glass-card rounded-2xl p-5">
          <div className="text-xs text-slate-400">Real-World Validation Status</div>
          <div className="mt-1 font-mono text-sm font-semibold text-amber-300">
            {perf?.real_world_status || "PENDING_ANNOTATED_EVALUATION"}
          </div>
        </div>
      </div>
    </div>
  );
}

export function SettingsPage() {
  const [settings, setSettings] = useState([]);
  const [drafts, setDrafts] = useState({});
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  const load = async () => {
    const { data } = await api.get("/api/settings");
    setSettings(data.settings || []);
    const map = {};
    for (const s of data.settings || []) map[s.key] = s.value;
    setDrafts(map);
  };

  useEffect(() => {
    load();
  }, []);

  const saveSetting = async (key) => {
    setMsg("");
    setErr("");
    try {
      await api.put(`/api/settings/${key}`, { value: drafts[key] });
      setMsg(`Updated ${key}.`);
      load();
    } catch (e) {
      setErr(e?.response?.data?.detail || "Update failed.");
    }
  };

  return (
    <div className="space-y-4 max-w-4xl">
      <h2 className="font-display text-xl font-bold text-white">System Settings</h2>
      {msg && <div className="p-3 rounded-lg bg-emerald-500/15 text-emerald-300 text-xs">{msg}</div>}
      {err && <div className="p-3 rounded-lg bg-rose-500/15 text-rose-300 text-xs">{err}</div>}

      <div className="space-y-3">
        {settings.map((s) => (
          <div key={s.key} className="glass-card rounded-2xl p-5 flex flex-wrap items-center justify-between gap-4">
            <div className="max-w-md">
              <div className="font-mono text-sm text-sky-300">{s.key}</div>
              <div className="text-xs text-slate-400 mt-0.5">{s.description}</div>
            </div>
            <div className="flex items-center gap-2">
              <input
                value={drafts[s.key] ?? ""}
                onChange={(e) => setDrafts({ ...drafts, [s.key]: e.target.value })}
                className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-1.5 text-sm text-white"
              />
              <button
                type="button"
                onClick={() => saveSetting(s.key)}
                className="px-4 py-1.5 rounded-lg bg-sky-500 text-white text-xs font-semibold"
              >
                Save
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
