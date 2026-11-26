import React, { useEffect, useState } from "react";
import {
  Users,
  Search,
  RefreshCw,
  Clock,
  Camera,
  CheckCircle2,
  AlertTriangle,
  History,
  X,
  ArrowRight,
  ShieldAlert,
  ChevronRight,
  Eye,
} from "lucide-react";
import api from "../../services/api";

function formatTimestamp(ts) {
  if (!ts) return "—";
  if (typeof ts === "number") {
    return new Date(ts * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }
  try {
    return new Date(ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return String(ts);
  }
}

function StateBadge({ state }) {
  switch (state) {
    case "ACTIVE":
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 dark:bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          ACTIVE
        </span>
      );
    case "RECOVERED":
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-sky-50 dark:bg-sky-500/15 text-sky-700 dark:text-sky-300 border border-sky-300 dark:border-sky-500/30">
          <CheckCircle2 className="w-3 h-3 text-sky-500" />
          RECOVERED
        </span>
      );
    case "TEMPORARILY_LOST":
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-50 dark:bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-500/30">
          <Clock className="w-3 h-3 text-amber-500" />
          TEMPORARILY_LOST
        </span>
      );
    case "UNCERTAIN":
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-50 dark:bg-rose-500/15 text-rose-700 dark:text-rose-300 border border-rose-300 dark:border-rose-500/30">
          <AlertTriangle className="w-3 h-3 text-rose-500" />
          UNCERTAIN
        </span>
      );
    case "EXPIRED":
    default:
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700">
          {state || "UNKNOWN"}
        </span>
      );
  }
}

export function IdentitiesPage({ conflictsOnly = false }) {
  const [identities, setIdentities] = useState([]);
  const [stateFilter, setStateFilter] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [inspectLoading, setInspectLoading] = useState(false);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const params = stateFilter ? { state: stateFilter } : {};
      const { data } = await api.get("/api/identities", { params });
      let list = data.identities || [];
      if (conflictsOnly) {
        list = list.filter((i) => i.has_conflict || i.lifecycle_state === "UNCERTAIN");
      }
      setIdentities(list);
    } catch (e) {
      setError(e?.response?.data?.detail || "Failed to load global identities.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [stateFilter, conflictsOnly]);

  const inspectIdentity = async (gid) => {
    setInspectLoading(true);
    try {
      const { data } = await api.get(`/api/identities/${gid}`);
      setSelected(data);
    } catch (e) {
      setError(e?.response?.data?.detail || "Failed to load identity details.");
    } finally {
      setInspectLoading(false);
    }
  };

  const filtered = identities.filter((id) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase().trim();
    return (
      id.global_id?.toLowerCase().includes(q) ||
      id.current_camera_code?.toLowerCase().includes(q) ||
      id.lifecycle_state?.toLowerCase().includes(q)
    );
  });

  return (
    <div className="space-y-6">
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="font-display text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Users className="w-5 h-5 text-sky-500" />
            {conflictsOnly ? "Identity Conflict Management" : "Global Identities"}
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Persistent re-identification memory across all cameras • {identities.length} total tracked
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Search box */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-400" />
            <input
              type="text"
              placeholder="Search Global ID or Cam..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 pr-3 py-1.5 rounded-xl bg-white dark:bg-[#080f26] border border-slate-300 dark:border-slate-700 text-xs text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:border-sky-500 shadow-sm min-w-[200px]"
            />
          </div>

          {!conflictsOnly && (
            <select
              value={stateFilter}
              onChange={(e) => setStateFilter(e.target.value)}
              className="rounded-xl bg-white dark:bg-[#080f26] border border-slate-300 dark:border-slate-700 px-3 py-1.5 text-xs text-slate-900 dark:text-white shadow-sm focus:outline-none"
            >
              <option value="">All Lifecycle States</option>
              <option value="ACTIVE">ACTIVE</option>
              <option value="TEMPORARILY_LOST">TEMPORARILY_LOST</option>
              <option value="RECOVERED">RECOVERED</option>
              <option value="UNCERTAIN">UNCERTAIN</option>
              <option value="EXPIRED">EXPIRED</option>
            </select>
          )}

          <button
            type="button"
            onClick={load}
            disabled={loading}
            title="Refresh Identities"
            className="p-2 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700 shadow-sm transition disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3.5 rounded-xl bg-rose-50 dark:bg-rose-500/15 border border-rose-300 dark:border-rose-500/30 text-rose-800 dark:text-rose-300 text-xs font-medium">
          {error}
        </div>
      )}

      {/* Main Table */}
      <div className="glass-card rounded-2xl p-5 overflow-x-auto shadow-sm">
        {loading && identities.length === 0 ? (
          <div className="py-8 text-center text-xs text-slate-500 dark:text-slate-400">
            Loading global identities from AI memory...
          </div>
        ) : filtered.length === 0 ? (
          <div className="py-8 text-center space-y-2">
            <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
              {conflictsOnly
                ? "No identity conflicts detected."
                : searchQuery
                ? `No identities matched '${searchQuery}'.`
                : "No global identities recorded yet."}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {conflictsOnly
                ? "All multi-camera re-identifications are consistent."
                : "When cameras detect persons, stable Global IDs will be registered here."}
            </p>
          </div>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="py-2.5 pr-4">Global ID</th>
                <th className="py-2.5 pr-4">Lifecycle State</th>
                <th className="py-2.5 pr-4">Current Camera</th>
                <th className="py-2.5 pr-4">Camera Journey</th>
                <th className="py-2.5 pr-4 text-center">Observations</th>
                <th className="py-2.5 pr-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60">
              {filtered.map((id) => (
                <tr
                  key={id.global_id}
                  className={`hover:bg-slate-50/60 dark:hover:bg-slate-800/40 transition ${
                    selected?.identity?.global_id === id.global_id
                      ? "bg-sky-50/70 dark:bg-sky-950/40"
                      : ""
                  }`}
                >
                  <td className="py-3 pr-4 font-mono text-sky-600 dark:text-sky-300 font-bold text-xs">
                    {id.global_id}
                  </td>
                  <td className="py-3 pr-4">
                    <StateBadge state={id.lifecycle_state} />
                  </td>
                  <td className="py-3 pr-4 font-mono text-xs text-slate-700 dark:text-slate-300">
                    {id.current_camera_code || "—"}
                  </td>
                  <td className="py-3 pr-4">
                    <div className="flex items-center gap-1 text-[11px] font-mono text-slate-600 dark:text-slate-400">
                      {(id.camera_sequence || []).slice(-3).map((c, idx) => (
                        <React.Fragment key={idx}>
                          {idx > 0 && <ChevronRight className="w-3 h-3 text-slate-400" />}
                          <span className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                            {c}
                          </span>
                        </React.Fragment>
                      ))}
                      {(id.camera_sequence || []).length === 0 && <span>—</span>}
                    </div>
                  </td>
                  <td className="py-3 pr-4 text-center font-mono text-xs font-semibold text-slate-700 dark:text-slate-300">
                    {id.observation_count}
                  </td>
                  <td className="py-3 pr-4 text-right">
                    <button
                      type="button"
                      onClick={() => inspectIdentity(id.global_id)}
                      disabled={inspectLoading}
                      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-sky-50 hover:bg-sky-100 dark:bg-sky-500/15 dark:hover:bg-sky-500/25 text-sky-700 dark:text-sky-300 border border-sky-300 dark:border-sky-500/30 text-xs font-semibold transition"
                    >
                      <Eye className="w-3 h-3" />
                      View History
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Selected Identity Inspector Drawer */}
      {selected && (
        <div className="glass-card rounded-2xl p-6 space-y-5 shadow-md border-2 border-sky-300 dark:border-sky-500/40 animate-in fade-in duration-200">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200 dark:border-slate-800">
            <div className="flex items-center gap-3">
              <span className="p-2 rounded-xl bg-sky-100 dark:bg-sky-950 text-sky-600 dark:text-sky-300 font-bold">
                <Users className="w-5 h-5" />
              </span>
              <div>
                <h3 className="font-display text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <span>{selected.identity.global_id}</span>
                  <StateBadge state={selected.identity.lifecycle_state} />
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  First Seen: {formatTimestamp(selected.identity.first_seen_at)} • Last Seen:{" "}
                  {formatTimestamp(selected.identity.last_seen_at)} • Total Observations:{" "}
                  {selected.observations?.length || selected.identity.observation_count || 0}
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setSelected(null)}
              className="inline-flex items-center gap-1 px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 text-xs font-medium transition"
            >
              <X className="w-4 h-4" />
              Close Details
            </button>
          </div>

          {/* Camera Journey Timeline */}
          <div>
            <div className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-1.5">
              <Camera className="w-3.5 h-3.5 text-sky-500" />
              Multi-Camera Journey Timeline
            </div>
            <div className="flex flex-wrap items-center gap-2 p-3 rounded-xl bg-slate-50 dark:bg-[#070d22] border border-slate-200 dark:border-slate-800/80">
              {(selected.identity.camera_sequence || []).length === 0 ? (
                <span className="text-xs text-slate-500">No cross-camera transitions yet.</span>
              ) : (
                selected.identity.camera_sequence.map((camCode, idx) => (
                  <React.Fragment key={idx}>
                    {idx > 0 && <ArrowRight className="w-3.5 h-3.5 text-slate-400" />}
                    <div className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-white dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 shadow-xs">
                      <span className="w-2 h-2 rounded-full bg-sky-500" />
                      <span className="font-mono text-xs font-semibold text-slate-900 dark:text-white">
                        {camCode}
                      </span>
                    </div>
                  </React.Fragment>
                ))
              )}
            </div>
          </div>

          {/* Observation Records */}
          <div>
            <div className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-1.5">
              <History className="w-3.5 h-3.5 text-violet-500" />
              Tracklet Observations &amp; Feature History
            </div>
            {(selected.observations || []).length === 0 ? (
              <p className="text-xs text-slate-500 p-3 rounded-xl bg-slate-50 dark:bg-[#070d22]">
                No detailed observation records stored for this identity.
              </p>
            ) : (
              <div className="max-h-56 overflow-y-auto rounded-xl border border-slate-200 dark:border-slate-800">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-100 dark:bg-slate-800/80 text-slate-600 dark:text-slate-300 sticky top-0">
                    <tr>
                      <th className="py-2 px-3">Camera</th>
                      <th className="py-2 px-3">Track ID</th>
                      <th className="py-2 px-3">Observation Time</th>
                      <th className="py-2 px-3">Embedding ID</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                    {selected.observations.map((obs, idx) => (
                      <tr key={idx} className="hover:bg-slate-50/40 dark:hover:bg-slate-800/30">
                        <td className="py-2 px-3 font-mono text-slate-800 dark:text-slate-200">
                          {obs.camera_code || obs.camera_id}
                        </td>
                        <td className="py-2 px-3 font-mono text-slate-600 dark:text-slate-400">
                          #{obs.local_track_id}
                        </td>
                        <td className="py-2 px-3 text-slate-600 dark:text-slate-400">
                          {formatTimestamp(obs.timestamp)}
                        </td>
                        <td className="py-2 px-3 font-mono text-[11px] text-sky-600 dark:text-sky-400 truncate max-w-[150px]">
                          {obs.embedding_id || "emb_stored"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* AI Audit Trail (Decision history) */}
          {selected.audit_trail && selected.audit_trail.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-1.5">
                <ShieldAlert className="w-3.5 h-3.5 text-emerald-500" />
                AI Re-ID Association Decisions
              </div>
              <div className="max-h-48 overflow-y-auto rounded-xl border border-slate-200 dark:border-slate-800">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-100 dark:bg-slate-800/80 text-slate-600 dark:text-slate-300 sticky top-0">
                    <tr>
                      <th className="py-2 px-3">Time</th>
                      <th className="py-2 px-3">Decision</th>
                      <th className="py-2 px-3">Confidence</th>
                      <th className="py-2 px-3">Algorithmic Justification</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                    {selected.audit_trail.map((a, idx) => (
                      <tr key={idx} className="hover:bg-slate-50/40 dark:hover:bg-slate-800/30">
                        <td className="py-2 px-3 text-slate-600 dark:text-slate-400">
                          {formatTimestamp(a.timestamp)}
                        </td>
                        <td className="py-2 px-3 font-semibold text-sky-600 dark:text-sky-400">
                          {a.decision}
                        </td>
                        <td className="py-2 px-3 font-mono text-slate-700 dark:text-slate-300">
                          {a.confidence}
                        </td>
                        <td className="py-2 px-3 text-slate-600 dark:text-slate-400">
                          {a.reason}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function RecoveryHistoryPage() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const { data } = await api.get("/api/identities/recovery-history");
      setEvents(data.recovery_events || []);
    } catch {
      // Handled gracefully
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="glass-card rounded-2xl p-6 overflow-x-auto shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-display text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <History className="w-5 h-5 text-sky-500" />
            Identity Recovery History
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Log of all successful cross-camera and temporal identity recoveries
          </p>
        </div>
        <button
          type="button"
          onClick={load}
          disabled={loading}
          className="p-2 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700 shadow-sm transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {loading && events.length === 0 ? (
        <p className="text-sm text-slate-500 dark:text-slate-400">Loading recovery events...</p>
      ) : events.length === 0 ? (
        <p className="text-sm text-slate-500 dark:text-slate-400">No recovery events logged yet.</p>
      ) : (
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
            <tr>
              <th className="py-2.5 pr-4">Global ID</th>
              <th className="py-2.5 pr-4">Camera</th>
              <th className="py-2.5 pr-4">Track ID</th>
              <th className="py-2.5 pr-4">Decision</th>
              <th className="py-2.5 pr-4">Confidence</th>
              <th className="py-2.5 pr-4">Review Status</th>
              <th className="py-2.5 pr-4">Reason</th>
              <th className="py-2.5 pr-4">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60">
            {events.map((ev) => (
              <tr key={ev.id}>
                <td className="py-2.5 pr-4 font-mono text-sky-600 dark:text-sky-300 font-semibold">
                  {ev.global_id || "—"}
                </td>
                <td className="py-2.5 pr-4 font-mono text-xs">{ev.to_camera_code}</td>
                <td className="py-2.5 pr-4 font-mono text-xs">#{ev.local_track_id}</td>
                <td className="py-2.5 pr-4 font-semibold text-xs text-emerald-600 dark:text-emerald-400">
                  {ev.decision}
                </td>
                <td className="py-2.5 pr-4 font-mono text-xs">{ev.confidence}</td>
                <td className="py-2.5 pr-4">
                  <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-50 dark:bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30">
                    {ev.review_status}
                  </span>
                </td>
                <td className="py-2.5 pr-4 text-xs text-slate-500 dark:text-slate-400 max-w-xs truncate">
                  {ev.reason}
                </td>
                <td className="py-2.5 pr-4 text-xs text-slate-500 dark:text-slate-400">
                  {formatTimestamp(ev.timestamp)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function UncertainReviewPage() {
  const [items, setItems] = useState([]);
  const [identities, setIdentities] = useState([]);
  const [targetIds, setTargetIds] = useState({});
  const [reasons, setReasons] = useState({});
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  const load = async () => {
    setLoading(true);
    setErr("");
    try {
      const [uncRes, idsRes] = await Promise.all([
        api.get("/api/identities/uncertain"),
        api.get("/api/identities"),
      ]);
      setItems(uncRes.data.uncertain || []);
      setIdentities(idsRes.data.identities || []);
    } catch (e) {
      setErr(e?.response?.data?.detail || "Failed to load uncertain review queue.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const submitReview = async (obs, decision) => {
    setMsg("");
    setErr("");
    const chosenTarget = (targetIds[obs.observation_id] || obs.candidate_global_id || "").trim();
    if (decision === "ACCEPT" && !chosenTarget) {
      setErr("Please select or enter a Target Global ID before accepting.");
      return;
    }
    try {
      await api.post(`/api/identities/${encodeURIComponent(obs.observation_id)}/review`, {
        decision,
        target_global_id: chosenTarget || null,
        reason: (reasons[obs.observation_id] || `Operator ${decision.toLowerCase()} decision`).trim(),
        operator_confirmed: true,
      });
      setMsg(`Reviewed ${obs.observation_id} (${decision}).`);
      await load();
    } catch (e) {
      setErr(e?.response?.data?.detail || "Review failed.");
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-display text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            Uncertain Identity Review Queue
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Ambiguous re-identifications requiring human operator confirmation
          </p>
        </div>
        <button
          type="button"
          onClick={load}
          className="px-3.5 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-sky-700 border border-slate-300 dark:bg-slate-800 dark:hover:bg-slate-700 dark:text-sky-300 dark:border-transparent text-xs font-semibold shadow-sm transition"
        >
          Refresh Queue
        </button>
      </div>

      {msg && (
        <div className="p-3 rounded-xl bg-emerald-50 dark:bg-emerald-500/15 border border-emerald-300 dark:border-transparent text-emerald-800 dark:text-emerald-300 text-xs font-medium">
          {msg}
        </div>
      )}
      {err && (
        <div className="p-3 rounded-xl bg-rose-50 dark:bg-rose-500/15 border border-rose-300 dark:border-transparent text-rose-800 dark:text-rose-300 text-xs font-medium">
          {err}
        </div>
      )}

      {loading ? (
        <div className="glass-card rounded-2xl p-6 text-sm text-slate-500 dark:text-slate-400 shadow-sm">
          Loading uncertain observations...
        </div>
      ) : items.length === 0 ? (
        <div className="glass-card rounded-2xl p-6 space-y-2 shadow-sm">
          <div className="text-sm text-slate-900 dark:text-slate-200 font-semibold flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            No uncertain identity observations awaiting operator confirmation.
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400">
            All current detections matched with high confidence or were assigned distinct Global IDs ({identities.length} total identities tracked). When an ambiguous match occurs, it will appear here for manual Accept/Reject confirmation.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {items.map((obs) => (
            <div key={obs.observation_id} className="glass-card rounded-2xl p-5 space-y-3 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="font-mono text-sm text-amber-700 dark:text-amber-300 font-semibold">
                  {obs.observation_id} • {obs.camera_code} (Track #{obs.local_track_id})
                </div>
                <span className="text-xs text-slate-600 dark:text-slate-300">
                  Confidence: {obs.confidence}
                </span>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-400">Reason: {obs.reason}</p>
              <div className="grid sm:grid-cols-2 gap-3">
                <div className="space-y-2">
                  <select
                    value={targetIds[obs.observation_id] ?? obs.candidate_global_id ?? ""}
                    onChange={(e) =>
                      setTargetIds({ ...targetIds, [obs.observation_id]: e.target.value })
                    }
                    className="w-full rounded-xl bg-white dark:bg-[#080f26] border border-slate-300 dark:border-slate-700 px-3 py-2 text-xs text-slate-900 dark:text-white shadow-sm"
                  >
                    <option value="">-- Select Target Global ID --</option>
                    {identities.map((id) => (
                      <option key={id.global_id} value={id.global_id}>
                        {id.global_id} ({id.lifecycle_state})
                      </option>
                    ))}
                  </select>
                  <input
                    placeholder={`Or type Global ID (candidate: ${obs.candidate_global_id || "none"})`}
                    value={targetIds[obs.observation_id] || ""}
                    onChange={(e) =>
                      setTargetIds({ ...targetIds, [obs.observation_id]: e.target.value })
                    }
                    className="w-full rounded-xl bg-white dark:bg-[#080f26] border border-slate-300 dark:border-slate-700 px-3 py-2 text-xs text-slate-900 dark:text-white shadow-sm"
                  />
                </div>
                <input
                  placeholder="Review justification (min 3 chars)"
                  value={reasons[obs.observation_id] || ""}
                  onChange={(e) =>
                    setReasons({ ...reasons, [obs.observation_id]: e.target.value })
                  }
                  className="rounded-xl bg-white dark:bg-[#080f26] border border-slate-300 dark:border-slate-700 px-3 py-2 text-xs text-slate-900 dark:text-white self-start shadow-sm"
                />
              </div>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => submitReview(obs, "ACCEPT")}
                  className="px-4 py-1.5 rounded-xl bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-300 dark:bg-emerald-500/20 dark:border-emerald-500/40 dark:text-emerald-300 text-xs font-semibold shadow-sm transition"
                >
                  Confirm &amp; Accept
                </button>
                <button
                  type="button"
                  onClick={() => submitReview(obs, "REJECT")}
                  className="px-4 py-1.5 rounded-xl bg-rose-50 hover:bg-rose-100 text-rose-800 border border-rose-300 dark:bg-rose-500/20 dark:border-rose-500/40 dark:text-rose-300 text-xs font-semibold shadow-sm transition"
                >
                  Reject Match
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
