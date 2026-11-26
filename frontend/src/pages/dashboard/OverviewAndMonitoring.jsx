import React, { useState } from "react";
import { Link, useOutletContext } from "react-router-dom";
import { Activity, AlertTriangle, Camera, Users } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import api from "../../services/api";
import { useAuth } from "../../context/AuthContext";
import ZoomableVideoFeed from "../../components/ZoomableVideoFeed";

export function AdminOverviewPage() {
  const { snapshot, connectionState } = useOutletContext();
  const stats = snapshot?.statistics || {
    total_cameras: 0,
    online_cameras: 0,
    total_identities: 0,
    active_identities: 0,
    recovered_identities: 0,
    uncertain_count: 0,
  };
  const cameras = snapshot?.cameras || [];

  const cards = [
    { label: "Total Cameras", value: stats.total_cameras, icon: Camera, color: "text-sky-400" },
    { label: "Online Cameras", value: stats.online_cameras, icon: Activity, color: "text-emerald-400" },
    { label: "Active Identities", value: stats.active_identities, icon: Users, color: "text-violet-400" },
    { label: "Uncertain Alerts", value: stats.uncertain_count, icon: AlertTriangle, color: "text-amber-400" },
  ];

  const chartData = cameras.map((c) => ({
    name: c.camera_code,
    activeTracks: c.active_tracks || 0,
  }));

  return (
    <div className="space-y-6">
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((c) => {
          const Icon = c.icon;
          return (
            <div key={c.label} className="glass-card rounded-2xl p-5 flex items-center justify-between">
              <div>
                <div className="text-xs text-slate-500 dark:text-slate-400 font-medium">{c.label}</div>
                <div className="mt-1 font-display text-3xl font-bold text-slate-900 dark:text-white">{c.value}</div>
              </div>
              <Icon className={`w-7 h-7 ${c.color}`} />
            </div>
          );
        })}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="glass-card rounded-2xl p-6">
          <h2 className="font-display text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Active Tracks per Camera
          </h2>
          {chartData.length === 0 ? (
            <div className="h-56 flex items-center justify-center text-sm text-slate-500 dark:text-slate-400">
              No cameras registered yet. Connection: {connectionState}.
            </div>
          ) : (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" className="dark:stroke-slate-800" />
                  <XAxis dataKey="name" stroke="#64748b" className="dark:stroke-slate-400" fontSize={12} />
                  <YAxis stroke="#64748b" className="dark:stroke-slate-400" fontSize={12} allowDecimals={false} />
                  <Tooltip contentStyle={{ backgroundColor: "#ffffff", borderColor: "#cbd5e1", borderRadius: "12px", color: "#0f172a" }} />
                  <Bar dataKey="activeTracks" fill="#0284c7" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="glass-card rounded-2xl p-6">
          <h2 className="font-display text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Live Alerts &amp; Camera Disconnections
          </h2>
          <div className="space-y-2.5 max-h-56 overflow-y-auto">
            {(snapshot?.uncertain_alerts || []).length === 0 &&
              cameras.every((c) => c.status !== "ERROR") ? (
              <div className="text-sm text-slate-500 dark:text-slate-400">No active alerts or disconnections.</div>
            ) : (
              (snapshot?.uncertain_alerts || []).map((a) => (
                <div
                  key={a.observation_id}
                  className="p-3 rounded-xl bg-amber-50 dark:bg-amber-500/10 border border-amber-300 dark:border-amber-500/30 text-xs text-amber-800 dark:text-amber-200"
                >
                  Uncertain observation <strong>{a.observation_id}</strong> on {a.camera_code} (conf:{" "}
                  {a.confidence}) — {a.reason}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function LiveMonitoringPage() {
  const { user } = useAuth();
  const { snapshot, connectionState } = useOutletContext();
  const cameras = snapshot?.cameras || [];
  const identities = snapshot?.identities || [];
  const trackingEvents = snapshot?.tracking_events || [];
  const waitingIdentities = identities.filter((i) => i.lifecycle_state === "TEMPORARILY_LOST");
  const activeOrRecovered = identities.filter((i) =>
    ["ACTIVE", "RECOVERED"].includes(i.lifecycle_state)
  );
  const isAdmin = user?.role === "ADMIN";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="font-display text-xl font-bold text-slate-900 dark:text-white">Live Camera Feeds &amp; Tracking</h2>
        <div className="flex items-center gap-3">
          {isAdmin && (
            <Link
              to="/admin/cameras"
              className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-sky-700 dark:text-sky-300 text-xs font-semibold transition border border-slate-200 dark:border-transparent"
            >
              Manage Cameras
            </Link>
          )}
          <span className="text-xs text-slate-500 dark:text-slate-400">Stream status: {connectionState}</span>
        </div>
      </div>

      {cameras.length === 0 ? (
        <div className="glass-card rounded-2xl p-8 text-center space-y-4">
          <p className="text-sm text-slate-500 dark:text-slate-400">
            No cameras registered yet.
          </p>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-5">
          {cameras.map((cam) => (
            <div key={cam.camera_code} className="glass-card rounded-2xl overflow-hidden shadow-sm">
              <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
                <div>
                  <div className="font-display font-semibold text-slate-900 dark:text-white">{cam.name}</div>
                  <div className="text-xs font-mono text-slate-500 dark:text-slate-400">
                    {cam.camera_code} {cam.location ? `• ${cam.location}` : ""}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={async () => {
                      const action = cam.is_running ? "stop" : "start";
                      try {
                        await api.put(`/api/cameras/${cam.id}`, { action });
                      } catch (e) {
                        console.error(e);
                      }
                    }}
                    className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition ${cam.is_running
                      ? "bg-rose-50 hover:bg-rose-100 dark:bg-rose-500/15 dark:hover:bg-rose-500/25 text-rose-700 dark:text-rose-300 border border-rose-300 dark:border-rose-500/30"
                      : "bg-emerald-50 hover:bg-emerald-100 dark:bg-emerald-500/15 dark:hover:bg-emerald-500/25 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30"
                      }`}
                  >
                    {cam.is_running ? "Stop Stream" : "Start Stream"}
                  </button>
                  <span
                    className={`px-2.5 py-0.5 rounded-full text-xs font-medium border ${cam.is_running
                      ? "bg-emerald-50 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-transparent"
                      : "bg-slate-100 dark:bg-slate-700/60 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-transparent"
                      }`}
                  >
                    {cam.status}
                  </span>
                </div>
              </div>
              <ZoomableVideoFeed cam={cam} />
              <div className="p-3 bg-slate-50 dark:bg-[#080e24] border-t border-slate-200 dark:border-slate-800 space-y-2 text-xs text-slate-700 dark:text-slate-300">
                <div className="flex items-center justify-between">
                  <span>Active Tracks: {cam.active_tracks}</span>
                  <span className="font-mono text-slate-500 dark:text-slate-400 truncate max-w-[160px]">
                    {cam.source || "No source"}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {activeOrRecovered
                    .filter((i) => i.current_camera_code === cam.camera_code)
                    .map((i, idx) => (
                      <span
                        key={i.global_id}
                        className={`px-2 py-0.5 rounded font-mono text-[11px] ${i.lifecycle_state === "RECOVERED"
                          ? "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30"
                          : "bg-sky-100 dark:bg-sky-500/20 text-sky-800 dark:text-sky-300 border border-sky-300 dark:border-sky-500/30"
                          }`}
                      >
                        PERSON {idx + 1} ({i.lifecycle_state})
                      </span>
                    ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {waitingIdentities.length > 0 && (
        <div className="glass-card rounded-2xl p-4 border border-amber-300 dark:border-amber-500/30 bg-amber-50/60 dark:bg-amber-500/5">
          <div className="text-xs font-semibold text-amber-800 dark:text-amber-300 mb-2">
            Waiting for Reappearance (TEMPORARILY_LOST — preserving Global IDs)
          </div>
          <div className="flex flex-wrap gap-2">
            {waitingIdentities.map((i) => (
              <span
                key={i.global_id}
                className="px-2.5 py-1 rounded-lg bg-amber-100 dark:bg-amber-500/15 text-amber-800 dark:text-amber-200 font-mono text-xs border border-amber-200 dark:border-transparent"
              >
                {i.global_id} • last seen: {i.current_camera_code || "unknown"}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="glass-card rounded-2xl p-5">
        <h3 className="font-display text-base font-semibold text-slate-900 dark:text-white mb-3">
          Recent Tracking Events
        </h3>
        {trackingEvents.length === 0 ? (
          <p className="text-xs text-slate-500 dark:text-slate-400">No tracking events persisted yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  <th className="py-2 pr-4">Camera ID</th>
                  <th className="py-2 pr-4">Local Track</th>
                  <th className="py-2 pr-4">Global ID</th>
                  <th className="py-2 pr-4">Decision</th>
                  <th className="py-2 pr-4">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60">
                {trackingEvents.map((ev) => (
                  <tr key={ev.id}>
                    <td className="py-2 pr-4">{ev.camera_id}</td>
                    <td className="py-2 pr-4 font-mono">{ev.local_track_id}</td>
                    <td className="py-2 pr-4 font-mono text-sky-600 dark:text-sky-300">{ev.global_id || "—"}</td>
                    <td className="py-2 pr-4">{ev.decision}</td>
                    <td className="py-2 pr-4">{Number(ev.confidence).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
