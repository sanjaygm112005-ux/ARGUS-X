import React, { useEffect, useState } from "react";
import {
  AlertCircle,
  Camera,
  CheckCircle2,
  GitCompare,
  Loader2,
  Play,
  Plus,
  Trash2,
  Upload,
} from "lucide-react";
import api from "../../services/api";

export function MultiCameraAnalysisPage() {
  const [slots, setSlots] = useState([
    { camera_code: "cam_1", camera_name: "Camera 1 (Entry)", file: null, video_uid: "", uploading: false },
    { camera_code: "cam_2", camera_name: "Camera 2 (Corridor)", file: null, video_uid: "", uploading: false },
  ]);
  const [maxFrames, setMaxFrames] = useState(60);
  const [starting, setStarting] = useState(false);
  const [activeSession, setActiveSession] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .get("/api/multicamera/sessions")
      .then(({ data }) => {
        const list = data.sessions || [];
        setSessions(list);
        if (list.length > 0 && !activeSession) setActiveSession(list[0]);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!activeSession || !["QUEUED", "PROCESSING"].includes(activeSession.status)) return;
    const timer = setInterval(async () => {
      try {
        const { data } = await api.get(`/api/multicamera/sessions/${activeSession.session_uid}`);
        if (data.session) {
          setActiveSession(data.session);
          setSessions((prev) =>
            prev.map((s) => (s.session_uid === data.session.session_uid ? data.session : s))
          );
        }
      } catch {
        // ignore transient poll error
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [activeSession]);

  const updateSlot = (idx, patch) => {
    setSlots((prev) => prev.map((s, i) => (i === idx ? { ...s, ...patch } : s)));
  };

  const addSlot = () => {
    const n = slots.length + 1;
    setSlots((prev) => [
      ...prev,
      { camera_code: `cam_${n}`, camera_name: `Camera ${n}`, file: null, video_uid: "", uploading: false },
    ]);
  };

  const removeSlot = (idx) => {
    if (slots.length <= 2) return;
    setSlots((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleFileSelect = async (idx, file) => {
    if (!file) return;
    updateSlot(idx, { file, uploading: true });
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("camera_label", slots[idx].camera_name || slots[idx].camera_code);
      const { data } = await api.post("/api/videos/upload", fd);
      updateSlot(idx, { video_uid: data.video.video_uid, uploading: false });
    } catch (err) {
      setError(err?.response?.data?.detail || `Failed to upload video for ${slots[idx].camera_code}`);
      updateSlot(idx, { uploading: false });
    }
  };

  const handleStartRecovery = async () => {
    const ready = slots.every((s) => s.video_uid && s.camera_code.trim());
    if (!ready) {
      setError("Please upload a video and assign a camera ID for every camera slot (minimum 2).");
      return;
    }
    setStarting(true);
    setError("");
    try {
      const { data } = await api.post("/api/multicamera/analyze", {
        cameras: slots.map((s) => ({
          video_uid: s.video_uid,
          camera_code: s.camera_code.trim(),
          camera_name: s.camera_name.trim() || s.camera_code.trim(),
        })),
        max_frames_per_camera: Number(maxFrames) || 60,
      });
      setActiveSession(data.session);
      setSessions((prev) => [data.session, ...prev]);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to start multi-camera identity recovery.");
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-display text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <GitCompare className="w-6 h-6 text-violet-400" />
          Multi-Camera Upload & Persistent Identity Recovery
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Upload multiple camera feeds, assign Camera IDs, and run cross-camera Re-ID with conflict-safe operator review.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/15 border border-rose-500/40 text-rose-300 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Camera Slots */}
      <div className="glass-card rounded-2xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
            Camera Video Inputs ({slots.length} Cameras)
          </h3>
          <button
            type="button"
            onClick={addSlot}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-sky-500/15 border border-sky-400/30 text-sky-300 text-xs font-semibold hover:bg-sky-500/25"
          >
            <Plus className="w-3.5 h-3.5" />
            Add Camera Video
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {slots.map((slot, idx) => (
            <div key={idx} className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-sky-400 flex items-center gap-1.5">
                  <Camera className="w-4 h-4" />
                  Camera Slot #{idx + 1}
                </span>
                {slots.length > 2 && (
                  <button
                    type="button"
                    onClick={() => removeSlot(idx)}
                    className="text-slate-400 hover:text-rose-400"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[11px] text-slate-400 mb-1">Camera ID</label>
                  <input
                    type="text"
                    value={slot.camera_code}
                    onChange={(e) => updateSlot(idx, { camera_code: e.target.value })}
                    className="w-full rounded-lg px-2.5 py-1.5 text-xs border"
                  />
                </div>
                <div>
                  <label className="block text-[11px] text-slate-400 mb-1">Camera Name</label>
                  <input
                    type="text"
                    value={slot.camera_name}
                    onChange={(e) => updateSlot(idx, { camera_name: e.target.value })}
                    className="w-full rounded-lg px-2.5 py-1.5 text-xs border"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between gap-2">
                <input
                  type="file"
                  accept=".mp4,.avi,.mov,.mkv,video/*"
                  onChange={(e) => handleFileSelect(idx, e.target.files?.[0])}
                  className="text-xs text-slate-300 file:mr-2 file:py-1.5 file:px-2.5 file:rounded-lg file:border-0 file:bg-violet-500/20 file:text-violet-300"
                />
                {slot.uploading ? (
                  <Loader2 className="w-4 h-4 animate-spin text-sky-400 shrink-0" />
                ) : slot.video_uid ? (
                  <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400 shrink-0">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Ready
                  </span>
                ) : (
                  <Upload className="w-4 h-4 text-slate-500 shrink-0" />
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-400">Max Frames / Camera:</label>
            <input
              type="number"
              min={5}
              max={2000}
              value={maxFrames}
              onChange={(e) => setMaxFrames(e.target.value)}
              className="w-24 rounded-xl px-2.5 py-1.5 text-xs border"
            />
          </div>

          <button
            type="button"
            disabled={starting || slots.some((s) => !s.video_uid || s.uploading)}
            onClick={handleStartRecovery}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-700 to-indigo-700 text-white text-sm font-semibold disabled:opacity-50"
          >
            {starting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Start Identity Recovery
          </button>
        </div>
      </div>

      {/* Active Session Results */}
      {activeSession && (
        <div className="space-y-5">
          <div className="glass-card rounded-2xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-display font-semibold text-lg text-slate-900 dark:text-white">
                  Session {activeSession.session_uid}
                </h3>
                <p className="text-xs text-slate-400">
                  Cross-camera Hungarian + FAISS identity recovery (never auto-merges uncertain matches)
                </p>
              </div>
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-sky-500/20 text-sky-300">
                {activeSession.status} ({activeSession.progress}%)
              </span>
            </div>

            <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-sky-500 to-violet-600 transition-all"
                style={{ width: `${activeSession.progress || 0}%` }}
              />
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
              <div className="p-3 rounded-xl bg-slate-900/50 border border-slate-800">
                <p className="text-[11px] text-slate-400">Accepted / Recovered</p>
                <p className="text-xl font-bold text-emerald-400">
                  {activeSession.counts?.accepted_matches ?? 0}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/50 border border-slate-800">
                <p className="text-[11px] text-slate-400">Uncertain (Queued)</p>
                <p className="text-xl font-bold text-amber-400">
                  {activeSession.counts?.uncertain_matches ?? 0}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/50 border border-slate-800">
                <p className="text-[11px] text-slate-400">New Identities</p>
                <p className="text-xl font-bold text-sky-400">
                  {activeSession.counts?.new_identities ?? 0}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/50 border border-slate-800">
                <p className="text-[11px] text-slate-400">Rejected Matches</p>
                <p className="text-xl font-bold text-rose-400">
                  {activeSession.counts?.rejected_matches ?? 0}
                </p>
              </div>
            </div>
          </div>

          {/* Camera-wise Summary */}
          {activeSession.camera_summaries?.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {activeSession.camera_summaries.map((cs) => (
                <div key={cs.camera_code} className="glass-card rounded-2xl p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-sm text-sky-400">{cs.camera_code}</span>
                    <span className="text-xs text-slate-400">{cs.frames_processed} frames</span>
                  </div>
                  <p className="text-xs text-slate-300">
                    Local Track IDs:{" "}
                    <span className="font-mono text-slate-100">
                      {cs.local_track_ids.length ? cs.local_track_ids.join(", ") : "None"}
                    </span>
                  </p>
                  <p className="text-xs text-slate-300">
                    Global Identity IDs:{" "}
                    <span className="font-mono text-violet-300">
                      {cs.global_ids.length ? cs.global_ids.join(", ") : "None"}
                    </span>
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Recovery Event Log */}
          {activeSession.recovery_events?.length > 0 && (
            <div className="glass-card rounded-2xl p-5">
              <h4 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">
                Identity Recovery & Association Decisions
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b">
                      <th className="py-2 pr-3">Camera</th>
                      <th className="py-2 pr-3">Local Track</th>
                      <th className="py-2 pr-3">Global ID</th>
                      <th className="py-2 pr-3">Decision</th>
                      <th className="py-2 pr-3">Confidence</th>
                      <th className="py-2 pr-3">Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeSession.recovery_events.map((ev, i) => (
                      <tr key={i} className="border-b">
                        <td className="py-2 pr-3 font-mono text-sky-400">{ev.camera_code}</td>
                        <td className="py-2 pr-3">#{ev.local_track_id}</td>
                        <td className="py-2 pr-3 font-mono text-violet-300">{ev.global_id || "—"}</td>
                        <td className="py-2 pr-3">
                          <span
                            className={`px-2 py-0.5 rounded-full font-medium ${
                              ev.decision === "RECOVERED"
                                ? "bg-emerald-500/20 text-emerald-300"
                                : ev.decision === "UNCERTAIN"
                                ? "bg-amber-500/20 text-amber-300"
                                : "bg-sky-500/20 text-sky-300"
                            }`}
                          >
                            {ev.decision}
                          </span>
                        </td>
                        <td className="py-2 pr-3">{(ev.confidence * 100).toFixed(1)}%</td>
                        <td className="py-2 pr-3 text-slate-400">{ev.reason}</td>
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
