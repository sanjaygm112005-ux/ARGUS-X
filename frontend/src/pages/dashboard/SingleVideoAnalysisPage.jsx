import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  Film,
  Loader2,
  Play,
  Trash2,
  Upload,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import api, { apiUrl } from "../../services/api";

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(2)} MB` : `${(bytes / 1024).toFixed(1)} KB`;
}

export function SingleVideoAnalysisPage() {
  const [videos, setVideos] = useState([]);
  const [selected, setSelected] = useState(null);
  const [file, setFile] = useState(null);
  const [cameraLabel, setCameraLabel] = useState("Entrance Camera 1");
  const [maxFrames, setMaxFrames] = useState("");
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [streamKind, setStreamKind] = useState("processed");
  const [error, setError] = useState("");

  const loadVideos = useCallback(async () => {
    try {
      const { data } = await api.get("/api/videos");
      const list = data.videos || [];
      setVideos(list);
      if (list.length > 0 && !selected) {
        setSelected(list[0]);
      } else if (selected) {
        const updated = list.find((v) => v.video_uid === selected.video_uid);
        if (updated) setSelected(updated);
      }
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to load uploaded videos.");
    }
  }, [selected]);

  useEffect(() => {
    loadVideos();
  }, []);

  // Poll while QUEUED or PROCESSING
  useEffect(() => {
    if (!selected || !["QUEUED", "PROCESSING"].includes(selected.analysis_status)) return;
    const timer = setInterval(async () => {
      try {
        const { data } = await api.get(`/api/videos/${selected.video_uid}/results`);
        if (data.video) {
          setSelected(data.video);
          setVideos((prev) =>
            prev.map((v) => (v.video_uid === data.video.video_uid ? data.video : v))
          );
        }
      } catch {
        // ignore transient poll errors
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [selected]);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("camera_label", cameraLabel.trim() || "Single Camera");
      const { data } = await api.post("/api/videos/upload", fd);
      setSelected(data.video);
      setVideos((prev) => [data.video, ...prev]);
      setFile(null);
    } catch (err) {
      setError(err?.response?.data?.detail || "Video upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!selected) return;
    setAnalyzing(true);
    setError("");
    try {
      const body = maxFrames ? { max_frames: Number(maxFrames) } : {};
      const { data } = await api.post(`/api/videos/${selected.video_uid}/analyze`, body);
      setSelected(data.video);
      setVideos((prev) =>
        prev.map((v) => (v.video_uid === data.video.video_uid ? data.video : v))
      );
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to start analysis.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleDelete = async (uid) => {
    try {
      await api.delete(`/api/videos/${uid}`);
      setVideos((prev) => prev.filter((v) => v.video_uid !== uid));
      if (selected?.video_uid === uid) setSelected(null);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to delete video.");
    }
  };

  const results = selected?.results;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-display text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Film className="w-6 h-6 text-sky-400" />
          Single Video Upload & AI Analysis
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Upload private MP4/MOV/AVI/MKV footage, process with YOLO11 + BoT-SORT + OSNet, and inspect tracking analytics.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/15 border border-rose-500/40 text-rose-300 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Upload Card */}
      <form onSubmit={handleUpload} className="glass-card rounded-2xl p-5 grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <div className="md:col-span-2">
          <label className="block text-xs text-slate-400 mb-1.5">Video File (MP4, AVI, MOV, MKV — max 200 MB)</label>
          <input
            type="file"
            accept=".mp4,.avi,.mov,.mkv,video/*"
            required
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="w-full text-xs text-slate-300 file:mr-3 file:py-2 file:px-3 file:rounded-xl file:border-0 file:bg-sky-500/20 file:text-sky-300"
          />
        </div>
        <div>
          <label className="block text-xs text-slate-400 mb-1.5">Camera Label</label>
          <input
            type="text"
            value={cameraLabel}
            onChange={(e) => setCameraLabel(e.target.value)}
            placeholder="e.g. Lobby East"
            className="w-full rounded-xl px-3 py-2 text-sm border"
          />
        </div>
        <button
          type="submit"
          disabled={uploading || !file}
          className="flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-gradient-to-r from-blue-700 to-indigo-700 text-white text-sm font-semibold disabled:opacity-50"
        >
          {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
          {uploading ? "Uploading..." : "Upload Video"}
        </button>
      </form>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video List */}
        <div className="glass-card rounded-2xl p-4 space-y-3">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
            Private Uploaded Videos ({videos.length})
          </h3>
          <div className="space-y-2 max-h-[460px] overflow-y-auto">
            {videos.map((v) => (
              <div
                key={v.video_uid}
                onClick={() => setSelected(v)}
                className={`p-3 rounded-xl border cursor-pointer transition ${
                  selected?.video_uid === v.video_uid
                    ? "border-sky-400 bg-sky-500/10"
                    : "border-slate-200 dark:border-slate-800 hover:border-slate-600"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-slate-900 dark:text-white truncate">
                    {v.original_filename}
                  </span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(v.video_uid);
                    }}
                    className="text-slate-400 hover:text-rose-400"
                    title="Delete video"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
                <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-slate-400">
                  <span>{v.camera_label}</span>
                  <span>•</span>
                  <span>{formatBytes(v.file_size_bytes)}</span>
                  <span>•</span>
                  <span>{v.duration_sec}s</span>
                </div>
                <div className="mt-1.5 flex items-center justify-between text-[11px]">
                  <span className="text-sky-400">
                    {v.width}×{v.height} @ {v.fps} FPS
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-full font-medium ${
                      v.analysis_status === "COMPLETED"
                        ? "bg-emerald-500/20 text-emerald-300"
                        : v.analysis_status === "FAILED"
                        ? "bg-rose-500/20 text-rose-300"
                        : "bg-violet-500/20 text-violet-300"
                    }`}
                  >
                    {v.analysis_status}
                  </span>
                </div>
              </div>
            ))}
            {videos.length === 0 && (
              <p className="text-xs text-slate-400 text-center py-6">No videos uploaded yet.</p>
            )}
          </div>
        </div>

        {/* Video Player & Analysis */}
        <div className="lg:col-span-2 space-y-5">
          {selected ? (
            <>
              <div className="glass-card rounded-2xl p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="font-display font-semibold text-lg text-slate-900 dark:text-white">
                      {selected.original_filename}
                    </h3>
                    <p className="text-xs text-slate-400">
                      Label: {selected.camera_label} • Size: {formatBytes(selected.file_size_bytes)} • Duration:{" "}
                      {selected.duration_sec}s • {selected.total_frames} frames
                    </p>
                  </div>

                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      placeholder="All frames"
                      min={10}
                      max={5000}
                      value={maxFrames}
                      onChange={(e) => setMaxFrames(e.target.value)}
                      className="w-28 rounded-xl px-2.5 py-2 text-xs border"
                    />
                    <button
                      type="button"
                      disabled={analyzing || ["QUEUED", "PROCESSING"].includes(selected.analysis_status)}
                      onClick={handleAnalyze}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-blue-700 to-indigo-700 text-white text-xs font-semibold disabled:opacity-50"
                    >
                      {["QUEUED", "PROCESSING"].includes(selected.analysis_status) ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Play className="w-3.5 h-3.5" />
                      )}
                      {["QUEUED", "PROCESSING"].includes(selected.analysis_status)
                        ? `Processing (${selected.analysis_progress}%)`
                        : "Analyze Video"}
                    </button>
                  </div>
                </div>

                {["QUEUED", "PROCESSING"].includes(selected.analysis_status) && (
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-sky-500 to-violet-600 transition-all"
                      style={{ width: `${selected.analysis_progress || 0}%` }}
                    />
                  </div>
                )}

                {selected.has_processed_video && (
                  <div className="flex items-center gap-2 text-xs">
                    <button
                      type="button"
                      onClick={() => setStreamKind("processed")}
                      className={`px-3 py-1.5 rounded-lg font-medium ${
                        streamKind === "processed"
                          ? "bg-sky-500/20 text-sky-300 border border-sky-400/40"
                          : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      Processed (Tracked Overlay)
                    </button>
                    <button
                      type="button"
                      onClick={() => setStreamKind("original")}
                      className={`px-3 py-1.5 rounded-lg font-medium ${
                        streamKind === "original"
                          ? "bg-sky-500/20 text-sky-300 border border-sky-400/40"
                          : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      Original Video
                    </button>
                  </div>
                )}

                <div className="rounded-xl overflow-hidden bg-black aspect-video">
                  <video
                    key={`${selected.video_uid}-${streamKind}-${selected.analysis_status}`}
                    controls
                    className="w-full h-full object-contain"
                    src={apiUrl(
                      `/api/videos/${selected.video_uid}/stream?kind=${
                        selected.has_processed_video ? streamKind : "original"
                      }`
                    )}
                  />
                </div>
              </div>

              {results && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                    <div className="glass-card rounded-xl p-3">
                      <p className="text-[11px] text-slate-400">Persistent Global IDs</p>
                      <p className="text-xl font-bold text-sky-400">{results.unique_track_count}</p>
                    </div>
                    <div className="glass-card rounded-xl p-3">
                      <p className="text-[11px] text-slate-400">Reappearance Recoveries</p>
                      <p className="text-xl font-bold text-emerald-400">{results.reappearance_recoveries ?? 0}</p>
                    </div>
                    <div className="glass-card rounded-xl p-3">
                      <p className="text-[11px] text-slate-400">Total Detections</p>
                      <p className="text-xl font-bold text-violet-400">{results.total_detections}</p>
                    </div>
                    <div className="glass-card rounded-xl p-3">
                      <p className="text-[11px] text-slate-400">Processing FPS</p>
                      <p className="text-xl font-bold text-cyan-400">{results.processing_fps}</p>
                    </div>
                    <div className="glass-card rounded-xl p-3">
                      <p className="text-[11px] text-slate-400">Avg Latency</p>
                      <p className="text-xl font-bold text-amber-400">{results.avg_latency_ms} ms</p>
                    </div>
                  </div>

                  {results.tracks?.length > 0 && (
                    <div className="glass-card rounded-2xl p-4 overflow-x-auto">
                      <h4 className="text-xs font-semibold text-slate-400 mb-3">
                        Persistent Global Identities (Old ID preserved when person leaves & re-enters)
                      </h4>
                      <table className="w-full text-xs text-left">
                        <thead>
                          <tr className="border-b">
                            <th className="py-2 pr-3">Global ID</th>
                            <th className="py-2 pr-3">Local BoT-SORT Tracks</th>
                            <th className="py-2 pr-3">Frames Seen</th>
                            <th className="py-2 pr-3">Recoveries</th>
                            <th className="py-2 pr-3">Max Conf</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results.tracks.map((t) => (
                            <tr key={t.global_id || t.track_id} className="border-b">
                              <td className="py-2 pr-3 font-mono font-semibold text-sky-400">
                                {t.global_id || `ID:${t.track_id}`}
                              </td>
                              <td className="py-2 pr-3 font-mono text-slate-300">
                                {Array.isArray(t.local_track_ids) ? t.local_track_ids.join(", ") : t.track_id}
                              </td>
                              <td className="py-2 pr-3">{t.frames_seen}</td>
                              <td className="py-2 pr-3 text-emerald-400">{t.recovered_count ?? 0}</td>
                              <td className="py-2 pr-3">{t.max_confidence}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {results.timeline?.length > 0 && (
                    <div className="glass-card rounded-2xl p-4">
                      <h4 className="text-xs font-semibold text-slate-400 mb-3 flex items-center gap-1.5">
                        <Activity className="w-4 h-4 text-sky-400" />
                        Detections & Active Tracks Over Time
                      </h4>
                      <div className="h-44">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={results.timeline}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="time_sec" stroke="#94a3b8" fontSize={11} />
                            <YAxis stroke="#94a3b8" fontSize={11} />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#0f172a",
                                borderColor: "#334155",
                                color: "#f8fafc",
                              }}
                            />
                            <Area
                              type="monotone"
                              dataKey="detections"
                              stroke="#38bdf8"
                              fill="#38bdf8"
                              fillOpacity={0.2}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <div className="glass-card rounded-2xl p-12 text-center text-slate-400 text-sm">
              Upload or select a video on the left to preview and run ARGUS-X tracking analysis.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
