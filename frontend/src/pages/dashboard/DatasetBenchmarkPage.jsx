import React, { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  Film,
  Loader2,
  Play,
  RefreshCw,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import api from "../../services/api";

export function DatasetBenchmarkPage({ defaultTab = "sequences" }) {
  const [tab, setTab] = useState(defaultTab);
  const [mot17, setMot17] = useState(null);
  const [detectorFilter, setDetectorFilter] = useState("FRCNN");
  const [selectedSeq, setSelectedSeq] = useState("");
  const [maxFrames, setMaxFrames] = useState(40);
  const [jobs, setJobs] = useState([]);
  const [activeJob, setActiveJob] = useState(null);
  const [activeResult, setActiveResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");

  const loadDataset = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [dsRes, jobsRes] = await Promise.all([
        api.get(`/api/datasets/mot17?detector=${detectorFilter}&split=train`),
        api.get("/api/benchmark"),
      ]);
      setMot17(dsRes.data);
      const seqs = dsRes.data?.sequences || [];
      if (seqs.length > 0 && !seqs.some((s) => s.name === selectedSeq)) {
        const firstValid = seqs.find((s) => s.is_valid) || seqs[0];
        setSelectedSeq(firstValid.name);
      }
      const jList = jobsRes.data?.jobs || [];
      setJobs(jList);
      if (jList.length > 0 && !activeJob) {
        setActiveJob(jList[0]);
        if (jList[0].result) setActiveResult(jList[0].result);
      }
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to load MOT17 dataset.");
    } finally {
      setLoading(false);
    }
  }, [detectorFilter]);

  useEffect(() => {
    loadDataset();
  }, [loadDataset]);

  // Poll active benchmark job while QUEUED or RUNNING
  useEffect(() => {
    if (!activeJob || !["QUEUED", "RUNNING"].includes(activeJob.status)) return;
    const timer = setInterval(async () => {
      try {
        const { data } = await api.get(`/api/benchmark/${activeJob.job_uid}/results`);
        if (data.job) {
          setActiveJob(data.job);
          setJobs((prev) =>
            prev.map((j) => (j.job_uid === data.job.job_uid ? { ...data.job, result: data.result } : j))
          );
        }
        if (data.result) {
          setActiveResult(data.result);
        }
      } catch {
        // ignore transient poll error
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [activeJob]);

  const handleStartBenchmark = async () => {
    if (!selectedSeq) return;
    setStarting(true);
    setError("");
    setActiveResult(null);
    try {
      const { data } = await api.post("/api/benchmark/mot17", {
        sequence_name: selectedSeq,
        max_frames: Number(maxFrames) || 40,
      });
      setActiveJob(data.job);
      setJobs((prev) => [data.job, ...prev]);
      setTab("benchmark");
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to start MOT17 benchmark.");
    } finally {
      setStarting(false);
    }
  };

  const selectedSeqObj = (mot17?.sequences || []).find((s) => s.name === selectedSeq);

  const metricChartData = activeResult
    ? [
        { metric: "MOTA %", value: activeResult.mota },
        { metric: "IDF1 %", value: activeResult.idf1 },
        { metric: "HOTA %", value: activeResult.hota },
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="font-display text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Database className="w-6 h-6 text-sky-400" />
            MOT17 Dataset & TrackEval Benchmarking
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Auto-discovered sequences from <code className="text-sky-400">datasets/MOT17/</code> evaluated with YOLO11n + BoT-SORT.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setTab("sequences")}
            className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition ${
              tab === "sequences"
                ? "bg-gradient-to-r from-blue-700 to-indigo-700 text-white"
                : "glass-card text-slate-600 dark:text-slate-300"
            }`}
          >
            MOT17 Sequences
          </button>
          <button
            type="button"
            onClick={() => setTab("benchmark")}
            className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition ${
              tab === "benchmark"
                ? "bg-gradient-to-r from-blue-700 to-indigo-700 text-white"
                : "glass-card text-slate-600 dark:text-slate-300"
            }`}
          >
            Benchmark & Results
          </button>
          <button
            type="button"
            onClick={loadDataset}
            className="p-2 rounded-xl glass-card text-slate-500 dark:text-slate-300 hover:text-sky-400"
            title="Rescan MOT17"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/15 border border-rose-500/40 text-rose-300 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {mot17?.status === "MISSING" && (
        <div className="glass-card rounded-2xl p-6 border-amber-500/40 bg-amber-500/10">
          <h3 className="font-semibold text-amber-300 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            MOT17 Dataset Not Found
          </h3>
          <p className="mt-2 text-sm text-slate-300">{mot17.setup_message}</p>
        </div>
      )}

      {tab === "sequences" && mot17?.status === "AVAILABLE" && (
        <div className="space-y-6">
          {/* Controls */}
          <div className="glass-card rounded-2xl p-5 flex flex-wrap items-end justify-between gap-4">
            <div className="flex flex-wrap items-center gap-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Detector Split</label>
                <select
                  value={detectorFilter}
                  onChange={(e) => setDetectorFilter(e.target.value)}
                  className="rounded-xl px-3 py-2 text-sm border"
                >
                  <option value="FRCNN">FRCNN (Recommended)</option>
                  <option value="SDP">SDP</option>
                  <option value="DPM">DPM</option>
                  <option value="ALL">All Detectors</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Max Frames to Benchmark</label>
                <input
                  type="number"
                  min={5}
                  max={2000}
                  value={maxFrames}
                  onChange={(e) => setMaxFrames(e.target.value)}
                  className="w-32 rounded-xl px-3 py-2 text-sm border"
                />
              </div>
            </div>

            <button
              type="button"
              disabled={!selectedSeqObj?.is_valid || starting}
              onClick={handleStartBenchmark}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-700 to-indigo-700 text-white text-sm font-semibold hover:opacity-95 disabled:opacity-50"
            >
              {starting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Run Benchmark on {selectedSeq || "Sequence"}
            </button>
          </div>

          {/* Sequence Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {(mot17.sequences || []).map((seq) => {
              const isSelected = seq.name === selectedSeq;
              return (
                <button
                  type="button"
                  key={seq.name}
                  onClick={() => setSelectedSeq(seq.name)}
                  className={`text-left glass-card rounded-2xl p-5 transition border ${
                    isSelected
                      ? "border-sky-400 ring-2 ring-sky-500/30"
                      : "border-slate-200 dark:border-slate-800"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-display font-bold text-slate-900 dark:text-white flex items-center gap-2">
                      <Film className="w-4 h-4 text-sky-400" />
                      {seq.name}
                    </span>
                    {seq.is_valid ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/15 text-emerald-400">
                        <CheckCircle2 className="w-3 h-3" /> Valid
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded-full text-[11px] bg-rose-500/15 text-rose-400">
                        Invalid
                      </span>
                    )}
                  </div>

                  <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-500 dark:text-slate-400">
                    <div>
                      Resolution:{" "}
                      <span className="text-slate-800 dark:text-slate-200 font-medium">
                        {seq.im_width}×{seq.im_height}
                      </span>
                    </div>
                    <div>
                      Frames:{" "}
                      <span className="text-slate-800 dark:text-slate-200 font-medium">
                        {seq.actual_frame_count} ({seq.frame_rate} FPS)
                      </span>
                    </div>
                    <div>
                      GT Pedestrians:{" "}
                      <span className="text-sky-400 font-medium">
                        {seq.gt_summary?.unique_gt_ids ?? 0} IDs
                      </span>
                    </div>
                    <div>
                      GT Boxes:{" "}
                      <span className="text-violet-400 font-medium">
                        {seq.gt_summary?.pedestrian_rows ?? 0}
                      </span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {tab === "benchmark" && (
        <div className="space-y-6">
          {/* Active Job Progress */}
          {activeJob && (
            <div className="glass-card rounded-2xl p-6 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="font-display font-semibold text-lg text-slate-900 dark:text-white">
                    Job {activeJob.job_uid} — {activeJob.sequence_name}
                  </h3>
                  <p className="text-xs text-slate-400">
                    Frames: {activeJob.processed_frames} / {activeJob.total_frames}
                  </p>
                </div>
                <span
                  className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    activeJob.status === "COMPLETED"
                      ? "bg-emerald-500/20 text-emerald-300"
                      : activeJob.status === "FAILED"
                      ? "bg-rose-500/20 text-rose-300"
                      : "bg-sky-500/20 text-sky-300"
                  }`}
                >
                  {activeJob.status} ({activeJob.progress}%)
                </span>
              </div>

              <div className="w-full h-2.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-sky-500 to-violet-600 transition-all duration-300"
                  style={{ width: `${activeJob.progress || 0}%` }}
                />
              </div>

              {activeJob.error_message && (
                <p className="text-xs text-rose-400">{activeJob.error_message}</p>
              )}
            </div>
          )}

          {/* Results Cards + Chart */}
          {activeResult && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[
                  { label: "MOTA", val: `${activeResult.mota}%`, color: "text-sky-400" },
                  { label: "IDF1", val: `${activeResult.idf1}%`, color: "text-violet-400" },
                  { label: "HOTA", val: `${activeResult.hota}%`, color: "text-emerald-400" },
                  { label: "FPS", val: activeResult.fps, color: "text-amber-400" },
                  { label: "ID Switches", val: activeResult.id_switches, color: "text-rose-400" },
                  { label: "False Positives", val: activeResult.false_positives, color: "text-slate-300" },
                  { label: "False Negatives", val: activeResult.false_negatives, color: "text-slate-300" },
                  {
                    label: "Latency",
                    val: `${activeResult.processing_latency_ms} ms`,
                    color: "text-cyan-400",
                  },
                ].map((m) => (
                  <div key={m.label} className="glass-card rounded-2xl p-4">
                    <p className="text-xs text-slate-400">{m.label}</p>
                    <p className={`mt-1 text-2xl font-display font-bold ${m.color}`}>{m.val}</p>
                  </div>
                ))}
              </div>

              <div className="glass-card rounded-2xl p-4">
                <h4 className="text-xs font-semibold text-slate-400 mb-3 flex items-center gap-1.5">
                  <BarChart3 className="w-4 h-4 text-sky-400" />
                  TrackEval Metrics (%)
                </h4>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={metricChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="metric" stroke="#94a3b8" fontSize={11} />
                      <YAxis domain={[0, 100]} stroke="#94a3b8" fontSize={11} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0f172a",
                          borderColor: "#334155",
                          color: "#f8fafc",
                        }}
                      />
                      <Bar dataKey="value" fill="#38bdf8" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* Job History */}
          <div className="glass-card rounded-2xl p-5">
            <h3 className="font-semibold text-sm text-slate-900 dark:text-white mb-3">
              Recent MOT17 Benchmark Runs (Persisted in Neon)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b">
                    <th className="py-2 pr-3">Job</th>
                    <th className="py-2 pr-3">Sequence</th>
                    <th className="py-2 pr-3">Status</th>
                    <th className="py-2 pr-3">MOTA</th>
                    <th className="py-2 pr-3">IDF1</th>
                    <th className="py-2 pr-3">HOTA</th>
                    <th className="py-2 pr-3">IDSW</th>
                    <th className="py-2 pr-3">FPS</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((j) => (
                    <tr
                      key={j.job_uid}
                      onClick={() => {
                        setActiveJob(j);
                        setActiveResult(j.result || null);
                      }}
                      className="border-b cursor-pointer hover:bg-slate-800/30"
                    >
                      <td className="py-2 pr-3 font-mono text-sky-400">{j.job_uid}</td>
                      <td className="py-2 pr-3 font-medium">{j.sequence_name}</td>
                      <td className="py-2 pr-3">{j.status}</td>
                      <td className="py-2 pr-3">{j.result ? `${j.result.mota}%` : "—"}</td>
                      <td className="py-2 pr-3">{j.result ? `${j.result.idf1}%` : "—"}</td>
                      <td className="py-2 pr-3">{j.result ? `${j.result.hota}%` : "—"}</td>
                      <td className="py-2 pr-3">{j.result ? j.result.id_switches : "—"}</td>
                      <td className="py-2 pr-3">{j.result ? j.result.fps : "—"}</td>
                    </tr>
                  ))}
                  {jobs.length === 0 && (
                    <tr>
                      <td colSpan={8} className="py-4 text-center text-slate-400">
                        No benchmark runs yet. Select a MOT17 sequence and click Run Benchmark.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
