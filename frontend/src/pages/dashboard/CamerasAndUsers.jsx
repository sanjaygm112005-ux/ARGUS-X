import React, { useEffect, useState } from "react";
import { Play, Plus, Square, Trash2, Upload } from "lucide-react";
import api from "../../services/api";

export function CameraManagementPage() {
  const [cameras, setCameras] = useState([]);
  const [form, setForm] = useState({ camera_code: "", name: "", source: "", location: "" });
  const [uploadForm, setUploadForm] = useState({
    camera_code: "",
    name: "",
    location: "",
    auto_start: true,
    file: null,
  });
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [error, setError] = useState("");

  const load = async () => {
    try {
      const { data } = await api.get("/api/cameras");
      setCameras(data.cameras || []);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to load cameras.");
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleCreate = async (e) => {
    e.preventDefault();
    setError("");
    try {
      await api.post("/api/cameras", {
        ...form,
        source: form.source.trim() || null,
        auto_start: false,
      });
      setForm({ camera_code: "", name: "", source: "", location: "" });
      load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to register camera.");
    }
  };

  const toggleStream = async (cam, action) => {
    setError("");
    try {
      await api.put(`/api/cameras/${cam.id}`, { action });
      load();
    } catch (err) {
      setError(err?.response?.data?.detail || `Failed to ${action} camera.`);
    }
  };

  const removeCam = async (id) => {
    await api.delete(`/api/cameras/${id}`);
    load();
  };

  const handleVideoUpload = async (e) => {
    e.preventDefault();
    if (!uploadForm.file) {
      setError("Please select a video file (.mp4, .avi, .mov, .mkv).");
      return;
    }
    setError("");
    setStatusMsg("");
    setUploading(true);
    setUploadProgress(0);
    try {
      const fd = new FormData();
      fd.append("file", uploadForm.file);
      fd.append("camera_code", uploadForm.camera_code);
      fd.append("name", uploadForm.name);
      fd.append("location", uploadForm.location);
      fd.append("auto_start", String(uploadForm.auto_start));

      await api.post("/api/cameras/upload-video", fd, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (evt) => {
          if (evt.total) {
            setUploadProgress(Math.round((evt.loaded * 100) / evt.total));
          }
        },
      });
      setStatusMsg(`Uploaded and registered camera '${uploadForm.camera_code}'.`);
      setUploadForm({ camera_code: "", name: "", location: "", auto_start: true, file: null });
      load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Video upload failed.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-2xl p-6">
        <h2 className="font-display text-lg font-bold text-white mb-2">Upload Video Stream</h2>
        <p className="text-xs text-slate-400 mb-4">
          Upload an `.mp4`, `.avi`, `.mov`, or `.mkv` video directly from your computer to run ARGUS-X detection, tracking, and Re-ID.
        </p>
        {statusMsg && <div className="mb-3 p-3 rounded-lg bg-emerald-500/15 text-emerald-300 text-xs">{statusMsg}</div>}
        {error && <div className="mb-3 p-3 rounded-lg bg-rose-500/15 text-rose-300 text-xs">{error}</div>}
        <form onSubmit={handleVideoUpload} className="grid sm:grid-cols-2 lg:grid-cols-5 gap-3 items-center">
          <input
            required
            placeholder="Camera Code (e.g. cam_1)"
            value={uploadForm.camera_code}
            onChange={(e) => setUploadForm({ ...uploadForm, camera_code: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <input
            required
            placeholder="Display Name"
            value={uploadForm.name}
            onChange={(e) => setUploadForm({ ...uploadForm, name: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <input
            placeholder="Location (optional)"
            value={uploadForm.location}
            onChange={(e) => setUploadForm({ ...uploadForm, location: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <input
            type="file"
            accept=".mp4,.avi,.mov,.mkv,video/*"
            onChange={(e) => setUploadForm({ ...uploadForm, file: e.target.files?.[0] || null })}
            className="text-xs text-slate-300 file:mr-2 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-sky-500/20 file:text-sky-300"
          />
          <button
            type="submit"
            disabled={uploading}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-sky-500 to-violet-600 text-white text-sm font-semibold px-4 py-2 disabled:opacity-60"
          >
            <Upload className="w-4 h-4" />
            {uploading ? `Uploading ${uploadProgress}%` : "Upload & Start"}
          </button>
        </form>
      </div>

      <div className="glass-card rounded-2xl p-6">
        <h2 className="font-display text-lg font-bold text-white mb-4">Register Camera by Path / RTSP</h2>
        <form onSubmit={handleCreate} className="grid sm:grid-cols-2 lg:grid-cols-5 gap-3">
          <input
            required
            placeholder="Camera Code (e.g. cam_1)"
            value={form.camera_code}
            onChange={(e) => setForm({ ...form, camera_code: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <input
            required
            placeholder="Display Name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <input
            placeholder="Video Path or RTSP URL"
            value={form.source}
            onChange={(e) => setForm({ ...form, source: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <input
            placeholder="Location (e.g. Lobby)"
            value={form.location}
            onChange={(e) => setForm({ ...form, location: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <button
            type="submit"
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-sky-500 text-white text-sm font-semibold px-4 py-2"
          >
            <Plus className="w-4 h-4" /> Add Camera
          </button>
        </form>
      </div>

      <div className="glass-card rounded-2xl p-6 overflow-x-auto">
        <h3 className="font-display text-lg font-semibold text-white mb-4">Registered Cameras</h3>
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-slate-400 border-b border-slate-800">
            <tr>
              <th className="py-2 pr-4">Code</th>
              <th className="py-2 pr-4">Name</th>
              <th className="py-2 pr-4">Location</th>
              <th className="py-2 pr-4">Status</th>
              <th className="py-2 pr-4">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {cameras.map((c) => (
              <tr key={c.id}>
                <td className="py-3 pr-4 font-mono text-sky-300">{c.camera_code}</td>
                <td className="py-3 pr-4 text-white">{c.name}</td>
                <td className="py-3 pr-4 text-slate-400">{c.location || "—"}</td>
                <td className="py-3 pr-4">{c.status}</td>
                <td className="py-3 pr-4 flex items-center gap-2">
                  {c.is_running ? (
                    <button
                      type="button"
                      onClick={() => toggleStream(c, "stop")}
                      className="p-1.5 rounded bg-amber-500/20 text-amber-300"
                      title="Stop Stream"
                    >
                      <Square className="w-4 h-4" />
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => toggleStream(c, "start")}
                      className="p-1.5 rounded bg-emerald-500/20 text-emerald-300"
                      title="Start Stream"
                    >
                      <Play className="w-4 h-4" />
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => removeCam(c.id)}
                    className="p-1.5 rounded bg-rose-500/20 text-rose-300"
                    title="Delete Camera"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function UserManagementPage() {
  const [users, setUsers] = useState([]);
  const [form, setForm] = useState({
    email: "",
    username: "",
    full_name: "",
    password: "",
    role: "OPERATOR",
  });
  const [error, setError] = useState("");

  const load = async () => {
    try {
      const { data } = await api.get("/api/users");
      setUsers(data.users || []);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to load users.");
    }
  };

  useEffect(() => {
    load();
  }, []);

  const createUser = async (e) => {
    e.preventDefault();
    setError("");
    try {
      await api.post("/api/users", form);
      setForm({ email: "", username: "", full_name: "", password: "", role: "OPERATOR" });
      load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to create user.");
    }
  };

  const toggleActive = async (u) => {
    setError("");
    try {
      await api.put(`/api/users/${u.id}`, { is_active: !u.is_active });
      load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Update failed.");
    }
  };

  const deleteUser = async (id) => {
    setError("");
    try {
      await api.delete(`/api/users/${id}`);
      load();
    } catch (err) {
      setError(err?.response?.data?.detail || "Delete failed.");
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-2xl p-6">
        <h2 className="font-display text-lg font-bold text-white mb-4">Create User</h2>
        {error && <div className="mb-3 p-3 rounded-lg bg-rose-500/15 text-rose-300 text-xs">{error}</div>}
        <form onSubmit={createUser} className="grid sm:grid-cols-2 lg:grid-cols-6 gap-3">
          <input
            required
            type="email"
            placeholder="Email"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <input
            required
            placeholder="Username"
            value={form.username}
            onChange={(e) => setForm({ ...form, username: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <input
            placeholder="Full Name"
            value={form.full_name}
            onChange={(e) => setForm({ ...form, full_name: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <input
            required
            type="password"
            minLength={8}
            placeholder="Password (8+ chars)"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          />
          <select
            value={form.role}
            onChange={(e) => setForm({ ...form, role: e.target.value })}
            className="rounded-lg bg-[#080f26] border border-slate-700 px-3 py-2 text-sm text-white"
          >
            <option value="OPERATOR">OPERATOR</option>
            <option value="ADMIN">ADMIN</option>
          </select>
          <button
            type="submit"
            className="rounded-lg bg-sky-500 text-white text-sm font-semibold px-4 py-2"
          >
            Create
          </button>
        </form>
      </div>

      <div className="glass-card rounded-2xl p-6 overflow-x-auto">
        <h3 className="font-display text-lg font-semibold text-white mb-4">Users</h3>
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-slate-400 border-b border-slate-800">
            <tr>
              <th className="py-2 pr-4">Username</th>
              <th className="py-2 pr-4">Email</th>
              <th className="py-2 pr-4">Role</th>
              <th className="py-2 pr-4">Status</th>
              <th className="py-2 pr-4">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {users.map((u) => (
              <tr key={u.id}>
                <td className="py-3 pr-4 text-white font-medium">{u.username}</td>
                <td className="py-3 pr-4 text-slate-400">{u.email}</td>
                <td className="py-3 pr-4 text-violet-300 font-mono text-xs">{u.role}</td>
                <td className="py-3 pr-4">
                  <button
                    type="button"
                    onClick={() => toggleActive(u)}
                    className={`px-2 py-0.5 rounded text-xs ${
                      u.is_active ? "bg-emerald-500/20 text-emerald-300" : "bg-rose-500/20 text-rose-300"
                    }`}
                  >
                    {u.is_active ? "ACTIVE" : "DISABLED"}
                  </button>
                </td>
                <td className="py-3 pr-4">
                  <button
                    type="button"
                    onClick={() => deleteUser(u.id)}
                    className="p-1.5 rounded bg-rose-500/20 text-rose-300"
                    title="Delete User"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
