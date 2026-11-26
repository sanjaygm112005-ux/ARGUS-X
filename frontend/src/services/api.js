import axios from "axios";

function getCookie(name) {
  const match = document.cookie.match(new RegExp(`(^|; )${name}=([^;]*)`));
  return match ? decodeURIComponent(match[2]) : "";
}

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "",
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.request.use((config) => {
  if (config.data instanceof FormData) {
    delete config.headers["Content-Type"];
  }
  const method = (config.method || "get").toLowerCase();
  if (["post", "put", "patch", "delete"].includes(method)) {
    const csrf = getCookie("argus_csrf_token");
    if (csrf) {
      config.headers["X-CSRF-Token"] = csrf;
    }
  }
  return config;
});

export default api;
