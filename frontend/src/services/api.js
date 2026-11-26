import axios from "axios";

function getCookie(name) {
  const match = document.cookie.match(new RegExp(`(^|; )${name}=([^;]*)`));
  return match ? decodeURIComponent(match[2]) : "";
}

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");

export function apiUrl(path) {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalized}`;
}

export function wsUrl(path) {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (API_BASE_URL) {
    const wsBase = API_BASE_URL.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
    return `${wsBase}${normalized}`;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${normalized}`;
}

const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

const TOKEN_STORAGE_KEY = "argus_access_token";

export function setStoredAccessToken(token) {
  if (token) {
    localStorage.setItem(TOKEN_STORAGE_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  }
}

export function getStoredAccessToken() {
  return localStorage.getItem(TOKEN_STORAGE_KEY) || "";
}

api.interceptors.request.use((config) => {
  if (config.data instanceof FormData) {
    delete config.headers["Content-Type"];
  }
  const bearer = getStoredAccessToken();
  if (bearer && !config.headers["Authorization"]) {
    config.headers["Authorization"] = `Bearer ${bearer}`;
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
