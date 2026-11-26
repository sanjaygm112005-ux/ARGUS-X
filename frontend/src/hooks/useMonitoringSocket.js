import { useEffect, useRef, useState } from "react";
import api from "../services/api";

export function useMonitoringSocket(enabled = true) {
  const [snapshot, setSnapshot] = useState(null);
  const [connectionState, setConnectionState] = useState("connecting");
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const pollTimerRef = useRef(null);

  useEffect(() => {
    if (!enabled) return undefined;
    let active = true;

    const loadInitial = async () => {
      try {
        const [statusRes, camsRes, idsRes] = await Promise.all([
          api.get("/api/monitoring/status"),
          api.get("/api/monitoring/cameras"),
          api.get("/api/monitoring/identities"),
        ]);
        if (!active) return;
        setSnapshot({
          status: statusRes.data.status,
          statistics: statusRes.data.statistics,
          cameras: camsRes.data.cameras,
          tracking_events: camsRes.data.tracking_events || [],
          identities: idsRes.data.identities,
          uncertain_alerts: idsRes.data.uncertain_alerts,
          timestamp: statusRes.data.timestamp,
        });
        setConnectionState("connected");
      } catch {
        // Auth might be pending or endpoints loading
      }
    };

    loadInitial();

    const connectWs = () => {
      if (!active) return;
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${proto}//${window.location.host}/ws/monitoring`;
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;
        ws.onopen = () => {
          if (active) setConnectionState("connected");
        };
        ws.onmessage = (event) => {
          if (!active) return;
          try {
            const payload = JSON.parse(event.data);
            setSnapshot((prev) => ({
              ...(prev || {}),
              ...payload,
            }));
            setConnectionState("connected");
          } catch {
            // ignore malformed tick
          }
        };
        ws.onerror = () => {
          if (active) {
            setConnectionState((prev) => (prev === "connected" ? "connected" : "connecting"));
          }
        };
        ws.onclose = () => {
          if (active) {
            // Reconnect automatically with backoff
            reconnectTimerRef.current = setTimeout(connectWs, 2000);
          }
        };
      } catch {
        if (active) {
          reconnectTimerRef.current = setTimeout(connectWs, 2000);
        }
      }
    };

    connectWs();

    // Fallback polling keeps monitoring live even if socket is momentarily reconnecting
    pollTimerRef.current = setInterval(() => {
      if (!active) return;
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        loadInitial();
      }
    }, 2500);

    return () => {
      active = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [enabled]);

  return { snapshot, connectionState };
}
