import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Minimize2,
  Move,
  X,
} from "lucide-react";
import { apiUrl } from "../services/api";

export default function ZoomableVideoFeed({ cam }) {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState(false);

  const containerRef = useRef(null);
  const modalContainerRef = useRef(null);

  // Helper to clamp scale between 1x and 5x
  const updateScale = (newScale) => {
    const clamped = Math.min(Math.max(Number(newScale.toFixed(2)), 1), 5);
    setScale(clamped);
    if (clamped === 1) {
      setPosition({ x: 0, y: 0 });
    }
  };

  const handleZoomIn = (e) => {
    e?.stopPropagation();
    updateScale(scale + 0.5);
  };

  const handleZoomOut = (e) => {
    e?.stopPropagation();
    updateScale(scale - 0.5);
  };

  const handleResetZoom = (e) => {
    e?.stopPropagation();
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  // Mouse wheel zoom
  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.25 : -0.25;
    updateScale(scale + delta);
  };

  // Drag-to-pan handlers
  const handleMouseDown = (e) => {
    if (scale <= 1) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging || scale <= 1) return;
    const maxX = (scale - 1) * 200;
    const maxY = (scale - 1) * 150;
    const newX = Math.min(Math.max(e.clientX - dragStart.x, -maxX), maxX);
    const newY = Math.min(Math.max(e.clientY - dragStart.y, -maxY), maxY);
    setPosition({ x: newX, y: newY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Double click toggles between 1x and 2.2x
  const handleDoubleClick = () => {
    if (scale > 1) {
      handleResetZoom();
    } else {
      updateScale(2.2);
    }
  };

  // Close fullscreen on ESC
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape" && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen]);

  const frameSrc = cam.frame_b64
    ? `data:image/jpeg;base64,${cam.frame_b64}`
    : cam.last_frame_timestamp
    ? apiUrl(`/api/cameras/${cam.id}/frame?t=${Math.floor(cam.last_frame_timestamp * 10)}`)
    : null;

  return (
    <>
      {/* Standard Card View */}
      <div
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onDoubleClick={handleDoubleClick}
        className={`h-56 sm:h-64 bg-slate-900 flex items-center justify-center relative overflow-hidden select-none ${
          scale > 1 ? (isDragging ? "cursor-grabbing" : "cursor-grab") : "cursor-default"
        }`}
      >
        {frameSrc ? (
          <img
            src={frameSrc}
            alt={cam.name}
            draggable={false}
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
              transition: isDragging ? "none" : "transform 0.15s ease-out",
            }}
            className="w-full h-full object-contain pointer-events-none"
          />
        ) : (
          <div className="text-center px-4 text-xs text-slate-400">
            {cam.last_error
              ? `Stream Alert: ${cam.last_error}`
              : cam.is_running
              ? "AI Engine analyzing live video frames..."
              : "Camera idle / awaiting live frames"}
          </div>
        )}

        {/* Hover / Active Controls Bar */}
        <div className="absolute bottom-2.5 right-2.5 flex items-center gap-1 bg-slate-900/80 backdrop-blur-md px-2 py-1 rounded-xl border border-slate-700/80 shadow-lg transition-opacity duration-200">
          <span className="text-[11px] font-mono text-sky-400 font-semibold px-1">
            {scale.toFixed(1)}x
          </span>

          <button
            type="button"
            onClick={handleZoomIn}
            title="Zoom In (+)"
            aria-label="Zoom in video"
            className="p-1 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white transition"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={handleZoomOut}
            title="Zoom Out (-)"
            aria-label="Zoom out video"
            disabled={scale <= 1}
            className="p-1 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white disabled:opacity-40 transition"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>

          {scale > 1 && (
            <button
              type="button"
              onClick={handleResetZoom}
              title="Reset Zoom (1.0x)"
              aria-label="Reset zoom"
              className="p-1 rounded-lg hover:bg-slate-800 text-amber-400 hover:text-amber-300 transition"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}

          <button
            type="button"
            onClick={() => setIsFullscreen(true)}
            title="Enlarge Video Inspector"
            aria-label="Inspect video in enlarged modal"
            className="p-1 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white transition"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Zoom Hint overlay when zoomed */}
        {scale > 1 && (
          <div className="absolute top-2 left-2 bg-slate-900/80 backdrop-blur-md px-2 py-0.5 rounded-md text-[10px] text-slate-300 border border-slate-700/60 pointer-events-none flex items-center gap-1">
            <Move className="w-2.5 h-2.5 text-sky-400" />
            Drag to pan • Scroll to zoom
          </div>
        )}
      </div>

      {/* Fullscreen Video Inspector Modal */}
      {isFullscreen && (
        <div className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md flex flex-col p-4 sm:p-6 animate-in fade-in duration-150">
          {/* Modal Header */}
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h3 className="text-base font-display font-bold text-white flex items-center gap-2">
                <span>{cam.name}</span>
                <span className="text-xs font-mono text-sky-400 bg-sky-950/60 px-2 py-0.5 rounded-full border border-sky-800">
                  {cam.camera_code}
                </span>
              </h3>
              <p className="text-xs text-slate-400">
                Live Video Inspector • Zoom in to inspect detected person features and tracklets
              </p>
            </div>

            <div className="flex items-center gap-3">
              {/* Zoom Slider Control */}
              <div className="flex items-center gap-2 bg-slate-900 px-3 py-1.5 rounded-xl border border-slate-800">
                <button
                  type="button"
                  onClick={handleZoomOut}
                  disabled={scale <= 1}
                  className="p-1 text-slate-400 hover:text-white disabled:opacity-30"
                >
                  <ZoomOut className="w-4 h-4" />
                </button>
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="0.1"
                  value={scale}
                  onChange={(e) => updateScale(parseFloat(e.target.value))}
                  className="w-24 sm:w-36 accent-sky-500 cursor-pointer"
                />
                <button
                  type="button"
                  onClick={handleZoomIn}
                  disabled={scale >= 5}
                  className="p-1 text-slate-400 hover:text-white disabled:opacity-30"
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
                <span className="font-mono text-xs text-sky-400 font-bold min-w-[36px] text-right">
                  {scale.toFixed(1)}x
                </span>
                {scale > 1 && (
                  <button
                    type="button"
                    onClick={handleResetZoom}
                    className="p-1 text-amber-400 hover:text-amber-300 ml-1"
                    title="Reset to 1.0x"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                )}
              </div>

              {/* Close Button */}
              <button
                type="button"
                onClick={() => setIsFullscreen(false)}
                className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Modal Main Viewing Area */}
          <div
            ref={modalContainerRef}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onDoubleClick={handleDoubleClick}
            className={`flex-1 mt-3 bg-black rounded-2xl flex items-center justify-center relative overflow-hidden select-none border border-slate-800/80 ${
              scale > 1 ? (isDragging ? "cursor-grabbing" : "cursor-grab") : "cursor-default"
            }`}
          >
            {frameSrc ? (
              <img
                src={frameSrc}
                alt={cam.name}
                draggable={false}
                style={{
                  transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
                  transition: isDragging ? "none" : "transform 0.15s ease-out",
                  maxHeight: "85vh",
                  maxWidth: "95vw",
                }}
                className="object-contain pointer-events-none"
              />
            ) : (
              <div className="text-center text-slate-400 text-sm">
                No active video frames to inspect.
              </div>
            )}

            {/* Hint Badge */}
            <div className="absolute bottom-4 left-4 bg-slate-900/80 backdrop-blur-md px-3 py-1 rounded-lg text-xs text-slate-300 border border-slate-700 flex items-center gap-2">
              <Move className="w-3.5 h-3.5 text-sky-400" />
              <span>
                {scale > 1
                  ? "Click & drag anywhere to pan around the frame • Scroll to zoom"
                  : "Scroll mouse wheel or use slider above to zoom in"}
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
