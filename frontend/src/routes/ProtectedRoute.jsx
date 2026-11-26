import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export function ProtectedRoute({ children, allowedRoles }) {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#050816] text-slate-300">
        <div className="glass-card px-6 py-4 rounded-xl text-sm tracking-wide">
          Verifying ARGUS-X session...
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (allowedRoles && !allowedRoles.includes(user.role)) {
    const fallback = user.role === "ADMIN" ? "/admin" : "/operator";
    return <Navigate to={fallback} replace />;
  }

  return children;
}
