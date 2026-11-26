import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import api, { setStoredAccessToken } from "../services/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const refreshUser = useCallback(async () => {
    try {
      const { data } = await api.get("/api/auth/me");
      setUser(data.user);
    } catch {
      setStoredAccessToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const login = async (identifier, password, rememberMe = false) => {
    const { data } = await api.post("/api/auth/login", {
      identifier,
      password,
      remember_me: rememberMe,
    });
    if (data?.access_token) {
      setStoredAccessToken(data.access_token);
    }
    setUser(data.user);
    return data.user;
  };

  const register = async (payload) => {
    const { data } = await api.post("/api/auth/register", payload);
    return data.user;
  };

  const logout = async () => {
    try {
      await api.post("/api/auth/logout");
    } finally {
      setStoredAccessToken(null);
      setUser(null);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        isAuthenticated: Boolean(user),
        isAdmin: user?.role === "ADMIN",
        isOperator: user?.role === "OPERATOR",
        login,
        register,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return ctx;
}
