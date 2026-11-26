import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { AppRoutes } from "../App";
import { AuthProvider } from "../context/AuthContext";
import api from "../services/api";

vi.mock("../services/api", () => ({
  API_BASE_URL: "",
  apiUrl: (path) => path,
  wsUrl: (path) => `ws://localhost${path}`,
  setStoredAccessToken: vi.fn(),
  getStoredAccessToken: () => "",
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}));

describe("ARGUS-X Frontend", () => {
  it("renders the public hero and pending real-world evaluation status", async () => {
    api.get.mockRejectedValueOnce(new Error("unauthenticated"));
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>
    );

    expect(screen.getByText(/TRACK THE UNTRACKABLE\./i)).toBeInTheDocument();
    expect(screen.getByText(/Real-World Annotated Validation: Pending/i)).toBeInTheDocument();
  });

  it("redirects unauthenticated visitors from /admin to /login", async () => {
    api.get.mockRejectedValueOnce(new Error("unauthenticated"));
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/admin"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/Sign in to access ARGUS-X Command Console/i)).toBeInTheDocument();
    });
  });

  it("supports show/hide password and submits login credentials", async () => {
    api.get.mockRejectedValueOnce(new Error("unauthenticated"));
    api.post.mockResolvedValueOnce({
      data: { user: { id: 1, username: "admin", role: "ADMIN" } },
    });

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/login"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/admin or operator/i)).toBeInTheDocument();
    });

    const pwdInput = screen.getByPlaceholderText(/Enter your password/i);
    expect(pwdInput.type).toBe("password");
    fireEvent.click(screen.getByRole("button", { name: /Show password/i }));
    expect(pwdInput.type).toBe("text");

    fireEvent.change(screen.getByPlaceholderText(/admin or operator/i), {
      target: { value: "admin" },
    });
    fireEvent.change(pwdInput, { target: { value: "ChangeMeNow!123" } });
    fireEvent.click(screen.getByRole("button", { name: /^Login$/i }));

    await waitFor(() => {
      expect(api.post).toHaveBeenCalledWith("/api/auth/login", {
        identifier: "admin",
        password: "ChangeMeNow!123",
        remember_me: false,
      });
    });
  });
});
