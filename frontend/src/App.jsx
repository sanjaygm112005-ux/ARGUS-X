import React from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import { ThemeProvider } from "./context/ThemeContext";
import { ProtectedRoute } from "./routes/ProtectedRoute";
import { PublicLayout } from "./layouts/PublicLayout";
import { DashboardLayout } from "./layouts/DashboardLayout";
import { HomePage } from "./pages/public/HomePage";
import {
  AboutPage,
  ArchitecturePage,
  ContactPage,
  FeaturesPage,
  TechnologyPage,
} from "./pages/public/PublicSubPages";
import { LoginPage } from "./pages/auth/LoginPage";
import { RegisterPage } from "./pages/auth/RegisterPage";
import { ForgotPasswordPage, ResetPasswordPage } from "./pages/auth/PasswordResetPages";
import { ProfilePage } from "./pages/auth/ProfilePage";
import { AdminOverviewPage, LiveMonitoringPage } from "./pages/dashboard/OverviewAndMonitoring";
import { CameraManagementPage, UserManagementPage } from "./pages/dashboard/CamerasAndUsers";
import {
  IdentitiesPage,
  RecoveryHistoryPage,
  UncertainReviewPage,
} from "./pages/dashboard/IdentityAndReview";
import {
  AnalyticsPage,
  AuditLogsPage,
  SettingsPage,
} from "./pages/dashboard/AdminGovernance";
import { DatasetBenchmarkPage } from "./pages/dashboard/DatasetBenchmarkPage";
import { SingleVideoAnalysisPage } from "./pages/dashboard/SingleVideoAnalysisPage";
import { MultiCameraAnalysisPage } from "./pages/dashboard/MultiCameraAnalysisPage";

export function AppRoutes() {
  return (
    <Routes>
      {/* Public Website & Auth */}
      <Route element={<PublicLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/technology" element={<TechnologyPage />} />
        <Route path="/architecture" element={<ArchitecturePage />} />
        <Route path="/features" element={<FeaturesPage />} />
        <Route path="/contact" element={<ContactPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />
        <Route path="/reset-password" element={<ResetPasswordPage />} />
      </Route>

      {/* Admin Dashboard */}
      <Route
        path="/admin"
        element={
          <ProtectedRoute allowedRoles={["ADMIN"]}>
            <DashboardLayout mode="admin" />
          </ProtectedRoute>
        }
      >
        <Route index element={<AdminOverviewPage />} />
        <Route path="cameras" element={<CameraManagementPage />} />
        <Route path="monitoring" element={<LiveMonitoringPage />} />
        <Route path="mot17" element={<DatasetBenchmarkPage defaultTab="sequences" />} />
        <Route path="benchmarks" element={<DatasetBenchmarkPage defaultTab="benchmark" />} />
        <Route path="single-video" element={<SingleVideoAnalysisPage />} />
        <Route path="multicamera" element={<MultiCameraAnalysisPage />} />
        <Route path="identities" element={<IdentitiesPage />} />
        <Route path="recovery-history" element={<RecoveryHistoryPage />} />
        <Route path="uncertain" element={<UncertainReviewPage />} />
        <Route path="conflicts" element={<IdentitiesPage conflictsOnly />} />
        <Route path="users" element={<UserManagementPage />} />
        <Route path="audit-logs" element={<AuditLogsPage />} />
        <Route path="analytics" element={<AnalyticsPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>

      {/* Operator Dashboard */}
      <Route
        path="/operator"
        element={
          <ProtectedRoute allowedRoles={["OPERATOR", "ADMIN"]}>
            <DashboardLayout mode="operator" />
          </ProtectedRoute>
        }
      >
        <Route index element={<LiveMonitoringPage />} />
        <Route path="tracking" element={<LiveMonitoringPage />} />
        <Route path="mot17" element={<DatasetBenchmarkPage defaultTab="sequences" />} />
        <Route path="benchmarks" element={<DatasetBenchmarkPage defaultTab="benchmark" />} />
        <Route path="single-video" element={<SingleVideoAnalysisPage />} />
        <Route path="multicamera" element={<MultiCameraAnalysisPage />} />
        <Route path="identities" element={<IdentitiesPage />} />
        <Route path="recovery-history" element={<RecoveryHistoryPage />} />
        <Route path="uncertain" element={<UncertainReviewPage />} />
        <Route path="review" element={<UncertainReviewPage />} />
      </Route>

      {/* Authenticated Profile */}
      <Route
        path="/profile"
        element={
          <ProtectedRoute allowedRoles={["ADMIN", "OPERATOR"]}>
            <DashboardLayout mode="operator" />
          </ProtectedRoute>
        }
      >
        <Route index element={<ProfilePage />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}
