import { GraduationCap } from "lucide-react";
import { Link, Outlet } from "react-router-dom";

export default function Layout() {
  return (
    <div className="min-h-screen bg-ink-50 print:bg-white">
      <header className="border-b border-ink-200 bg-white print:hidden">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
          <Link to="/" className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-md bg-brand-600 text-white">
              <GraduationCap className="h-5 w-5" aria-hidden />
            </span>
            <span className="font-serif text-lg font-semibold text-ink-900">SkillPulse</span>
            <span className="hidden text-sm text-ink-500 sm:inline">· Syllabus vs industry skills</span>
          </Link>
          <nav className="flex items-center gap-1">
            <Link to="/" className="rounded-md px-3 py-1.5 text-sm font-medium text-brand-700 hover:bg-brand-50">New analysis</Link>
            <Link to="/models" className="rounded-md px-3 py-1.5 text-sm font-medium text-ink-700 hover:bg-ink-100">ML models</Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <Outlet />
      </main>
    </div>
  );
}
