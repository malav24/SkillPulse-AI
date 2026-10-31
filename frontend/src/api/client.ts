/** Small fetch wrapper that normalises API errors. */
const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(method: string, path: string, form?: FormData): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { method, body: form });
  } catch {
    throw new ApiError(0, "Cannot reach the SkillPulse server. Is the backend running on port 8000?");
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    const msg = (data as { error?: { message?: string } } | null)?.error?.message;
    throw new ApiError(res.status, msg ?? `Request failed (${res.status})`);
  }
  return data as T;
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, form?: FormData) => request<T>("POST", path, form),
  del: (path: string) => request<void>("DELETE", path),
};

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : "Something went wrong";
}
