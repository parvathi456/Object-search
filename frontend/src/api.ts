
const API_URL =
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export async function apiRequest<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, options);

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;

    throw new Error(
      typeof detail === "string"
        ? detail
        : detail
          ? JSON.stringify(detail)
          : `Request failed (${response.status})`,
    );
  }

  return response.json() as Promise<T>;
}

export { API_URL };
