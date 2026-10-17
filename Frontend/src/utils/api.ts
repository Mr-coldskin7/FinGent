/**
 * 后端请求封装：统一 API 地址，并自动附带访问令牌（Authorization: Bearer）。
 *
 * 令牌保存在 localStorage（仅本机浏览器）。后端未配置 FINGENT_API_TOKEN 时可留空，
 * 此时仅本机回环地址可访问受保护接口；远程访问必须配置令牌。
 * 鉴权规则见后端 api_auth.py 与 docs/AUTH.md。
 */

export const API_BASE: string = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const TOKEN_KEY = 'fingent.apiToken';

export function getApiToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) || '';
  } catch {
    return '';
  }
}

export function setApiToken(token: string): void {
  const value = token.trim();
  if (value) {
    localStorage.setItem(TOKEN_KEY, value);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

/** 与 fetch 等价的调用；自动加上 Authorization 头（调用方已显式设置时不覆盖）。 */
export function apiFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers || {});
  const token = getApiToken();
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  return fetch(input, { ...init, headers });
}
