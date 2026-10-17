/**
 * 把各种底层错误（浏览器 fetch 失败、JSON 解析失败、超时）统一翻译成可直接展示给用户的中文提示。
 * 原始报错（如 "Failed to fetch"、"NetworkError"）只会让用户困惑，不直接展示。
 */
const NETWORK_RE = /failed to fetch|networkerror|network error|load failed|fetch failed|network request failed/i;

export function friendlyError(e: unknown, fallback = '请求失败，请稍后重试'): string {
  const raw = e instanceof Error ? e.message : typeof e === 'string' ? e : '';
  if (NETWORK_RE.test(raw)) {
    return '后端服务暂时无响应，请稍后重试。若持续出现，请确认后端服务已启动（8000 端口）。';
  }
  if (e instanceof SyntaxError) {
    return '服务返回了异常数据，请稍后重试。';
  }
  if (/timeout|超时|aborted/i.test(raw)) {
    return '请求超时，请稍后重试。';
  }
  return raw || fallback;
}
