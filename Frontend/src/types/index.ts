export type VoteType = 'BUY' | 'SELL' | 'HOLD' | 'STRONG_BUY' | 'STRONG_SELL';

export interface ToolCall {
  type: 'input' | 'tool_call' | 'tool_result';
  name?: string;
  args?: Record<string, any>;
  content?: string;
}

export interface AgentToolChain {
  agent: string;
  steps: ToolCall[];
}

export interface Decision {
  symbol: string;
  vote: VoteType;
  reason: string;
  target_position_pct?: number;
  confidence?: number;
  /** 部分响应的 decision 会携带来源 agent 名（缺失时由 allToolChains 顺序兜底） */
  agent?: string;
}

export interface AnalysisData {
  decisions?: Decision[];
  portfolio_suggestion?: string;
  answer?: string;
  // 双Agent模式的最终决策
  final_decision?: {
    vote: VoteType;
    confidence: number;
    target_position_pct: number;
    reason: string;
  };
}

export interface FinalDecision {
  symbol: string;
  final_vote: VoteType;
  target_position_pct: number;
  confidence: number;
  suggestion: string;
  details?: {
    morefit?: { vote: VoteType; target_position_pct: number; reason: string };
    technical?: { vote: VoteType; target_position_pct: number; reason: string };
  };
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  data?: AnalysisData;
  isClarification?: boolean;
  timestamp?: Date;
  raw?: any;
  // 工具调用链相关 - 单Agent模式
  toolChain?: ToolCall[];
  agentName?: string;
  stock?: string;
  // 工具调用链相关 - 双Agent模式
  allToolChains?: AgentToolChain[];
  finalDecision?: FinalDecision;
}

export interface ClarificationInfo {
  issue_type?: string;
  message?: string;
  options?: string[];
}

export interface ParseResult {
  status: 'ready' | 'clarification_needed';
  intent?: { type: string };
  entities?: {
    symbols?: string[];
    names?: string[];
    code?: string[];
  };
  time_range?: { start: string; end: string };
  original_input?: string;
  clarification?: ClarificationInfo | string;
}

/** 路由解释（T2）：本次为什么选这些 agent，来源显式/语义/兜底 */
export interface RouteInfo {
  mode: 'single' | 'multi' | 'chat' | 'clarify' | 'sector';
  agents: string[];
  reason: string;
  source: 'explicit' | 'semantic' | 'fallback' | 'chat' | 'clarify' | 'sector' | 'none';
}

export interface ChatResponse {
  success: boolean;
  thread_id: string;
  input?: string;
  parse_result?: ParseResult;
  result?: string;
  status?: string;
  error?: string;
  
  // 单Agent模式工具调用链
  tool_chain?: ToolCall[];
  agent_name?: string;
  stock?: string;
  
  // 路由解释（T2）
  route_info?: RouteInfo;
  selected_agents?: string[];
  
  // 双Agent模式
  all_tool_chains?: AgentToolChain[];
  final_decision?: FinalDecision;
  detailed_analysis?: AnalysisData;  // 包含两个Agent的详细决策
  
  // 其他可能字段
  data?: AnalysisData;
  needs_clarification?: boolean;
  message?: string;
  answer?: string;
}

// ===== 可靠性评测（Eval/reliability.py 产物，/api/v1/eval/runs*）=====
export interface EvalCaseSummary {
  case_id: string;
  runs: number;
  pipeline_ok: number;
  pipeline_rate: number;
  pipeline_ci95: [number, number];
  strict_ok: number;
  strict_rate: number;
  strict_ci95: [number, number];
  latency_p50_s: number | null;
  latency_p95_s: number | null;
  latency_mean_s: number | null;
  degraded_run_rate: number | null;
  routes: Record<string, number>;
  final_votes: Record<string, number>;
}

export interface EvalRunRecord {
  run_id: string;
  case_id: string;
  run_idx: number;
  query: string;
  latency_s: number;
  status: string | null;
  result_preview: string;
  eval: {
    pipeline_ok: boolean;
    strict_ok: boolean;
    route: string;
    route_ok: boolean;
    degraded_agents: string[];
    votes_valid: boolean;
    final_vote: string | null;
    reasons: string[];
  };
}

export interface EvalReport {
  generated_at: string;
  k: number;
  total_runs: number;
  cases: EvalCaseSummary[];
  runs: EvalRunRecord[];
  cases_file?: string;
}

export interface EvalRunSummary {
  run_id: string;
  generated_at: string | null;
  k: number | null;
  total_runs: number;
  case_count: number;
  pipeline_ok: number;
  strict_ok: number;
  cases: string[];
}

export interface EvalTraceEvent {
  t: number;
  stage: string;
  agent?: string;
  vote?: string;
  error?: string;
  mode?: string;
  reason?: string;
  source?: string;
}

export interface EvalTraceRun {
  run_id: string;
  case_id: string;
  run_idx: number | null;
  events: EvalTraceEvent[];
  complete: boolean;
  latency_s: number | null;
  status: string | null;
  query: string | null;
  route: string | null;
  strict_ok: boolean | null;
  pipeline_ok: boolean | null;
  final_vote: string | null;
  reasons: string[];
}

// ===== 结构化运行日志（Eval/runs/<id>/run_log.jsonl，/api/v1/eval/runs/{id}/log）=====
export type EvalLogLevel = 'INFO' | 'WARN' | 'ERROR';

export interface EvalLogAgentSummary {
  vote: string | null;
  error: boolean;
  parse_fallback: boolean;
}

export interface EvalLogEntry {
  ts: string;
  level: EvalLogLevel;
  event: 'suite_start' | 'run_start' | 'run_end' | 'suite_end';
  run_id?: string;
  case_id?: string;
  run_idx?: number;
  latency_s?: number;
  status?: string | null;
  route?: string;
  pipeline_ok?: boolean;
  strict_ok?: boolean;
  final_vote?: string | null;
  reasons?: string[];
  agents?: Record<string, EvalLogAgentSummary>;
  [key: string]: unknown;
}

export interface EvalLogResponse {
  success: boolean;
  run_id: string;
  available: boolean;
  count: number;
  skipped_lines: number;
  rows: EvalLogEntry[];
}

// 规则管理（/api/v1/memory/rules）
export interface MemoryRule {
  id: number;
  agent_name: string;  // 标准名（如 morefit）或 ALL
  rule_text: string;
  trigger_count: number;
  source: string;      // explicit_feedback | self_mined
  active: boolean;
  created_at: string;
}
