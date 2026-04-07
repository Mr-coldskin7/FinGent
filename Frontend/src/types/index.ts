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
