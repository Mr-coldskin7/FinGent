/**
 * Agent 展示元数据（唯一来源）。
 *
 * 键 = 注册表标准名（LLM/registry.py），与后端 agent_name / agent_votes 一致。
 * 历史数据或旧版本可能出现 `TECHNICAL_NERD`、`RiskManager` 这类写法，
 * 查找时统一按「忽略大小写与分隔符」归一化，因此都能命中同一条记录。
 */

export interface AgentMeta {
  /** 短标签：卡片、规则面板等空间有限的位置 */
  label: string;
  /** 完整名称：加载气泡等 */
  fullName: string;
  /** 强调色（边框、徽标） */
  color: string;
  /** 消息气泡的样式类（见 MessageBubble.vue 的 .agent-analysis-section.*） */
  cssClass: string;
}

const AGENTS: Record<string, AgentMeta> = {
  'technical-nerd': { label: '技术面分析', fullName: '技术面分析师', color: '#34d399', cssClass: 'technical' },
  morefit: { label: '基本面分析', fullName: '基本面分析师', color: '#60a5fa', cssClass: 'morefit' },
  'risk-manager': { label: '风控分析', fullName: '风控分析师', color: '#fbbf24', cssClass: 'riskmanager' },
  'sentiment-analyzer': { label: '舆情分析', fullName: '舆情分析师', color: '#f5a524', cssClass: 'sentimentanalyzer' },
  'tiktok-auntie': { label: '涨停大姨（短线情绪）', fullName: '涨停大姨', color: '#f472b6', cssClass: 'default' },
};

const DEFAULT_COLOR = '#6b7280';

const normalize = (name: string): string => name.toLowerCase().replace(/[^a-z0-9]/g, '');

/** 把任意写法（TECHNICAL_NERD / Morefit / technical-nerd）映射到标准名；未知名字原样返回。 */
export function canonicalAgentKey(name: string | null | undefined): string {
  if (!name) return '';
  const target = normalize(name);
  return Object.keys(AGENTS).find((key) => normalize(key) === target) ?? name;
}

/** 查找 Agent 元数据；未知 Agent 返回 undefined。 */
export function agentMeta(name: string | null | undefined): AgentMeta | undefined {
  return AGENTS[canonicalAgentKey(name)];
}

export function agentLabel(name: string | null | undefined): string {
  return agentMeta(name)?.label ?? (name || '');
}

export function agentFullName(name: string | null | undefined): string {
  return agentMeta(name)?.fullName ?? (name || '');
}

export function agentColor(name: string | null | undefined): string {
  return agentMeta(name)?.color ?? DEFAULT_COLOR;
}

export function agentCssClass(name: string | null | undefined): string {
  return agentMeta(name)?.cssClass ?? 'default';
}
