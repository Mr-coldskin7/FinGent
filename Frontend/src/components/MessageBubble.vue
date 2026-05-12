<script setup lang="ts">
import type { Message } from '../types/index';
import DecisionCard from './DecisionCard.vue';
import PortfolioSuggestion from './PortfolioSuggestion.vue';
import ToolChain from './ToolChain.vue';
import MarkdownRender from './MarkdownRender.vue';

interface Props {
  message: Message;
  index: number;
}

defineProps<Props>();

// 判断是否为双Agent模式的Markdown内容
function isDualAgentMarkdown(content: string): boolean {
  return content.includes('## 📊 双Agent综合分析') ||
         content.includes('## 双Agent投票结果');
}

// Agent 配置
const AGENT_CONFIG: Record<string, { name: string; icon: string; class: string; color: string }> = {
  'Morefit': { name: 'Morefit 基本面', icon: '📊', class: 'morefit', color: '#2563eb' },
  'TECHNICAL_NERD': { name: 'Technical_Nerd 技术面', icon: '📈', class: 'technical', color: '#059669' },
  'RiskManager': { name: 'RiskManager 风控', icon: '🛡️', class: 'riskmanager', color: '#d97706' },
  'SentimentAnalyzer': { name: 'SentimentAnalyzer 舆情', icon: '💬', class: 'sentimentanalyzer', color: '#7c3aed' },
};

function getAgentClass(agentName: string): string {
  return AGENT_CONFIG[agentName]?.class || 'default';
}

function getAgentIcon(agentName: string): string {
  return AGENT_CONFIG[agentName]?.icon || '🤖';
}

function getAgentDisplayName(agentName: string): string {
  return AGENT_CONFIG[agentName]?.name || agentName;
}

function formatVote(vote: string): string {
  const map: Record<string, string> = {
    'STRONG_BUY': '强烈买入',
    'BUY': '买入',
    'HOLD': '持有',
    'SELL': '卖出',
    'STRONG_SELL': '强烈卖出',
    'REDUCE': '减仓',
  };
  return map[vote] || vote;
}
</script>

<template>
  <!-- User Message -->
  <div v-if="message.role === 'user'" class="user-message" :style="{ animationDelay: `${index * 0.05}s` }">
    <div class="user-content">
      <div class="user-bubble">
        <p class="text">{{ message.content }}</p>
      </div>
    </div>
    <div class="user-avatar">👤</div>
  </div>

  <!-- AI Message -->
  <div v-else class="ai-message" :style="{ animationDelay: `${index * 0.05}s` }">
    <div class="ai-avatar">🤖</div>
    
    <div class="content">
      <!-- Debug: 显示工具链状态 -->
      <div v-if="message.toolChain !== undefined || message.allToolChains !== undefined" class="debug-info">
        <small>ToolChain: {{ message.toolChain?.length || 0 }} items | 
                AllToolChains: {{ message.allToolChains?.length || 0 }} items</small>
      </div>

      <!-- Tool Chain - 单Agent或双Agent模式 -->
      <ToolChain 
        v-if="message.toolChain?.length || message.allToolChains?.length" 
        :tool-chain="message.toolChain"
        :agent-name="message.agentName"
        :stock="message.stock"
        :all-tool-chains="message.allToolChains"
      />

      <!-- Clarification Box -->
      <div v-if="message.isClarification" class="clarification-box">
        <div class="clarification-header">
          <span class="icon">💬</span>
          <span class="title">需要更多信息</span>
        </div>
        <p class="text">{{ message.content }}</p>
      </div>

      <!-- 多Agent模式：美化展示 -->
      <template v-if="message.data?.decisions?.length && message.allToolChains?.length">
        <!-- 分析标题 -->
        <div class="analysis-title-bar">
          <span class="analysis-icon">🔬</span>
          <span class="analysis-title">{{ message.data.decisions.length }}Agent 深度分析</span>
        </div>

        <!-- 动态渲染各 Agent 分析 -->
        <div
          v-for="(decision, idx) in message.data.decisions"
          :key="idx"
          class="agent-analysis-section"
          :class="getAgentClass(decision.agent || message.allToolChains?.[idx]?.agent || 'default')"
        >
          <div class="agent-section-header">
            <div class="agent-badge">
              <span class="agent-icon">{{ getAgentIcon(decision.agent || message.allToolChains?.[idx]?.agent || '') }}</span>
              <span class="agent-name">{{ getAgentDisplayName(decision.agent || message.allToolChains?.[idx]?.agent || '') }}</span>
            </div>
            <div class="vote-tag" :class="decision.vote">
              {{ formatVote(decision.vote) }}
            </div>
          </div>
          <div class="agent-content">
            <div class="metrics-row">
              <div class="metric">
                <span class="metric-label">建议仓位</span>
                <span class="metric-value">{{ ((decision.target_position_pct || 0) * 100).toFixed(0) }}%</span>
              </div>
              <div class="metric">
                <span class="metric-label">置信度</span>
                <span class="metric-value">{{ ((decision.confidence || 0) * 100).toFixed(0) }}%</span>
              </div>
            </div>
            <div class="reason-box">
              <p class="reason-text">{{ decision.reason }}</p>
            </div>
          </div>
        </div>

        <!-- 最终投票结果 -->
        <div v-if="message.data.final_decision" class="final-vote-section">
          <div class="final-vote-header">
            <span class="icon">🎯</span>
            <span class="title">综合投票结果</span>
          </div>
          <div class="final-vote-card" :class="message.data.final_decision.vote">
            <div class="final-vote-main">
              <div class="final-vote-label">最终建议</div>
              <div class="final-vote-value">
                {{ formatVote(message.data.final_decision.vote) }}
              </div>
            </div>
            <div class="final-metrics">
              <div class="final-metric">
                <span class="label">置信度</span>
                <span class="value">{{ (message.data.final_decision.confidence * 100).toFixed(0) }}%</span>
              </div>
              <div class="final-metric">
                <span class="label">建议仓位</span>
                <span class="value">{{ (message.data.final_decision.target_position_pct * 100).toFixed(0) }}%</span>
              </div>
            </div>
            <p class="final-reason">{{ message.data.final_decision.reason }}</p>
          </div>
        </div>
      </template>

      <!-- 单Agent模式：决策卡片 -->
      <div v-else-if="message.data?.decisions?.length" class="decisions-container">
        <div class="analysis-header">
          <span class="analysis-icon">📊</span>
          <span class="analysis-title">分析报告</span>
        </div>
        <DecisionCard 
          v-for="(decision, idx) in message.data.decisions" 
          :key="decision.symbol"
          :decision="decision"
          :index="idx"
        />
      </div>

      <!-- Markdown 内容渲染（双Agent文本） -->
      <div v-else-if="isDualAgentMarkdown(message.content)" class="markdown-container dual-agent-markdown">
        <MarkdownRender :content="message.content" />
      </div>

      <!-- Portfolio Suggestion -->
      <PortfolioSuggestion 
        v-else-if="message.data?.portfolio_suggestion && !message.data?.decisions?.length" 
        :suggestion="message.data.portfolio_suggestion" 
      />

      <!-- Plain Text -->
      <div v-else-if="message.content && !message.isClarification" class="ai-bubble">
        <p class="text">{{ message.content }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.user-message {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  justify-content: flex-end;
  animation: slide-up 0.4s ease-out forwards;
}

.user-content {
  flex: 1;
  display: flex;
  justify-content: flex-end;
}

.user-avatar {
  width: 2.25rem;
  height: 2.25rem;
  border-radius: 50%;
  background: linear-gradient(to bottom right, #3b82f6, #06b6d4);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 1rem;
  box-shadow: 0 0 20px -5px rgba(59, 130, 246, 0.4);
}

.user-bubble {
  max-width: 80%;
  background: rgba(59, 130, 246, 0.9);
  border-radius: 1rem;
  border-top-right-radius: 0.125rem;
  padding: 0.875rem 1.25rem;
  box-shadow: 0 4px 15px -5px rgba(59, 130, 246, 0.3);
}

.user-bubble .text {
  color: white;
  font-size: 0.9375rem;
  line-height: 1.6;
  margin: 0;
}

.ai-message {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  animation: slide-up 0.4s ease-out forwards;
}

.ai-avatar {
  width: 2.25rem;
  height: 2.25rem;
  border-radius: 50%;
  background: linear-gradient(to bottom right, #10b981, #14b8a6);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 1rem;
  box-shadow: 0 0 20px -5px rgba(16, 185, 129, 0.4);
}

.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.debug-info {
  padding: 0.5rem;
  background: rgba(0, 0, 0, 0.04);
  border-radius: 0.5rem;
  color: rgba(0, 0, 0, 0.5);
  font-size: 0.75rem;
}

/* 分析标题栏 */
.analysis-title-bar {
  display: flex;
  align-items: center;
  gap: 0.625rem;
  padding: 0.875rem 1rem;
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(124, 58, 237, 0.05));
  border: 1px solid rgba(139, 92, 246, 0.2);
  border-radius: 0.875rem;
  margin-bottom: 0.5rem;
}

.analysis-title-bar .analysis-icon {
  font-size: 1.25rem;
}

.analysis-title-bar .analysis-title {
  font-size: 1rem;
  font-weight: 700;
  color: #7c3aed;
}

/* 双Agent模式样式 */
.agent-analysis-section {
  border-radius: 1rem;
  overflow: hidden;
  border: 1px solid rgba(0, 0, 0, 0.08);
}

.agent-analysis-section.morefit {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.06), rgba(37, 99, 235, 0.03));
  border-color: rgba(59, 130, 246, 0.15);
}

.agent-analysis-section.technical {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.06), rgba(5, 150, 105, 0.03));
  border-color: rgba(16, 185, 129, 0.15);
}

.agent-analysis-section.riskmanager {
  background: linear-gradient(135deg, rgba(245, 158, 11, 0.06), rgba(217, 119, 6, 0.03));
  border-color: rgba(245, 158, 11, 0.15);
}

.agent-analysis-section.sentimentanalyzer {
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.06), rgba(124, 58, 237, 0.03));
  border-color: rgba(139, 92, 246, 0.15);
}

.agent-analysis-section.default {
  background: linear-gradient(135deg, rgba(107, 114, 128, 0.06), rgba(75, 85, 99, 0.03));
  border-color: rgba(107, 114, 128, 0.15);
}

.agent-section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
}

.agent-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.agent-icon {
  font-size: 1.25rem;
}

.agent-name {
  font-weight: 600;
  font-size: 0.9375rem;
}

.morefit .agent-name {
  color: #2563eb;
}

.technical .agent-name {
  color: #059669;
}

.riskmanager .agent-name {
  color: #d97706;
}

.sentimentanalyzer .agent-name {
  color: #7c3aed;
}

.default .agent-name {
  color: #6b7280;
}

.vote-tag {
  padding: 0.375rem 0.875rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.vote-tag.STRONG_BUY {
  background: rgba(16, 185, 129, 0.12);
  color: #059669;
}

.vote-tag.BUY {
  background: rgba(74, 222, 128, 0.12);
  color: #16a34a;
}

.vote-tag.HOLD {
  background: rgba(251, 191, 36, 0.15);
  color: #d97706;
}

.vote-tag.SELL {
  background: rgba(248, 113, 113, 0.12);
  color: #dc2626;
}

.vote-tag.STRONG_SELL {
  background: rgba(251, 113, 133, 0.12);
  color: #e11d48;
}

.vote-tag.REDUCE {
  background: rgba(245, 158, 11, 0.12);
  color: #d97706;
}

.agent-content {
  padding: 1.25rem;
}

.metrics-row {
  display: flex;
  gap: 1.5rem;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
}

.metric {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.metric-label {
  font-size: 0.6875rem;
  color: rgba(0, 0, 0, 0.45);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.metric-value {
  font-size: 1.125rem;
  font-weight: 700;
  color: #1e293b;
}

.reason-box {
  background: rgba(0, 0, 0, 0.04);
  border-radius: 0.75rem;
  padding: 1rem;
}

.reason-text {
  color: rgba(0, 0, 0, 0.75);
  font-size: 0.9375rem;
  line-height: 1.75;
  margin: 0;
}

/* 最终结果 */
.final-vote-section {
  margin-top: 0.5rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(0, 0, 0, 0.08);
}

.final-vote-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.875rem;
}

.final-vote-header .icon {
  font-size: 1.25rem;
}

.final-vote-header .title {
  font-weight: 600;
  color: rgba(0, 0, 0, 0.75);
  font-size: 1rem;
}

.final-vote-card {
  border-radius: 1rem;
  padding: 1.25rem;
  border: 1px solid;
}

.final-vote-card.STRONG_BUY {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.05));
  border-color: rgba(16, 185, 129, 0.2);
}

.final-vote-card.BUY {
  background: linear-gradient(135deg, rgba(74, 222, 128, 0.08), rgba(34, 197, 94, 0.05));
  border-color: rgba(74, 222, 128, 0.2);
}

.final-vote-card.HOLD {
  background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.05));
  border-color: rgba(251, 191, 36, 0.25);
}

.final-vote-card.SELL {
  background: linear-gradient(135deg, rgba(248, 113, 113, 0.08), rgba(239, 68, 68, 0.05));
  border-color: rgba(248, 113, 113, 0.2);
}

.final-vote-card.STRONG_SELL {
  background: linear-gradient(135deg, rgba(251, 113, 133, 0.08), rgba(225, 29, 72, 0.05));
  border-color: rgba(251, 113, 133, 0.2);
}

.final-vote-main {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}

.final-vote-label {
  font-size: 0.875rem;
  color: rgba(0, 0, 0, 0.55);
}

.final-vote-value {
  font-size: 1.5rem;
  font-weight: 700;
}

.final-vote-card.STRONG_BUY .final-vote-value,
.final-vote-card.BUY .final-vote-value {
  color: #059669;
}

.final-vote-card.HOLD .final-vote-value {
  color: #d97706;
}

.final-vote-card.SELL .final-vote-value,
.final-vote-card.STRONG_SELL .final-vote-value {
  color: #dc2626;
}

.final-metrics {
  display: flex;
  gap: 1.5rem;
  margin-bottom: 1rem;
}

.final-metric {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.final-metric .label {
  font-size: 0.6875rem;
  color: rgba(0, 0, 0, 0.45);
  text-transform: uppercase;
}

.final-metric .value {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1e293b;
}

.final-reason {
  font-size: 0.9375rem;
  color: rgba(0, 0, 0, 0.7);
  line-height: 1.7;
  margin: 0;
  padding-top: 1rem;
  border-top: 1px solid rgba(0, 0, 0, 0.08);
}

/* 其他样式 */
.analysis-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.analysis-icon {
  font-size: 1.25rem;
}

.analysis-title {
  font-size: 1rem;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.7);
}

.decisions-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.clarification-box {
  background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.05));
  border-radius: 1rem;
  padding: 1.25rem;
  border: 1px solid rgba(251, 191, 36, 0.25);
}

.clarification-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.clarification-header .icon {
  font-size: 1.25rem;
}

.clarification-header .title {
  font-weight: 600;
  color: #d97706;
}

.clarification-box .text {
  color: rgba(0, 0, 0, 0.75);
  font-size: 0.9375rem;
  line-height: 1.6;
  margin: 0;
  white-space: pre-line;
}

.markdown-container {
  background: rgba(0, 0, 0, 0.03);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 1rem;
  padding: 1.25rem;
}

.ai-bubble {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(24px);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 1rem;
  border-top-left-radius: 0.125rem;
  padding: 1rem 1.25rem;
  max-width: 80%;
}

.ai-bubble .text {
  color: rgba(0, 0, 0, 0.8);
  font-size: 0.9375rem;
  line-height: 1.7;
  margin: 0;
  white-space: pre-wrap;
}

@keyframes slide-up {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
