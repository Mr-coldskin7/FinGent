<script setup lang="ts">
import type { ToolCall, AgentToolChain } from '../types/index';

interface Props {
  toolChain?: ToolCall[];
  agentName?: string;
  stock?: string;
  allToolChains?: AgentToolChain[];
}

const props = defineProps<Props>();

// 判断是否为双Agent模式
const isDualMode = () => {
  return props.allToolChains && props.allToolChains.length > 0;
};

// 获取工具图标
function getToolIcon(toolName?: string): string {
  const icons: Record<string, string> = {
    'get_stock_price': '📈',
    'get_stock_basic_info': '📋',
    'get_stock_company_info': '🏢',
    'get_stock_financial_statements': '📊',
    'get_stock_financial_report_links': '📑',
    'get_full_stock_analysis': '🔍',
  };
  return icons[toolName || ''] || '🔧';
}

// 格式化工具名称
function formatToolName(name?: string): string {
  if (!name) return '未知工具';
  const names: Record<string, string> = {
    'get_stock_price': '获取股价数据',
    'get_stock_basic_info': '获取基础信息',
    'get_stock_company_info': '获取公司信息',
    'get_stock_financial_statements': '获取财务报表',
    'get_stock_financial_report_links': '获取财报链接',
    'get_full_stock_analysis': '全面分析',
  };
  return names[name] || name;
}

// 格式化参数
function formatArgs(args?: Record<string, any>): string {
  if (!args) return '';
  const entries = Object.entries(args);
  if (entries.length === 0) return '';
  return entries.map(([k, v]) => `${k}=${v}`).join(', ');
}

// 获取Agent显示名称
function getAgentName(name: string): string {
  const names: Record<string, string> = {
    'Morefit': '📊 基本面分析',
    'TECHNICAL_NERD': '📈 技术面分析',
  };
  return names[name] || name;
}

// 获取Agent颜色
function getAgentColor(name: string): string {
  const colors: Record<string, string> = {
    'Morefit': '#3b82f6',
    'TECHNICAL_NERD': '#10b981',
  };
  return colors[name] || '#6b7280';
}
</script>

<template>
  <!-- 双Agent模式 -->
  <div v-if="isDualMode()" class="tool-chain dual-mode">
    <div class="chain-header">
      <span class="header-icon">🔧</span>
      <span class="header-text">工具调用过程</span>
      <span class="dual-badge">双Agent分析</span>
    </div>
    
    <div class="agents-container">
      <div 
        v-for="(agentChain, idx) in allToolChains" 
        :key="agentChain.agent"
        class="agent-section"
      >
        <div class="agent-header" :style="{ borderColor: getAgentColor(agentChain.agent) }">
          <div class="agent-badge" :style="{ background: getAgentColor(agentChain.agent) + '20', color: getAgentColor(agentChain.agent) }">
            {{ getAgentName(agentChain.agent) }}
          </div>
        </div>
        
        <div class="agent-steps">
          <div 
            v-for="(step, stepIdx) in agentChain.steps" 
            :key="stepIdx"
            :class="['chain-step', step.type]"
          >
            <!-- 输入步骤 -->
            <template v-if="step.type === 'input'">
              <div class="step-icon input-icon">👤</div>
              <div class="step-content">
                <div class="step-label">用户输入</div>
                <div class="step-text">{{ step.content }}</div>
              </div>
            </template>
            
            <!-- 工具调用步骤 -->
            <template v-else-if="step.type === 'tool_call'">
              <div class="step-icon tool-icon">{{ getToolIcon(step.name) }}</div>
              <div class="step-content">
                <div class="step-label">调用工具</div>
                <div class="step-text tool-name">{{ formatToolName(step.name) }}</div>
                <div v-if="step.args" class="step-args">
                  <span class="args-label">参数:</span>
                  <code class="args-code">{{ formatArgs(step.args) }}</code>
                </div>
              </div>
            </template>
            
            <!-- 工具结果步骤 -->
            <template v-else-if="step.type === 'tool_result'">
              <div class="step-icon result-icon">📊</div>
              <div class="step-content">
                <div class="step-label">工具返回</div>
                <div class="step-text result-text">{{ step.content }}</div>
              </div>
            </template>
            
            <!-- 连接线 -->
            <div v-if="stepIdx < agentChain.steps.length - 1" class="chain-connector"></div>
          </div>
        </div>
        
        <!-- Agent间分隔线 -->
        <div v-if="idx < (allToolChains?.length || 0) - 1" class="agent-divider">
          <span class="divider-text">综合分析</span>
        </div>
      </div>
    </div>
  </div>

  <!-- 单Agent模式 -->
  <div v-else-if="toolChain?.length" class="tool-chain single-mode">
    <div class="chain-header">
      <span class="header-icon">🔧</span>
      <span class="header-text">工具调用过程</span>
      <span v-if="agentName" class="agent-badge">{{ agentName }}</span>
    </div>
    
    <div class="chain-body">
      <div 
        v-for="(step, idx) in toolChain" 
        :key="idx"
        :class="['chain-step', step.type]"
      >
        <!-- 输入步骤 -->
        <template v-if="step.type === 'input'">
          <div class="step-icon input-icon">👤</div>
          <div class="step-content">
            <div class="step-label">用户输入</div>
            <div class="step-text">{{ step.content }}</div>
          </div>
        </template>
        
        <!-- 工具调用步骤 -->
        <template v-else-if="step.type === 'tool_call'">
          <div class="step-icon tool-icon">{{ getToolIcon(step.name) }}</div>
          <div class="step-content">
            <div class="step-label">调用工具</div>
            <div class="step-text tool-name">{{ formatToolName(step.name) }}</div>
            <div v-if="step.args" class="step-args">
              <span class="args-label">参数:</span>
              <code class="args-code">{{ formatArgs(step.args) }}</code>
            </div>
          </div>
        </template>
        
        <!-- 工具结果步骤 -->
        <template v-else-if="step.type === 'tool_result'">
          <div class="step-icon result-icon">📊</div>
          <div class="step-content">
            <div class="step-label">工具返回</div>
            <div class="step-text result-text">{{ step.content }}</div>
          </div>
        </template>
        
        <!-- 连接线 -->
        <div v-if="idx < toolChain.length - 1" class="chain-connector"></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tool-chain {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 1rem;
  overflow: hidden;
  margin-bottom: 1rem;
}

.chain-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.875rem 1rem;
  background: rgba(255, 255, 255, 0.03);
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.header-icon {
  font-size: 1rem;
}

.header-text {
  font-size: 0.8125rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.6);
  flex: 1;
}

.agent-badge {
  font-size: 0.6875rem;
  padding: 0.25rem 0.625rem;
  background: rgba(59, 130, 246, 0.2);
  color: #60a5fa;
  border-radius: 9999px;
  font-weight: 600;
}

.dual-badge {
  font-size: 0.6875rem;
  padding: 0.25rem 0.625rem;
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(16, 185, 129, 0.2));
  color: #34d399;
  border-radius: 9999px;
  font-weight: 600;
}

/* 单Agent模式 */
.chain-body {
  padding: 1rem;
}

/* 双Agent模式 */
.agents-container {
  padding: 1rem;
}

.agent-section {
  margin-bottom: 1rem;
}

.agent-section:last-child {
  margin-bottom: 0;
}

.agent-header {
  padding-bottom: 0.75rem;
  margin-bottom: 0.75rem;
  border-bottom: 2px solid;
}

.agent-badge {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.375rem 0.875rem;
  border-radius: 0.5rem;
}

.agent-steps {
  padding-left: 0.5rem;
}

.agent-divider {
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 1rem 0;
  position: relative;
}

.agent-divider::before {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(to right, transparent, rgba(255,255,255,0.2), transparent);
}

.divider-text {
  position: relative;
  background: #020617;
  padding: 0 1rem;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.4);
  font-weight: 500;
}

/* 通用步骤样式 */
.chain-step {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  position: relative;
  padding-bottom: 1rem;
}

.chain-step:last-child {
  padding-bottom: 0;
}

.step-icon {
  width: 2rem;
  height: 2rem;
  border-radius: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.875rem;
  flex-shrink: 0;
}

.input-icon {
  background: rgba(59, 130, 246, 0.15);
}

.tool-icon {
  background: rgba(251, 191, 36, 0.15);
}

.result-icon {
  background: rgba(16, 185, 129, 0.15);
}

.step-content {
  flex: 1;
  min-width: 0;
}

.step-label {
  font-size: 0.6875rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.4);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.25rem;
}

.step-text {
  font-size: 0.8125rem;
  color: rgba(255, 255, 255, 0.85);
  line-height: 1.5;
  word-break: break-word;
}

.tool-name {
  font-weight: 600;
  color: #fbbf24;
}

.result-text {
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.75rem;
}

.step-args {
  margin-top: 0.375rem;
  padding: 0.5rem 0.625rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 0.375rem;
  font-size: 0.75rem;
}

.args-label {
  color: rgba(255, 255, 255, 0.4);
  margin-right: 0.375rem;
}

.args-code {
  color: #60a5fa;
  font-family: monospace;
}

.chain-connector {
  position: absolute;
  left: 1rem;
  top: 2rem;
  bottom: 0;
  width: 2px;
  background: linear-gradient(to bottom, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
  transform: translateX(-50%);
}
</style>
