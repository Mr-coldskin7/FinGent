<script setup lang="ts">
import { ref, computed } from 'vue';

interface ToolCall {
  name: string;
  args: Record<string, any>;
  result?: string;
}

interface ThinkingStep {
  id: string;
  type: 'input' | 'thinking' | 'tool_decision' | 'tool_call' | 'tool_result' | 'observation' | 'analysis' | 'conclusion';
  title: string;
  content: string;
  details?: string;
  toolCalls?: ToolCall[];
  status: 'pending' | 'active' | 'completed' | 'error';
  timestamp?: number;
  agent?: string;
}

interface Props {
  steps: ThinkingStep[];
  isActive?: boolean;
}

const props = defineProps<Props>();

const expandedSteps = ref<Set<string>>(new Set());
const showToolDetails = ref<Set<string>>(new Set());

function toggleStep(stepId: string) {
  if (expandedSteps.value.has(stepId)) {
    expandedSteps.value.delete(stepId);
  } else {
    expandedSteps.value.add(stepId);
  }
}

function toggleToolDetails(stepId: string) {
  if (showToolDetails.value.has(stepId)) {
    showToolDetails.value.delete(stepId);
  } else {
    showToolDetails.value.add(stepId);
  }
}

function isExpanded(stepId: string): boolean {
  return expandedSteps.value.has(stepId);
}

function isToolDetailsVisible(stepId: string): boolean {
  return showToolDetails.value.has(stepId);
}

function getStepIcon(type: string): string {
  const icons: Record<string, string> = {
    'input': '👤',
    'thinking': '💭',
    'tool_decision': '🎯',
    'tool_call': '🔧',
    'tool_result': '📊',
    'observation': '👁️',
    'analysis': '🧠',
    'conclusion': '✅'
  };
  return icons[type] || '•';
}

function getStepLabel(type: string): string {
  const labels: Record<string, string> = {
    'input': '用户输入',
    'thinking': '思考推理',
    'tool_decision': '工具决策',
    'tool_call': '调用工具',
    'tool_result': '工具返回',
    'observation': '观察结果',
    'analysis': '分析判断',
    'conclusion': '得出结论'
  };
  return labels[type] || type;
}

function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    'pending': '#6b7280',
    'active': '#3b82f6',
    'completed': '#10b981',
    'error': '#ef4444'
  };
  return colors[status] || '#6b7280';
}

function formatArgs(args: Record<string, any>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return '无参数';
  return entries.map(([k, v]) => `${k}=${JSON.stringify(v).slice(0, 50)}`).join(', ');
}

const completedStepsCount = computed(() => {
  return props.steps.filter(s => s.status === 'completed').length;
});

const progressPercent = computed(() => {
  if (props.steps.length === 0) return 0;
  return (completedStepsCount.value / props.steps.length) * 100;
});

// 按工具调用分组步骤
const groupedSteps = computed(() => {
  const groups: any[] = [];
  let currentGroup: any = null;
  
  for (const step of props.steps) {
    if (step.type === 'tool_call') {
      // 开始新的工具调用组
      if (currentGroup) {
        groups.push(currentGroup);
      }
      currentGroup = {
        type: 'tool_execution',
        callStep: step,
        resultStep: null,
        id: step.id
      };
    } else if (step.type === 'tool_result' && currentGroup) {
      // 添加到当前组
      currentGroup.resultStep = step;
      groups.push(currentGroup);
      currentGroup = null;
    } else {
      // 普通步骤
      if (currentGroup) {
        groups.push(currentGroup);
        currentGroup = null;
      }
      groups.push({
        type: 'normal',
        step: step
      });
    }
  }
  
  if (currentGroup) {
    groups.push(currentGroup);
  }
  
  return groups;
});
</script>

<template>
  <div class="thinking-chain">
    <!-- 头部 -->
    <div class="chain-header">
      <div class="header-left">
        <span class="header-icon">🧠</span>
        <span class="header-title">Agent 思维链</span>
      </div>
      <div class="header-right">
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: `${progressPercent}%` }"></div>
        </div>
        <span class="progress-text">{{ completedStepsCount }}/{{ steps.length }}</span>
      </div>
    </div>

    <!-- 步骤列表 -->
    <div class="steps-container">
      <template v-for="(group, groupIdx) in groupedSteps" :key="groupIdx">
        <!-- 普通步骤 -->
        <div v-if="group.type === 'normal'" 
             :class="['step-item', group.step.status, { 'expanded': isExpanded(group.step.id) }]">
          <div class="step-connector">
            <div class="connector-line" v-if="groupIdx > 0"></div>
            <div class="status-dot" :style="{ background: getStatusColor(group.step.status) }">
              <span v-if="group.step.status === 'completed'">✓</span>
              <span v-else-if="group.step.status === 'active'" class="pulse"></span>
              <span v-else-if="group.step.status === 'error'">!</span>
            </div>
          </div>

          <div class="step-content" @click="toggleStep(group.step.id)">
            <div class="step-header">
              <div class="step-type">
                <span class="type-icon">{{ getStepIcon(group.step.type) }}</span>
                <span class="type-label">{{ getStepLabel(group.step.type) }}</span>
                <span v-if="group.step.agent" class="agent-badge">{{ group.step.agent }}</span>
              </div>
              <div class="step-status" :style="{ color: getStatusColor(group.step.status) }">
                {{ group.step.status === 'completed' ? '已完成' : 
                   group.step.status === 'active' ? '进行中' : 
                   group.step.status === 'error' ? '错误' : '等待中' }}
              </div>
            </div>

            <div class="step-title">{{ group.step.title }}</div>
            
            <div class="step-preview" v-if="!isExpanded(group.step.id)">
              {{ group.step.content.slice(0, 100) }}{{ group.step.content.length > 100 ? '...' : '' }}
            </div>

            <div class="step-details" v-else>
              <div class="detail-content">{{ group.step.content }}</div>
              <div class="detail-extra" v-if="group.step.details">
                <div class="extra-label">详细信息</div>
                <pre class="extra-text">{{ group.step.details }}</pre>
              </div>
            </div>

            <div class="expand-indicator">
              <svg class="arrow-icon" :class="{ 'expanded': isExpanded(group.step.id) }" 
                   viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M19 9l-7 7-7-7"/>
              </svg>
            </div>
          </div>
        </div>

        <!-- 工具执行组 -->
        <div v-else class="tool-execution-group">
          <div class="step-connector tool-connector">
            <div class="connector-line"></div>
            <div class="status-dot tool-dot" :style="{ background: '#fbbf24' }">
              <span>🔧</span>
            </div>
          </div>
          
          <div class="tool-execution-card" @click="toggleToolDetails(group.id)">
            <!-- 工具调用头部 -->
            <div class="tool-header">
              <div class="tool-title">
                <span class="tool-icon">⚡</span>
                <span>工具调用</span>
                <span class="tool-name">{{ group.callStep.title }}</span>
              </div>
              <div class="tool-toggle">
                <span class="toggle-text">{{ isToolDetailsVisible(group.id) ? '收起' : '展开' }}</span>
                <svg class="arrow-icon" :class="{ 'expanded': isToolDetailsVisible(group.id) }" 
                     viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M19 9l-7 7-7-7"/>
                </svg>
              </div>
            </div>

            <!-- 工具调用简要信息 -->
            <div class="tool-summary">
              <div class="summary-item">
                <span class="summary-label">工具:</span>
                <code class="summary-value">{{ group.callStep.content }}</code>
              </div>
            </div>

            <!-- 工具调用详情 -->
            <div v-if="isToolDetailsVisible(group.id)" class="tool-details">
              <!-- 调用步骤 -->
              <div class="tool-step call">
                <div class="tool-step-header">
                  <span class="step-dot call-dot"></span>
                  <span class="step-label">调用</span>
                </div>
                <div class="tool-step-content">
                  <div class="code-block">
                    <div class="code-header">
                      <span class="code-lang">参数</span>
                    </div>
                    <pre class="code-content">{{ group.callStep.details || group.callStep.content }}</pre>
                  </div>
                </div>
              </div>

              <!-- 结果步骤 -->
              <div v-if="group.resultStep" class="tool-step result">
                <div class="tool-step-header">
                  <span class="step-dot result-dot"></span>
                  <span class="step-label">返回</span>
                </div>
                <div class="tool-step-content">
                  <div class="code-block">
                    <div class="code-header">
                      <span class="code-lang">结果</span>
                    </div>
                    <pre class="code-content">{{ group.resultStep.content }}</pre>
                  </div>
                </div>
              </div>

              <!-- 等待结果 -->
              <div v-else class="tool-step pending">
                <div class="tool-step-header">
                  <span class="step-dot pending-dot"></span>
                  <span class="step-label">等待返回...</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.thinking-chain {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 1rem;
  overflow: hidden;
}

.chain-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.25rem;
  background: rgba(255, 255, 255, 0.03);
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.header-icon {
  font-size: 1.25rem;
}

.header-title {
  font-size: 0.9375rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.9);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.progress-bar {
  width: 80px;
  height: 4px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #10b981);
  border-radius: 2px;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
  font-family: monospace;
}

.steps-container {
  padding: 1rem 0;
}

/* 普通步骤样式 */
.step-item {
  display: flex;
  padding: 0.5rem 1.25rem;
  position: relative;
}

.step-connector {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 24px;
  margin-right: 0.875rem;
  flex-shrink: 0;
}

.tool-connector {
  padding-top: 0;
}

.connector-line {
  width: 2px;
  flex: 1;
  background: rgba(255, 255, 255, 0.1);
  margin-bottom: 4px;
}

.status-dot {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  color: white;
  font-weight: 600;
  flex-shrink: 0;
}

.tool-dot {
  background: #fbbf24 !important;
  font-size: 0.875rem;
}

.status-dot .pulse {
  width: 8px;
  height: 8px;
  background: white;
  border-radius: 50%;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.7; }
}

.step-content {
  flex: 1;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 0.75rem;
  padding: 0.875rem 1rem;
  cursor: pointer;
  transition: all 0.2s;
}

.step-content:hover {
  background: rgba(255, 255, 255, 0.05);
  border-color: rgba(255, 255, 255, 0.1);
}

.step-item.expanded .step-content {
  background: rgba(255, 255, 255, 0.06);
}

.step-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.step-type {
  display: flex;
  align-items: center;
  gap: 0.375rem;
}

.type-icon {
  font-size: 0.875rem;
}

.type-label {
  font-size: 0.6875rem;
  color: rgba(255, 255, 255, 0.5);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.agent-badge {
  font-size: 0.625rem;
  padding: 0.125rem 0.375rem;
  background: rgba(59, 130, 246, 0.2);
  color: #60a5fa;
  border-radius: 9999px;
  margin-left: 0.25rem;
}

.step-status {
  font-size: 0.6875rem;
  font-weight: 600;
}

.step-title {
  font-size: 0.9375rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.9);
  margin-bottom: 0.375rem;
}

.step-preview {
  font-size: 0.8125rem;
  color: rgba(255, 255, 255, 0.5);
  line-height: 1.5;
}

.step-details {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.detail-content {
  font-size: 0.875rem;
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.7;
  margin-bottom: 0.75rem;
}

.detail-extra {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 0.5rem;
  padding: 0.75rem;
}

.extra-label {
  font-size: 0.6875rem;
  color: rgba(255, 255, 255, 0.4);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.375rem;
}

.extra-text {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.6);
  line-height: 1.6;
  margin: 0;
  white-space: pre-wrap;
  font-family: monospace;
  overflow-x: auto;
}

.expand-indicator {
  display: flex;
  justify-content: center;
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid rgba(255, 255, 255, 0.04);
}

.arrow-icon {
  width: 1rem;
  height: 1rem;
  color: rgba(255, 255, 255, 0.3);
  transition: transform 0.2s;
}

.arrow-icon.expanded {
  transform: rotate(180deg);
}

/* 工具执行组样式 */
.tool-execution-group {
  display: flex;
  padding: 0.5rem 1.25rem;
}

.tool-execution-card {
  flex: 1;
  background: linear-gradient(135deg, rgba(251, 191, 36, 0.08), rgba(245, 158, 11, 0.04));
  border: 1px solid rgba(251, 191, 36, 0.2);
  border-radius: 0.75rem;
  padding: 0.875rem 1rem;
  cursor: pointer;
  transition: all 0.2s;
}

.tool-execution-card:hover {
  background: linear-gradient(135deg, rgba(251, 191, 36, 0.12), rgba(245, 158, 11, 0.06));
  border-color: rgba(251, 191, 36, 0.3);
}

.tool-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.tool-title {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  font-size: 0.875rem;
  font-weight: 600;
  color: #fbbf24;
}

.tool-icon {
  font-size: 1rem;
}

.tool-name {
  font-family: monospace;
  font-size: 0.75rem;
  padding: 0.125rem 0.5rem;
  background: rgba(251, 191, 36, 0.15);
  border-radius: 0.25rem;
}

.tool-toggle {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
}

.toggle-text {
  font-size: 0.6875rem;
}

.tool-summary {
  margin-bottom: 0.5rem;
}

.summary-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.summary-label {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.4);
}

.summary-value {
  font-size: 0.8125rem;
  color: rgba(255, 255, 255, 0.7);
  font-family: monospace;
}

/* 工具详情 */
.tool-details {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid rgba(251, 191, 36, 0.15);
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.tool-step {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.tool-step-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.step-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.call-dot {
  background: #3b82f6;
}

.result-dot {
  background: #10b981;
}

.pending-dot {
  background: #fbbf24;
  animation: pulse 1.5s ease-in-out infinite;
}

.step-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.6);
}

.tool-step-content {
  margin-left: 1rem;
}

.code-block {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 0.5rem;
  overflow: hidden;
}

.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.375rem 0.75rem;
  background: rgba(255, 255, 255, 0.05);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.code-lang {
  font-size: 0.625rem;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.4);
  font-weight: 600;
  letter-spacing: 0.05em;
}

.code-content {
  padding: 0.75rem;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.7);
  line-height: 1.6;
  margin: 0;
  white-space: pre-wrap;
  font-family: 'Fira Code', monospace;
  max-height: 200px;
  overflow-y: auto;
}

/* 状态样式 */
.step-item.completed .step-content {
  border-color: rgba(16, 185, 129, 0.15);
}

.step-item.active .step-content {
  border-color: rgba(59, 130, 246, 0.3);
  box-shadow: 0 0 20px -5px rgba(59, 130, 246, 0.2);
}

.step-item.error .step-content {
  border-color: rgba(239, 68, 68, 0.3);
}
</style>
