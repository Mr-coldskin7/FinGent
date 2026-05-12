<script setup lang="ts">
import { computed } from 'vue';
import type { Decision, VoteType } from '../types/index';

interface Props {
  decision: Decision;
  index?: number;
}

const props = withDefaults(defineProps<Props>(), {
  index: 0
});

// 股票代码到名称的映射（常见股票）
const stockNameMap: Record<string, string> = {
  '600519': '贵州茅台',
  '000858': '五粮液',
  '000333': '美的集团',
  '000651': '格力电器',
  '002594': '比亚迪',
  '300750': '宁德时代',
  '601012': '隆基绿能',
  '600036': '招商银行',
  '601398': '工商银行',
  '601318': '中国平安',
  '600276': '恒瑞医药',
  '002415': '海康威视',
  '000002': '万科A',
  '600030': '中信证券',
};

const voteConfig: Record<VoteType, {
  label: string;
  color: string;
  bgColor: string;
  icon: string;
  gradient: string;
}> = {
  STRONG_BUY: { 
    label: '强烈买入', 
    color: '#34d399', 
    bgColor: 'rgba(16, 185, 129, 0.2)', 
    icon: '🚀',
    gradient: 'linear-gradient(90deg, #10b981, #34d399)'
  },
  BUY: { 
    label: '买入', 
    color: '#4ade80', 
    bgColor: 'rgba(74, 222, 128, 0.2)', 
    icon: '📈',
    gradient: 'linear-gradient(90deg, #22c55e, #4ade80)'
  },
  HOLD: { 
    label: '持有观望', 
    color: '#fbbf24', 
    bgColor: 'rgba(251, 191, 36, 0.2)', 
    icon: '⏸️',
    gradient: 'linear-gradient(90deg, #f59e0b, #fbbf24)'
  },
  SELL: { 
    label: '卖出', 
    color: '#f87171', 
    bgColor: 'rgba(248, 113, 113, 0.2)', 
    icon: '📉',
    gradient: 'linear-gradient(90deg, #ef4444, #f87171)'
  },
  STRONG_SELL: { 
    label: '强烈卖出', 
    color: '#fb7185', 
    bgColor: 'rgba(251, 113, 133, 0.2)', 
    icon: '⚠️',
    gradient: 'linear-gradient(90deg, #e11d48, #fb7185)'
  },
};

const config = computed(() => voteConfig[props.decision.vote] || voteConfig.HOLD);
const confidencePercent = computed(() => Math.min((props.decision.confidence || 0) * 100, 100));
const positionPercent = computed(() => Math.min((props.decision.target_position_pct || 0) * 100, 100));

// 获取股票显示名称
const displayName = computed(() => {
  const symbol = props.decision.symbol;
  return stockNameMap[symbol] || symbol;
});

// 判断是否为A股代码
const isStockCode = computed(() => /^\d{6}$/.test(props.decision.symbol));
</script>

<template>
  <div class="decision-card" :style="{ animationDelay: `${index * 0.1}s` }">
    <!-- Header -->
    <div class="card-header">
      <div class="header-content">
        <div class="stock-info">
          <div class="stock-icon-box">
            <span class="stock-icon">{{ isStockCode ? '📈' : '💼' }}</span>
          </div>
          <div class="stock-details">
            <h3 class="stock-name">{{ displayName }}</h3>
            <p class="stock-code">{{ decision.symbol }}</p>
          </div>
        </div>
        
        <div class="vote-badge" :style="{ background: config.bgColor, borderColor: config.color + '40' }">
          <span class="vote-icon">{{ config.icon }}</span>
          <span class="vote-label" :style="{ color: config.color }">{{ config.label }}</span>
        </div>
      </div>
      
      <!-- Decorative gradient line -->
      <div class="gradient-line" :style="{ background: config.gradient }"></div>
    </div>

    <!-- Body -->
    <div class="card-body">
      <!-- Analysis Reason -->
      <div class="reason-section">
        <div class="reason-header">
          <div class="reason-bar" :style="{ background: config.gradient }"></div>
          <span class="reason-label">分析理由</span>
        </div>
        <p class="reason-text">{{ decision.reason }}</p>
      </div>

      <!-- Metrics -->
      <div class="metrics-grid">
        <!-- Confidence -->
        <div class="metric-box">
          <div class="metric-header">
            <span class="metric-label">
              <span class="metric-icon">🎯</span>
              置信度
            </span>
            <span class="metric-value" :style="{ color: config.color }">{{ confidencePercent.toFixed(0) }}%</span>
          </div>
          <div class="progress-bar">
            <div 
              class="progress-fill"
              :style="{ width: `${confidencePercent}%`, background: config.gradient }"
            />
          </div>
        </div>

        <!-- Target Position -->
        <div v-if="decision.target_position_pct !== undefined" class="metric-box">
          <div class="metric-header">
            <span class="metric-label">
              <span class="metric-icon">⚖️</span>
              建议仓位
            </span>
            <span class="metric-value">{{ positionPercent.toFixed(0) }}%</span>
          </div>
          <div class="progress-bar">
            <div 
              class="progress-fill position"
              :style="{ width: `${positionPercent}%` }"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.decision-card {
  animation: slide-up 0.5s ease-out forwards;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(24px);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 1.25rem;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}

.decision-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.1);
}

.card-header {
  position: relative;
  padding: 1.25rem 1.5rem;
  background: linear-gradient(to right, rgba(0,0,0,0.02), transparent);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.stock-info {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.stock-icon-box {
  width: 3rem;
  height: 3rem;
  border-radius: 0.875rem;
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(6, 182, 212, 0.08));
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(0, 0, 0, 0.06);
  box-shadow: 0 4px 15px -5px rgba(59, 130, 246, 0.1);
}

.stock-icon {
  font-size: 1.25rem;
}

.stock-details {
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
}

.stock-name {
  font-size: 1.25rem;
  font-weight: 700;
  color: #1e293b;
  margin: 0;
}

.stock-code {
  font-size: 0.8125rem;
  color: rgba(0, 0, 0, 0.5);
  margin: 0;
  font-family: monospace;
  letter-spacing: 0.05em;
}

.vote-badge {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.5rem 1rem;
  border-radius: 9999px;
  border: 1px solid;
  font-weight: 600;
  box-shadow: 0 4px 15px -5px rgba(0, 0, 0, 0.2);
}

.vote-icon {
  font-size: 1rem;
}

.vote-label {
  font-size: 0.875rem;
}

.gradient-line {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 2px;
  opacity: 0.6;
}

.card-body {
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.reason-section {
  margin-bottom: 0.25rem;
}

.reason-header {
  display: flex;
  align-items: center;
  gap: 0.625rem;
  margin-bottom: 0.875rem;
}

.reason-bar {
  width: 0.25rem;
  height: 1.125rem;
  border-radius: 9999px;
}

.reason-label {
  font-size: 0.6875rem;
  font-weight: 700;
  color: rgba(0, 0, 0, 0.5);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.reason-text {
  color: rgba(0, 0, 0, 0.8);
  font-size: 0.9375rem;
  line-height: 1.85;
  margin: 0;
  padding-left: 0.875rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
}

.metric-box {
  background: rgba(0, 0, 0, 0.02);
  border-radius: 0.875rem;
  padding: 1rem;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.metric-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.625rem;
}

.metric-label {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  font-size: 0.8125rem;
  color: rgba(0, 0, 0, 0.55);
}

.metric-icon {
  font-size: 0.875rem;
}

.metric-value {
  font-size: 1.125rem;
  font-weight: 700;
}

.progress-bar {
  position: relative;
  height: 0.5rem;
  background: rgba(0, 0, 0, 0.06);
  border-radius: 9999px;
  overflow: hidden;
}

.progress-fill {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  border-radius: 9999px;
  transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}

.progress-fill.position {
  background: linear-gradient(90deg, #3b82f6, #22d3ee);
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
