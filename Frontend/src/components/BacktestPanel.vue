<script setup lang="ts">
import { ref, reactive, computed, watch } from 'vue';
import type { BacktestRequest, BacktestResult, DailyUpdate } from '../types/backtest';
import { DEFAULT_BACKTEST_CONFIG } from '../types/backtest';
import BacktestCandleChart from './BacktestCandleChart.vue';

const API_BASE = 'http://localhost:8000';

const loading = ref(false);
const streaming = ref(false);

// 调试：监控状态变化
watch(loading, (v) => console.log('[Debug] loading changed:', v));
watch(streaming, (v) => console.log('[Debug] streaming changed:', v));

const config = reactive<BacktestRequest>({ ...DEFAULT_BACKTEST_CONFIG });
const result = ref<BacktestResult | null>(null);
const dailyUpdates = ref<DailyUpdate[]>([]);
const error = ref<string>('');
const tradeSignals = ref<any[]>([]);

// 图表数据 - 从回测实时数据构建
const chartData = ref<any>(null);
const chartLoading = ref(false);

// 回测进度
const progress = ref({ current: 0, total: 0 });

// AbortController 用于取消回测
let streamAbortController: AbortController | null = null;

// 格式化百分比
const formatPercent = (num: number | undefined, decimals = 2) => {
  if (num === undefined || num === null) return '-';
  return (num > 0 ? '+' : '') + num.toFixed(decimals) + '%';
};

// 从回测数据构建图表数据
const buildChartDataFromBacktest = () => {
  console.log('[Chart] 构建图表数据，dailyUpdates:', dailyUpdates.value.length);
  
  if (!dailyUpdates.value.length) {
    console.log('[Chart] 没有回测数据，无法构建图表');
    return null;
  }

  // 构建K线数据 - 使用OHLC数据
  const candles = dailyUpdates.value.map((update: any) => ({
    time: update.date,
    open: update.open_price || update.close_price,
    high: update.high_price || update.close_price,
    low: update.low_price || update.close_price,
    close: update.close_price,
    volume: update.volume || 0
  }));

  // 构建交易标记 - 只显示实际的买入/卖出信号，HOLD不显示
  console.log('[Chart] 构建交易标记, tradeSignals:', tradeSignals.value.length);
  const trade_markers = tradeSignals.value
    .filter((signal: any) => signal.signal === 'BUY' || signal.signal === 'SELL')
    .map((signal: any) => ({
      time: signal.date,
      type: signal.signal === 'BUY' ? 'buy' : 'sell',
      price: signal.price,
      size: 100, // 默认数量
      reason: signal.reason // 保留原因用于tooltip
    }));
  console.log('[Chart] 交易标记:', trade_markers);

  // 计算统计信息
  const initial_value = config.initial_cash;
  const final_value = dailyUpdates.value[dailyUpdates.value.length - 1]?.portfolio_value || initial_value;
  const return_pct = ((final_value / initial_value) - 1) * 100;

  const data = {
    symbol: config.symbol,
    candles,
    trade_markers,
    statistics: {
      initial_value,
      final_value,
      return_pct: Math.round(return_pct * 100) / 100,
      total_trades: trade_markers.length,
      buy_count: trade_markers.filter((m: any) => m.type === 'buy').length,
      sell_count: trade_markers.filter((m: any) => m.type === 'sell').length
    }
  };

  console.log('[Chart] 图表数据构建完成:', data.candles.length, '根K线');
  return data;
};

// 从审计文件加载图表数据（用于普通回测）
const loadChartDataFromAudit = async () => {
  try {
    console.log('[Chart] 从审计文件加载图表数据');
    const res = await fetch(`${API_BASE}/api/v1/backtest-chart?symbol=${config.symbol}`);
    const data = await res.json();
    
    if (data.success) {
      console.log('[Chart] 图表数据加载成功:', data.data?.candles?.length, '根K线');
      chartData.value = data.data;
    } else {
      console.error('[Chart] 加载图表数据失败:', data.error);
    }
  } catch (err) {
    console.error('[Chart] 加载图表数据错误:', err);
  }
};

// 执行普通回测
const runBacktest = async () => {
  console.log('[Backtest] 点击运行回测按钮');
  loading.value = true;
  error.value = '';
  result.value = null;
  dailyUpdates.value = [];
  tradeSignals.value = [];
  chartData.value = null;
  
  try {
    console.log('[Backtest] 发送请求到:', `${API_BASE}/api/v1/backtest`);
    const res = await fetch(`${API_BASE}/api/v1/backtest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...config,
        end: config.end || undefined,
      }),
    });
    
    const data = await res.json();
    
    if (data.success) {
      result.value = data.result;
      console.log('[Backtest] 回测成功，准备获取图表数据');
      // 普通回测完成后，从审计文件获取图表数据
      await loadChartDataFromAudit();
    } else {
      error.value = data.error || '回测执行失败';
    }
  } catch (err: any) {
    console.error('[Backtest] 回测错误:', err);
    error.value = '网络错误: ' + err.message;
  } finally {
    if (!streaming.value) {
      loading.value = false;
    }
  }
};

// 执行流式回测
const runStreamingBacktest = async (retryCount = 0) => {
  console.log('[Backtest] 点击流式回测按钮, 重试次数:', retryCount);
  if (retryCount === 0) {
    streaming.value = true;
    loading.value = true;
    error.value = '';
    result.value = null;
    dailyUpdates.value = [];
    tradeSignals.value = [];
    chartData.value = null;
    progress.value = { current: 0, total: 0 };
  }
  
  // 创建新的 AbortController
  streamAbortController = new AbortController();
  
  // 用于累积不完整的 SSE 数据
  let buffer = '';
  // 存储当前解析的事件
  let currentEvent: { event?: string; data?: string } = {};
  
  try {
    const response = await fetch(`${API_BASE}/api/v1/backtest-stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...config,
        end: config.end || undefined,
      }),
      signal: streamAbortController.signal,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    
    if (!reader) {
      error.value = '无法读取流数据';
      streaming.value = false;
      loading.value = false;
      return;
    }
    
    console.log('[Stream] 开始读取流数据...');
    
    while (true) {
      // 检查是否被取消
      if (streamAbortController.signal.aborted) {
        console.log('[Stream] 回测已被取消');
        break;
      }
      
      const { done, value } = await reader.read();
      if (done) {
        console.log('[Stream] 读取完成');
        break;
      }
      
      // 解码并追加到缓冲区
      buffer += decoder.decode(value, { stream: true });
      
      // 按行分割处理
      const lines = buffer.split('\n');
      // 保留最后一行（可能不完整）到缓冲区
      buffer = lines.pop() || '';
      
      let shouldBreak = false;
      for (const line of lines) {
        const trimmedLine = line.trim();
        
        // SSE 格式: 空行表示一个事件结束
        if (!trimmedLine) {
          if (currentEvent.event || currentEvent.data) {
            // 处理完整的事件
            const shouldStop = handleSSEEvent(currentEvent);
            currentEvent = {};
            // 如果收到取消事件，停止读取
            if (shouldStop) {
              console.log('[Stream] 收到停止信号，中断读取');
              shouldBreak = true;
              break;
            }
          }
          continue;
        }
        
        // 解析 SSE 字段
        if (trimmedLine.startsWith('event:')) {
          currentEvent.event = trimmedLine.slice(6).trim();
        } else if (trimmedLine.startsWith('data:')) {
          // data 字段可能有多行，需要累积
          const dataLine = trimmedLine.slice(5).trim();
          currentEvent.data = currentEvent.data ? currentEvent.data + '\n' + dataLine : dataLine;
        } else if (trimmedLine.startsWith('id:')) {
          // 忽略 id 字段
        } else if (trimmedLine.startsWith('retry:')) {
          // 忽略 retry 字段
        }
      }
      
      // 检查是否需要停止
      if (shouldBreak || streamAbortController.signal.aborted) {
        break;
      }
    }
    
    // 处理缓冲区中剩余的数据
    if (buffer.trim() && !streamAbortController.signal.aborted) {
      console.log('[Stream] 处理剩余缓冲区:', buffer);
      const trimmedLine = buffer.trim();
      if (trimmedLine.startsWith('event:')) {
        currentEvent.event = trimmedLine.slice(6).trim();
      } else if (trimmedLine.startsWith('data:')) {
        currentEvent.data = trimmedLine.slice(5).trim();
      }
      if (currentEvent.event || currentEvent.data) {
        handleSSEEvent(currentEvent);
      }
    }
    
  } catch (err: any) {
    if (err.name === 'AbortError') {
      console.log('[Stream] 请求已取消');
      error.value = '回测已取消';
      streaming.value = false;
    } else if (err.message?.includes('timeout') || err.message?.includes('网络')) {
      // 超时或网络错误，尝试重连
      if (retryCount < 3) {
        console.log(`[Stream] 连接中断，${5 * (retryCount + 1)}秒后重试 (${retryCount + 1}/3)...`);
        error.value = `连接中断，${5 * (retryCount + 1)}秒后重试...`;
        await new Promise(resolve => setTimeout(resolve, 5000 * (retryCount + 1)));
        // 重新启动流式回测，保留已有数据
        return runStreamingBacktest(retryCount + 1);
      } else {
        console.error('[Stream] 重试次数用尽');
        error.value = '回测连接失败，请稍后重试';
        streaming.value = false;
      }
    } else {
      console.error('[Stream] 流式回测错误:', err);
      error.value = '流式回测错误: ' + err.message;
      streaming.value = false;
    }
  } finally {
    if (!streaming.value) {
      loading.value = false;
    }
    console.log('[Stream] 流式回测结束，共接收', dailyUpdates.value.length, '天数据');
    
    // 流式回测结束后构建图表数据
    chartData.value = buildChartDataFromBacktest();
  }
};

// 取消回测
const cancelBacktest = async () => {
  console.log('[Backtest] 用户点击取消按钮');
  
  // 1. 中止 fetch 请求
  if (streamAbortController) {
    streamAbortController.abort();
    console.log('[Backtest] 已中止 fetch 请求');
  }
  
  // 2. 发送取消请求到后端
  try {
    const res = await fetch(`${API_BASE}/api/v1/backtest-cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    const data = await res.json();
    console.log('[Backtest] 取消请求响应:', data);
  } catch (err) {
    console.error('[Backtest] 发送取消请求失败:', err);
  }
  
  // 3. 重置前端状态
  streaming.value = false;
  loading.value = false;
  error.value = '回测已取消';
};

// 处理单个 SSE 事件
// 返回 true 表示应该停止读取
const handleSSEEvent = (event: { event?: string; data?: string }): boolean => {
  if (!event.event || !event.data) return false;
  
  console.log(`[Stream] 收到事件: ${event.event}`);
  
  try {
    switch (event.event) {
      case 'start': {
        const data = JSON.parse(event.data);
        console.log('[Stream] 回测开始:', data);
        progress.value.total = data.total_days || 0;
        break;
      }
      case 'daily_update': {
        const update = JSON.parse(event.data) as DailyUpdate;
        console.log('[Stream] 日更新:', update.date, update.portfolio_value);
        dailyUpdates.value.push(update);
        
        // 如果有交易信号，则添加到交易信号数组中
        if (update.signal) {
          const signalPoint = {
            date: update.date,
            price: update.close_price,
            signal: update.signal.vote,
            confidence: update.signal.confidence,
            reason: update.signal.reason || ''
          };
          tradeSignals.value.push(signalPoint);
          
          // 实时更新图表数据
          chartData.value = buildChartDataFromBacktest();
        }
        
        progress.value.current = update.day_number || dailyUpdates.value.length;
        break;
      }
      case 'final_result': {
        const finalData = JSON.parse(event.data);
        console.log('[Stream] 最终结果:', finalData);
        result.value = finalData;
        streaming.value = false;
        loading.value = false;
        return true; // 停止读取
      }
      case 'cancelled': {
        const cancelledData = JSON.parse(event.data);
        console.log('[Stream] 回测已取消:', cancelledData);
        error.value = '回测已取消';
        streaming.value = false;
        loading.value = false;
        return true;
      }
      case 'error': {
        const errData = JSON.parse(event.data);
        console.error('[Stream] 错误:', errData);
        error.value = errData.error || '回测出错';
        streaming.value = false;
        loading.value = false;
        return true; // 停止读取
      }
      case 'ping': {
        // 心跳包，可以显示等待状态
        const pingData = JSON.parse(event.data);
        if (pingData.status === 'waiting') {
          console.log(`[Stream] 等待数据中... ${pingData.elapsed?.toFixed(0)}秒`);
        }
        break;
      }
      default:
        console.log('[Stream] 未知事件类型:', event.event);
    }
  } catch (e: any) {
    console.error('[Stream] 解析 JSON 失败:', e.message);
  }
  return false;
};

// 计算权益曲线数据
const equityCurveData = computed(() => {
  if (!config.initial_cash) return [];
  
  const data = [
    { date: '开始', value: config.initial_cash },
    ...dailyUpdates.value.map(u => ({
      date: u.date.slice(5),
      value: u.portfolio_value,
    })),
  ];
  
  if (result.value) {
    data.push({
      date: '结束',
      value: result.value.end_value,
    });
  }
  
  return data;
});

// 重置配置
const resetConfig = () => {
  Object.assign(config, DEFAULT_BACKTEST_CONFIG);
};

// 快速设置常用参数
const quickConfigs = [
  { name: '保守型', min_confidence: 0.7, rebalance_threshold: 0.05 },
  { name: '平衡型', min_confidence: 0.5, rebalance_threshold: 0.02 },
  { name: '激进型', min_confidence: 0.3, rebalance_threshold: 0.01 },
];

const applyQuickConfig = (cfg: typeof quickConfigs[0]) => {
  config.min_confidence = cfg.min_confidence;
  config.rebalance_threshold = cfg.rebalance_threshold;
};
</script>

<template>
  <div class="backtest-panel">
    <div class="panel-header">
      <h2 class="panel-title">📊 策略回测</h2>
      <p class="panel-desc">基于历史数据验证双 Agent 投票策略的有效性</p>
    </div>
    
    <div class="panel-content">
      <!-- 配置区域 -->
      <div class="config-section">
        <h3 class="section-title">回测参数</h3>
        
        <!-- 快速配置 -->
        <div class="quick-configs">
          <span class="quick-label">快速配置:</span>
          <button
            v-for="cfg in quickConfigs"
            :key="cfg.name"
            class="quick-btn"
            @click="applyQuickConfig(cfg)"
          >
            {{ cfg.name }}
          </button>
          <button class="quick-btn reset" @click="resetConfig">重置</button>
        </div>
        
        <div class="form-grid">
          <div class="form-group">
            <label>股票代码</label>
            <input 
              v-model="config.symbol" 
              type="text" 
              placeholder="如: AAPL, NVDA"
              class="form-input"
            />
          </div>
          
          <div class="form-group">
            <label>初始资金 ($)</label>
            <input 
              v-model.number="config.initial_cash" 
              type="number" 
              class="form-input"
            />
          </div>
          
          <div class="form-group">
            <label>开始日期</label>
            <input 
              v-model="config.start" 
              type="date" 
              class="form-input"
            />
          </div>
          
          <div class="form-group">
            <label>结束日期 (可选)</label>
            <input 
              v-model="config.end" 
              type="date" 
              class="form-input"
              placeholder="留空为至今"
            />
          </div>
          
          <div class="form-group">
            <label>手续费率 ({{ (config.commission * 100).toFixed(2) }}%)</label>
            <input 
              v-model.number="config.commission" 
              type="range" 
              min="0" 
              max="0.005" 
              step="0.0001"
              class="form-range"
            />
          </div>
          
          <div class="form-group">
            <label>滑点 ({{ (config.slippage * 100).toFixed(2) }}%)</label>
            <input 
              v-model.number="config.slippage" 
              type="range" 
              min="0" 
              max="0.002" 
              step="0.0001"
              class="form-range"
            />
          </div>
          
          <div class="form-group">
            <label>最小置信度 ({{ config.min_confidence.toFixed(1) }})</label>
            <input 
              v-model.number="config.min_confidence" 
              type="range" 
              min="0" 
              max="1" 
              step="0.1"
              class="form-range"
            />
          </div>
          
          <div class="form-group">
            <label>再平衡阈值 ({{ (config.rebalance_threshold * 100).toFixed(1) }}%)</label>
            <input 
              v-model.number="config.rebalance_threshold" 
              type="range" 
              min="0" 
              max="0.1" 
              step="0.01"
              class="form-range"
            />
          </div>
        </div>
        
        <!-- 流式进度显示 -->
        <div v-if="streaming" class="streaming-progress">
          <div class="progress-header">
            <span class="progress-text">正在回测: 第 {{ progress.current }} 天</span>
            <span v-if="dailyUpdates.length > 0" class="progress-value">
              资金: ${{ dailyUpdates[dailyUpdates.length - 1]?.portfolio_value?.toFixed(0) || 0 }}
            </span>
          </div>
          <div class="progress-bar">
            <div 
              class="progress-fill" 
              :style="{ width: progress.total > 0 ? (progress.current / progress.total * 100) + '%' : '0%' }"
            />
          </div>
        </div>
        
        <div class="action-buttons">
          <button 
            class="btn btn-primary" 
            @click="() => runStreamingBacktest()"
            :disabled="loading"
          >
            <span v-if="streaming">⏳ 实时流式回测中...</span>
            <span v-else>🚀 运行回测</span>
          </button>
          
          <!-- 取消按钮 - 只在回测进行中显示 -->
          <button 
            v-if="streaming || loading"
            class="btn btn-cancel" 
            @click="cancelBacktest"
          >
            ⛔ 取消回测
          </button>
        </div>
        
        <div v-if="error" class="error-message">
          ❌ {{ error }}
        </div>
      </div>
      
      <!-- K线图区域 - 使用实时构建的数据 -->
      <BacktestCandleChart 
        :data="chartData" 
        :loading="chartLoading" 
      />
      
      <div v-if="result" class="results-section">
        <h3 class="section-title">回测结果</h3>
        
        <!-- 概览卡片 -->
        <div class="summary-cards">
          <div class="summary-card profit">
            <div class="card-label">总收益</div>
            <div class="card-value" :class="result.total_return_pct > 0 ? 'positive' : 'negative'">
              {{ formatPercent(result.total_return_pct) }}
            </div>
            <div class="card-sub">{{ result.pnl?.toFixed(0) || 0 }} USD</div>
          </div>
          
          <div class="summary-card">
            <div class="card-label">年化收益</div>
            <div class="card-value" :class="(result.annual_return_pct || 0) > 0 ? 'positive' : 'negative'">
              {{ formatPercent(result.annual_return_pct) }}
            </div>
          </div>
          
          <div class="summary-card">
            <div class="card-label">最大回撤</div>
            <div class="card-value negative">
              {{ formatPercent(result.max_drawdown_pct) }}
            </div>
          </div>
          
          <div class="summary-card">
            <div class="card-label">夏普比率</div>
            <div class="card-value" :class="(result.sharpe_ratio || 0) > 1 ? 'positive' : ''">
              {{ result.sharpe_ratio?.toFixed(3) || '-' }}
            </div>
          </div>
          
          <div class="summary-card">
            <div class="card-label">总交易次数</div>
            <div class="card-value">{{ result.total_trades || 0 }}</div>
            <div class="card-sub">胜率 {{ formatPercent(result.win_rate_pct) }}</div>
          </div>
          
          <div class="summary-card">
            <div class="card-label">最终资金</div>
            <div class="card-value">${{ result.end_value?.toFixed(0) || 0 }}</div>
            <div class="card-sub">初始 ${{ result.start_value?.toFixed(0) || 0 }}</div>
          </div>
        </div>
        
        <!-- 最后信号 -->
        <div v-if="result.last_signal" class="signal-card">
          <div class="signal-header">最后交易信号</div>
          <div class="signal-content">
            <span class="signal-vote" :class="result.last_signal.vote.toLowerCase()">
              {{ result.last_signal.vote }}
            </span>
            <span class="signal-confidence">
              置信度: {{ (result.last_signal.confidence * 100).toFixed(0) }}%
            </span>
            <span class="signal-position">
              目标仓位: {{ (result.last_signal.target_position_pct * 100).toFixed(0) }}%
            </span>
          </div>
        </div>
        
        <!-- 权益曲线简化展示 -->
        <div v-if="dailyUpdates.length > 0" class="equity-curve">
          <h4 class="curve-title">权益曲线 ({{ dailyUpdates.length }}个交易日)</h4>
          <div class="curve-chart">
            <div class="curve-line">
              <div
                v-for="(point, idx) in equityCurveData"
                :key="idx"
                class="curve-point"
                :style="{
                  left: `${(idx / (equityCurveData.length - 1)) * 100}%`,
                  bottom: `${((point.value - Math.min(...equityCurveData.map(d => d.value))) / 
                    (Math.max(...equityCurveData.map(d => d.value)) - Math.min(...equityCurveData.map(d => d.value)) || 1)) * 100}%`
                }"
                :title="`${point.date}: $${point.value.toFixed(2)}`"
              />
            </div>
            <div class="curve-labels">
              <span>{{ equityCurveData[0]?.date }}</span>
              <span>{{ equityCurveData[equityCurveData.length - 1]?.date }}</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 实时更新列表 -->
      <div v-if="dailyUpdates.length > 0" class="updates-section">
        <h3 class="section-title">逐日交易记录 ({{ dailyUpdates.length }}天)</h3>
        <div class="updates-scroll">
          <div 
            v-for="update in dailyUpdates" 
            :key="update.date"
            class="update-item"
          >
            <div class="update-date">{{ update.date }}</div>
            <div class="update-price">${{ update.close_price.toFixed(2) }}</div>
            <div class="update-value">${{ update.portfolio_value.toFixed(0) }}</div>
            <div v-if="update.signal" class="update-signal" :class="update.signal.vote.toLowerCase()">
              {{ update.signal.vote }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.backtest-panel {
  padding: 1.5rem;
  max-width: 1200px;
  margin: 0 auto;
}

.panel-header {
  margin-bottom: 1.5rem;
}

.panel-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.9);
  margin-bottom: 0.5rem;
}

.panel-desc {
  color: rgba(0, 0, 0, 0.5);
  font-size: 0.875rem;
}

.panel-content {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.config-section {
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: 1rem;
  padding: 1.25rem;
}

.section-title {
  font-size: 1rem;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.85);
  margin-bottom: 1rem;
}

.quick-configs {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.quick-label {
  font-size: 0.8125rem;
  color: rgba(0, 0, 0, 0.5);
}

.quick-btn {
  padding: 0.375rem 0.75rem;
  background: rgba(139, 92, 246, 0.08);
  border: 1px solid rgba(139, 92, 246, 0.2);
  border-radius: 0.5rem;
  color: #7c3aed;
  font-size: 0.8125rem;
  cursor: pointer;
  transition: all 0.2s;
}

.quick-btn:hover {
  background: rgba(139, 92, 246, 0.25);
}

.quick-btn.reset {
  background: rgba(0, 0, 0, 0.03);
  border-color: rgba(0, 0, 0, 0.1);
  color: rgba(0, 0, 0, 0.55);
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin-bottom: 1.25rem;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.form-group label {
  font-size: 0.8125rem;
  color: rgba(0, 0, 0, 0.6);
}

.form-input {
  padding: 0.625rem 0.875rem;
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 0.5rem;
  color: rgba(0, 0, 0, 0.85);
  font-size: 0.875rem;
  outline: none;
  transition: all 0.2s;
}

.form-input:focus {
  border-color: rgba(139, 92, 246, 0.5);
}

.form-range {
  -webkit-appearance: none;
  height: 4px;
  background: rgba(0, 0, 0, 0.08);
  border-radius: 2px;
  outline: none;
}

.form-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  background: #a78bfa;
  border-radius: 50%;
  cursor: pointer;
}

.action-buttons {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.btn {
  padding: 0.75rem 1.5rem;
  border-radius: 0.625rem;
  font-size: 0.9375rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
}

.btn-secondary {
  background: rgba(0, 0, 0, 0.05);
  color: rgba(0, 0, 0, 0.8);
  border: 1px solid rgba(0, 0, 0, 0.1);
}

.btn-secondary:hover:not(:disabled) {
  background: rgba(0, 0, 0, 0.08);
}

.btn-cancel {
  background: rgba(239, 68, 68, 0.15);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.3);
  animation: pulse 2s infinite;
}

.btn-cancel:hover {
  background: rgba(239, 68, 68, 0.25);
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.8; }
}

.streaming-progress {
  margin-bottom: 1rem;
  padding: 0.875rem 1rem;
  background: rgba(139, 92, 246, 0.06);
  border: 1px solid rgba(139, 92, 246, 0.2);
  border-radius: 0.75rem;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.625rem;
  font-size: 0.875rem;
}

.progress-text {
  color: #7c3aed;
  font-weight: 500;
}

.progress-value {
  color: rgba(0, 0, 0, 0.65);
}

.progress-bar {
  height: 6px;
  background: rgba(0, 0, 0, 0.08);
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #8b5cf6, #c4b5fd);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.error-message {
  margin-top: 1rem;
  padding: 0.875rem 1rem;
  background: rgba(239, 68, 68, 0.06);
  border: 1px solid rgba(239, 68, 68, 0.2);
  border-radius: 0.5rem;
  color: #dc2626;
  font-size: 0.875rem;
}

.results-section {
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: 1rem;
  padding: 1.25rem;
}

.summary-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.875rem;
  margin-bottom: 1.25rem;
}

.summary-card {
  background: rgba(0, 0, 0, 0.02);
  border: 1px solid rgba(0, 0, 0, 0.05);
  border-radius: 0.75rem;
  padding: 1rem;
  text-align: center;
}

.summary-card.profit {
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(124, 58, 237, 0.05) 100%);
  border-color: rgba(139, 92, 246, 0.2);
}

.card-label {
  font-size: 0.75rem;
  color: rgba(0, 0, 0, 0.5);
  margin-bottom: 0.5rem;
}

.card-value {
  font-size: 1.5rem;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.9);
}

.card-value.positive {
  color: #34d399;
}

.card-value.negative {
  color: #f87171;
}

.card-sub {
  font-size: 0.75rem;
  color: rgba(0, 0, 0, 0.45);
  margin-top: 0.25rem;
}

.signal-card {
  background: rgba(0, 0, 0, 0.02);
  border: 1px solid rgba(0, 0, 0, 0.05);
  border-radius: 0.75rem;
  padding: 1rem;
  margin-bottom: 1.25rem;
}

.signal-header {
  font-size: 0.8125rem;
  color: rgba(0, 0, 0, 0.5);
  margin-bottom: 0.625rem;
}

.signal-content {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.signal-vote {
  padding: 0.375rem 0.875rem;
  border-radius: 9999px;
  font-size: 0.875rem;
  font-weight: 600;
}

.signal-vote.buy,
.signal-vote.strong_buy {
  background: rgba(52, 211, 153, 0.15);
  color: #34d399;
}

.signal-vote.sell,
.signal-vote.strong_sell {
  background: rgba(248, 113, 113, 0.15);
  color: #f87171;
}

.signal-vote.hold {
  background: rgba(251, 191, 36, 0.15);
  color: #fbbf24;
}

.signal-confidence,
.signal-position {
  font-size: 0.875rem;
  color: rgba(0, 0, 0, 0.65);
}

.equity-curve {
  margin-top: 1rem;
}

.curve-title {
  font-size: 0.875rem;
  color: rgba(0, 0, 0, 0.6);
  margin-bottom: 0.75rem;
}

.curve-chart {
  height: 120px;
  background: rgba(0, 0, 0, 0.02);
  border: 1px solid rgba(0, 0, 0, 0.05);
  border-radius: 0.75rem;
  padding: 1rem;
  position: relative;
}

.curve-line {
  position: relative;
  height: 80px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}

.curve-point {
  position: absolute;
  width: 6px;
  height: 6px;
  background: #a78bfa;
  border-radius: 50%;
  transform: translate(-50%, 50%);
  cursor: pointer;
}

.curve-point:hover {
  background: #c4b5fd;
  box-shadow: 0 0 8px rgba(167, 139, 250, 0.5);
}

.curve-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 0.5rem;
  font-size: 0.6875rem;
  color: rgba(0, 0, 0, 0.4);
}

.updates-section {
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: 1rem;
  padding: 1.25rem;
}

.updates-scroll {
  max-height: 300px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.update-item {
  display: grid;
  grid-template-columns: 100px 80px 100px 80px;
  gap: 1rem;
  padding: 0.625rem 0.875rem;
  background: rgba(0, 0, 0, 0.02);
  border-radius: 0.5rem;
  align-items: center;
  font-size: 0.8125rem;
}

.update-date {
  color: rgba(0, 0, 0, 0.55);
}

.update-price,
.update-value {
  color: rgba(0, 0, 0, 0.75);
}

.update-signal {
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.6875rem;
  font-weight: 600;
  text-align: center;
}

.update-signal.buy,
.update-signal.strong_buy {
  background: rgba(52, 211, 153, 0.15);
  color: #34d399;
}

.update-signal.sell,
.update-signal.strong_sell {
  background: rgba(248, 113, 113, 0.15);
  color: #f87171;
}

.update-signal.hold {
  background: rgba(251, 191, 36, 0.15);
  color: #fbbf24;
}
</style>
