<script setup lang="ts">
import { ref, onMounted, watch, onUnmounted, shallowRef, computed } from 'vue';
import { createChart, ColorType, CandlestickSeries, HistogramSeries, createSeriesMarkers } from 'lightweight-charts';

interface CandleData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TradeMarker {
  time: string;
  type: 'buy' | 'sell';
  price: number;
  size: number;
}

interface ChartStatistics {
  initial_value: number;
  final_value: number;
  return_pct: number;
  total_trades: number;
  buy_count: number;
  sell_count: number;
}

interface ChartData {
  symbol: string;
  candles: CandleData[];
  trade_markers: TradeMarker[];
  statistics: ChartStatistics;
}

const props = defineProps<{
  data?: ChartData | null;
  loading?: boolean;
}>();

const chartDiv = ref<HTMLDivElement | null>(null);
const chart = shallowRef<any>(null);
const candleSeries = shallowRef<any>(null);
const volumeSeries = shallowRef<any>(null);
const seriesMarkers = shallowRef<any>(null);

// 是否有数据
const hasData = computed(() => {
  return props.data && props.data.candles && props.data.candles.length > 0;
});

// 初始化图表
const initChart = () => {
  if (!chartDiv.value) {
    console.log('[Chart] chartDiv 不存在');
    return false;
  }

  console.log('[Chart] 初始化图表...');

  // 如果已有图表，先移除
  if (chart.value) {
    chart.value.remove();
  }

  const rect = chartDiv.value.getBoundingClientRect();
  console.log('[Chart] 容器尺寸:', rect.width, 'x', rect.height);
  
  if (rect.width === 0 || rect.height === 0) {
    console.log('[Chart] 容器尺寸为0，稍后重试');
    return false;
  }

  // 创建图表
  const newChart = createChart(chartDiv.value, {
    width: rect.width,
    height: rect.height,
    layout: {
      background: { type: ColorType.Solid, color: '#ffffff' },
      textColor: 'rgba(0, 0, 0, 0.6)',
    },
    grid: {
      vertLines: { color: 'rgba(0, 0, 0, 0.06)' },
      horzLines: { color: 'rgba(0, 0, 0, 0.06)' },
    },
    timeScale: {
      borderColor: 'rgba(0, 0, 0, 0.1)',
      timeVisible: true,
    },
  });

  chart.value = newChart;

  // 创建K线系列 - v5 API
  try {
    candleSeries.value = newChart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    });
    console.log('[Chart] K线系列创建成功');
  } catch (e) {
    console.error('[Chart] 创建K线系列失败:', e);
    return false;
  }

  // 创建成交量系列 - v5 API
  try {
    volumeSeries.value = newChart.addSeries(HistogramSeries, {
      color: '#3b82f6',
      priceScaleId: 'volume',
    });
    volumeSeries.value.priceScale().applyOptions({
      scaleMargins: { top: 0.7, bottom: 0 },
    });
    console.log('[Chart] 成交量系列创建成功');
  } catch (e) {
    console.error('[Chart] 创建成交量系列失败:', e);
  }

  console.log('[Chart] 图表初始化完成');
  return true;
};

// 更新图表数据
const updateChart = () => {
  console.log('[Chart] 更新图表, hasData:', hasData.value);
  
  if (!hasData.value) {
    console.log('[Chart] 无数据，跳过');
    return;
  }

  // 如果图表未初始化，先初始化
  if (!chart.value || !candleSeries.value) {
    console.log('[Chart] 图表未初始化，先初始化');
    const success = initChart();
    if (!success) {
      console.log('[Chart] 初始化失败，稍后重试');
      setTimeout(updateChart, 200);
      return;
    }
  }

  const data = props.data!;
  console.log(`[Chart] 设置数据: ${data.candles.length} 根K线`);

  // 转换数据格式
  const candleData = data.candles.map(c => ({
    time: c.time,
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
  }));

  const volumeData = data.candles.map(c => ({
    time: c.time,
    value: c.volume,
    color: c.close >= c.open ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)',
  }));

  // 设置数据
  try {
    candleSeries.value.setData(candleData);
    console.log('[Chart] K线数据设置成功');
  } catch (e) {
    console.error('[Chart] 设置K线数据失败:', e);
  }
  
  if (volumeSeries.value) {
    try {
      volumeSeries.value.setData(volumeData);
      console.log('[Chart] 成交量数据设置成功');
    } catch (e) {
      console.error('[Chart] 设置成交量数据失败:', e);
    }
  }

  // 添加买卖点标记 - v5 API 使用 createSeriesMarkers
  if (data.trade_markers?.length > 0) {
    const markers = data.trade_markers.map(trade => ({
      time: trade.time,
      position: (trade.type === 'buy' ? 'belowBar' : 'aboveBar') as 'belowBar' | 'aboveBar',
      color: trade.type === 'buy' ? '#22c55e' : '#ef4444',
      shape: (trade.type === 'buy' ? 'arrowUp' : 'arrowDown') as 'arrowUp' | 'arrowDown',
      text: trade.reason 
        ? `${trade.type === 'buy' ? '买入' : '卖出'} ${trade.size}股\n${trade.reason.substring(0, 30)}...`
        : `${trade.type === 'buy' ? '买入' : '卖出'} ${trade.size}股`,
      size: 2,
    }));
    
    try {
      // v5: 使用 createSeriesMarkers 替代 setMarkers
      if (!seriesMarkers.value) {
        seriesMarkers.value = createSeriesMarkers(candleSeries.value, markers);
      } else {
        seriesMarkers.value.setMarkers(markers);
      }
      console.log('[Chart] 标记设置成功:', markers.length, '个标记');
    } catch (e) {
      console.error('[Chart] 设置标记失败:', e);
    }
  } else {
    // 清空标记
    if (seriesMarkers.value) {
      seriesMarkers.value.setMarkers([]);
    }
  }

  // 自适应时间范围
  try {
    chart.value.timeScale().fitContent();
    console.log('[Chart] 更新完成');
  } catch (e) {
    console.error('[Chart] fitContent 失败:', e);
  }
};

// 监听数据变化
watch(() => props.data, (newData) => {
  console.log('[Chart] 数据变化:', newData ? `${newData.candles?.length || 0} 根K线` : 'null');
  if (newData?.candles?.length) {
    updateChart();
  }
}, { deep: true });

onMounted(() => {
  console.log('[Chart] 组件挂载');
  // 延迟初始化，确保 DOM 已渲染
  setTimeout(() => {
    if (props.data?.candles?.length) {
      updateChart();
    }
  }, 100);
});

onUnmounted(() => {
  if (chart.value) {
    chart.value.remove();
  }
});

// 格式化数字
const formatNumber = (num: number | undefined, decimals = 2) => {
  return num?.toFixed(decimals) || '0.00';
};
</script>

<template>
  <div class="candle-chart">
    <div class="chart-header">
      <h3 class="chart-title">
        📊 {{ data?.symbol || 'Stock' }} K线图
        <span v-if="hasData" class="candle-count">
          {{ data?.candles?.length }}根K线
        </span>
      </h3>
      <div v-if="data?.statistics" class="chart-stats">
        <span class="stat-item">
          收益率: 
          <span :class="(data.statistics.return_pct || 0) >= 0 ? 'positive' : 'negative'">
            {{ (data.statistics.return_pct || 0) > 0 ? '+' : '' }}{{ formatNumber(data.statistics.return_pct) }}%
          </span>
        </span>
        <span class="stat-item">
          交易: {{ data.statistics.total_trades || 0 }}次
        </span>
      </div>
    </div>
    
    <div class="chart-wrapper">
      <div v-if="loading" class="chart-overlay">
        <div class="loading-spinner"></div>
        <span>加载中...</span>
      </div>
      <div v-else-if="!hasData" class="chart-overlay">
        <span>暂无K线数据</span>
      </div>
      <div ref="chartDiv" class="chart-container"></div>
    </div>
  </div>
</template>

<style scoped>
.candle-chart {
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: 1rem;
  padding: 1.25rem;
  margin-bottom: 1.5rem;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.chart-title {
  font-size: 1rem;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.9);
  margin: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.candle-count {
  font-size: 0.75rem;
  color: rgba(0, 0, 0, 0.5);
  background: rgba(0, 0, 0, 0.05);
  padding: 0.125rem 0.5rem;
  border-radius: 9999px;
}

.chart-stats {
  display: flex;
  gap: 1.5rem;
  align-items: center;
}

.stat-item {
  font-size: 0.8125rem;
  color: rgba(0, 0, 0, 0.6);
}

.positive { color: #22c55e; }
.negative { color: #ef4444; }

.chart-wrapper {
  width: 100%;
  height: 450px;
  position: relative;
  background: rgba(0, 0, 0, 0.03);
  border-radius: 0.5rem;
}

.chart-container {
  width: 100%;
  height: 100%;
}

.chart-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  color: rgba(0, 0, 0, 0.4);
  font-size: 0.875rem;
  z-index: 10;
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(139, 92, 246, 0.15);
  border-top-color: #7c3aed;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
