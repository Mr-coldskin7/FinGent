// 回测相关类型定义

export interface BacktestRequest {
  symbol: string;
  start: string;
  end?: string;
  initial_cash: number;
  commission: number;
  slippage: number;
  min_confidence: number;
  rebalance_threshold: number;
  temperature: number;
  session_id?: string;
  interval: 'daily' | 'weekly' | 'monthly' | 'annually';
}

export interface BacktestResult {
  symbol: string;
  start_value: number;
  end_value: number;
  pnl: number;
  total_return_pct: number;
  annual_return_pct?: number;
  max_drawdown_pct?: number;
  sharpe_ratio?: number;
  volatility_ann_pct?: number;
  total_trades?: number;
  win_trades?: number;
  loss_trades?: number;
  win_rate_pct?: number;
  last_signal?: {
    vote: string;
    confidence: number;
    target_position_pct: number;
  };
  period?: {
    start: string;
    end: string;
  };
  analyzers?: {
    drawdown?: any;
    returns?: any;
    sharpe?: any;
    trades?: any;
  };
}

export interface BacktestResponse {
  success: boolean;
  result?: BacktestResult;
  symbol?: string;
  period?: { start: string; end: string };
  error?: string;
}

export interface DailyUpdate {
  date: string;
  cash: number;
  portfolio_value: number;
  position_size: number;
  avg_cost: number;
  // OHLCV 数据
  open_price?: number;
  high_price?: number;
  low_price?: number;
  close_price: number;
  volume?: number;
  signal?: {
    vote: string;
    confidence: number;
    target_position_pct: number;
    reason?: string;
  };
  day_number: number;
}

export interface BacktestStreamEvent {
  event: 'start' | 'daily_update' | 'final_result' | 'error';
  data: string;
}

export const DEFAULT_BACKTEST_CONFIG: BacktestRequest = {
  symbol: 'AAPL',
  start: '2024-01-01',
  end: '',
  initial_cash: 10000,
  commission: 0.001,
  slippage: 0.0005,
  min_confidence: 0.0,
  rebalance_threshold: 0.02,
  temperature: 0.0,
  interval: 'daily',
};
