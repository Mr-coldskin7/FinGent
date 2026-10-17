<script setup lang="ts">
import { API_BASE, apiFetch } from './utils/api';
import { friendlyError } from './utils/errors';
import { ref, watch, nextTick, computed } from 'vue';
import Sidebar from './components/Sidebar.vue';
import Header from './components/Header.vue';
import MessageBubble from './components/MessageBubble.vue';
import LoadingBubble from './components/LoadingBubble.vue';
import EmptyState from './components/EmptyState.vue';
import InputArea from './components/InputArea.vue';
import ThinkingChain from './components/ThinkingChain.vue';
import BacktestPanel from './components/BacktestPanel.vue';
import MarketPanel from './components/MarketPanel.vue';
import SettingsPanel from './components/SettingsPanel.vue';
import EvalPanel from './components/EvalPanel.vue';
import RulesPanel from './components/RulesPanel.vue';
import type { Message, ChatResponse, AnalysisData } from './types/index';
import { agentLabel } from './utils/agents';



const userInput = ref('');
const messages = ref<Message[]>([]);
const loading = ref(false);
const threadId = ref<string | null>(null);
const needsClarification = ref(false);
const messagesEndRef = ref<HTMLElement | null>(null);
const currentView = ref<'chat' | 'backtest' | 'market' | 'settings' | 'eval' | 'rules'>('chat');

const userId = ref(localStorage.getItem('fingent_user_id') || '');
if (!userId.value) {
  userId.value = `user_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  localStorage.setItem('fingent_user_id', userId.value);
}

const feedbackLoading = ref(false);
const reviewLoading = ref(false);
const correctionText = ref('');
const showCorrection = ref(false);
const feedbackMessage = ref('');
const reviewSummary = ref('');
const feedbackError = ref<string | null>(null);

const lastAssistantMessage = computed(() => {
  return [...messages.value].reverse().find((m) => m.role === 'assistant') || null;
});

// 多 Agent 结果：用户可以勾选具体是哪几个 Agent 判断有误/正确（反馈归因）
const feedbackAgents = computed<string[]>(() => {
  const agents = lastAssistantMessage.value?.raw?.selected_agents;
  return Array.isArray(agents) && agents.length > 1 ? agents : [];
});
const selectedAgents = ref<string[]>([]);
watch(lastAssistantMessage, () => { selectedAgents.value = []; });

const feedbackEligible = computed(() => {
  const msg = lastAssistantMessage.value;
  return !!(
    msg &&
    (msg.stock || msg.finalDecision || msg.raw?.stock || msg.content?.length > 0)
  );
});

// 从行情页面跳转到分析
const quickAnalyze = (symbol: string, name: string) => {
  const prompt = `分析一下${name}(${symbol})`;
  userInput.value = prompt;
  currentView.value = 'chat';
  // 延迟发送，确保视图切换完成
  setTimeout(() => {
    sendMessage();
  }, 100);
};

// 思维链步骤
const thinkingSteps = ref<any[]>([]);
const showThinkingChain = ref(false);

// 实时分析进度（SSE /chat-stream 推送，用于加载中的动态状态展示）
interface AgentProgressItem {
  name: string;
  status: 'running' | 'done' | 'error';
  vote?: string;
}
const chatProgress = ref<{
  stage: string;
  agents: AgentProgressItem[];
  startedAt: number;
} | null>(null);

function applyProgressEvent(evt: any) {
  const p = chatProgress.value;
  if (!p) return;
  switch (evt.stage) {
    case 'preprocess':
      p.stage = 'preprocess';
      break;
    case 'chat':
      p.stage = 'chat';
      break;
    case 'clarify':
      p.stage = 'clarify';
      break;
    case 'sector':
      p.stage = 'chat';
      break;
    case 'voting':
      p.stage = 'voting';
      break;
    case 'agent_start': {
      p.stage = 'agents';
      const existing = p.agents.find(a => a.name === evt.agent);
      if (existing) {
        existing.status = 'running';
      } else {
        p.agents.push({ name: evt.agent, status: 'running' });
      }
      break;
    }
    case 'agent_done': {
      const item = p.agents.find(a => a.name === evt.agent);
      if (item) {
        item.status = evt.error ? 'error' : 'done';
        item.vote = evt.vote;
      }
      break;
    }
  }
}

// 生成思维链步骤
function generateThinkingSteps(data: ChatResponse): any[] {
  const steps: any[] = [];
  let stepId = 0;
  
  // 调试信息
  console.log('[ThinkingChain] 生成思维链步骤, 数据:', {
    hasAllToolChains: !!data.all_tool_chains,
    allToolChainsLength: data.all_tool_chains?.length,
    hasToolChain: !!data.tool_chain,
    toolChainLength: data.tool_chain?.length,
    hasDetailedAnalysis: !!data.detailed_analysis,
    hasFinalDecision: !!data.final_decision
  });
  
  // 1. 用户输入
  steps.push({
    id: `step-${stepId++}`,
    type: 'input',
    title: '接收用户查询',
    content: data.input || '',
    status: 'completed'
  });
  
  // 2. 意图解析
  if (data.parse_result) {
    const intent = data.parse_result.intent?.type || 'UNKNOWN';
    const entities = data.parse_result.entities?.symbols?.join(', ') || '';
    steps.push({
      id: `step-${stepId++}`,
      type: 'thinking',
      title: '解析查询意图',
      content: `识别意图: ${intent}${entities ? `，股票: ${entities}` : ''}`,
      details: JSON.stringify(data.parse_result, null, 2),
      status: 'completed'
    });
  }
  
  // 3. 路由决策（优先用后端 route_info，缺失时回退旧的前端推断）
  const intentType = data.parse_result?.intent?.type;
  const routeInfo = data.route_info;
  if (routeInfo) {
    const isChat = routeInfo.mode === 'chat';
    const isSector = routeInfo.mode === 'sector';
    const agentsText = routeInfo.agents.length ? routeInfo.agents.join('、') : '';
    const modeText = isChat ? '直接回复（闲聊）'
      : isSector ? '板块/市场问答（未指定个股）'
      : routeInfo.mode === 'multi' ? `多 Agent 投票（${routeInfo.agents.length} 个：${agentsText}）`
      : routeInfo.mode === 'single' ? `${agentsText} Agent`
      : '澄清确认';
    steps.push({
      id: `step-${stepId++}`,
      type: 'tool_decision',
      title: '路由决策',
      content: isChat
        ? `闲聊/通用问题 → 直接回复｜${routeInfo.reason}`
        : isSector
          ? `板块/市场问题 → 直接回答｜${routeInfo.reason}`
          : `路由到: ${modeText}｜依据[${routeInfo.source}]: ${routeInfo.reason}`,
      details: JSON.stringify(routeInfo, null, 2),
      status: 'completed'
    });
  } else if (intentType === 'CHAT') {
    // 兼容旧后端：闲聊/通用问题走后端 chat_node 直接回复，不进分析流程
    steps.push({
      id: `step-${stepId++}`,
      type: 'tool_decision',
      title: '路由决策',
      content: '闲聊/通用问题 → 直接回复',
      status: 'completed'
    });
  } else {
    const route = intentType === 'TECHNICAL_ANALYSIS' ? 'technical-nerd' :
                  intentType === 'COMPANY_INFO' ? 'morefit' : 'ALL';
    steps.push({
      id: `step-${stepId++}`,
      type: 'tool_decision',
      title: '路由决策',
      content: `路由到: ${route === 'ALL' ? '多Agent投票模式' : route + ' Agent'}`,
      status: 'completed'
    });
  }
  
  // 4. 工具调用 - 优先处理 all_tool_chains（双Agent模式）
  if (data.all_tool_chains && data.all_tool_chains.length > 0) {
    console.log('[ThinkingChain] 处理 all_tool_chains:', data.all_tool_chains);
    
    data.all_tool_chains.forEach((agentChain: any, chainIdx: number) => {
      console.log(`[ThinkingChain] Agent ${chainIdx}:`, agentChain.agent, '步骤数:', agentChain.steps?.length);
      
      // Agent 开始分析
      steps.push({
        id: `step-${stepId++}`,
        type: 'thinking',
        title: `${agentLabel(agentChain.agent)} 开始分析`,
        content: `${agentChain.agent} 启动分析流程，准备调用工具获取数据`,
        agent: agentChain.agent,
        status: 'completed'
      });
      
      // 遍历工具调用步骤 - 保持调用和结果的配对
      if (agentChain.steps && Array.isArray(agentChain.steps)) {
        let toolCallCount = 0;
        for (let i = 0; i < agentChain.steps.length; i++) {
          const tool = agentChain.steps[i];
          
          if (tool.type === 'tool_call') {
            toolCallCount++;
            // 添加工具调用步骤
            steps.push({
              id: `step-${stepId++}`,
              type: 'tool_call',
              title: tool.name || '调用工具',
              content: tool.name || '未知工具',
              details: tool.args ? JSON.stringify(tool.args, null, 2) : '无参数',
              agent: agentChain.agent,
              status: 'completed'
            });
            
            // 查找并添加对应的 tool_result
            for (let j = i + 1; j < agentChain.steps.length; j++) {
              if (agentChain.steps[j].type === 'tool_result') {
                const toolResult = agentChain.steps[j];
                steps.push({
                  id: `step-${stepId++}`,
                  type: 'tool_result',
                  title: '工具返回结果',
                  content: toolResult.content?.slice(0, 300) + (toolResult.content?.length > 300 ? '...' : '') || '',
                  agent: agentChain.agent,
                  status: 'completed'
                });
                break;
              }
            }
          }
        }
        console.log(`[ThinkingChain] Agent ${agentChain.agent} 工具调用数:`, toolCallCount);
      }
      
      // Agent 分析完成
      steps.push({
        id: `step-${stepId++}`,
        type: 'analysis',
        title: `${agentLabel(agentChain.agent)} 分析完成`,
        content: `基于工具返回数据完成分析判断`,
        agent: agentChain.agent,
        status: 'completed'
      });
    });
  } else if (data.tool_chain && data.tool_chain.length > 0) {
    console.log('[ThinkingChain] 处理 tool_chain:', data.tool_chain);
    
    // 单Agent模式
    data.tool_chain.forEach((tool: any) => {
      if (tool.type === 'tool_call') {
        steps.push({
          id: `step-${stepId++}`,
          type: 'tool_call',
          title: tool.name || '调用工具',
          content: tool.name || '未知工具',
          details: tool.args ? JSON.stringify(tool.args, null, 2) : '无参数',
          status: 'completed'
        });
      } else if (tool.type === 'tool_result') {
        steps.push({
          id: `step-${stepId++}`,
          type: 'tool_result',
          title: '工具返回结果',
          content: tool.content?.slice(0, 300) + (tool.content?.length > 300 ? '...' : '') || '',
          status: 'completed'
        });
      }
    });
  } else {
    console.log('[ThinkingChain] 没有工具调用链数据');
  }
  
  // 5. 分析判断
  if (data.detailed_analysis?.decisions) {
    data.detailed_analysis.decisions.forEach((decision: any) => {
      steps.push({
        id: `step-${stepId++}`,
        type: 'analysis',
        title: `${decision.agent || 'Agent'} ${decision.vote}`,
        content: `投票: ${decision.vote}，建议仓位: ${((decision.target_position_pct || 0) * 100).toFixed(0)}%，置信度: ${((decision.confidence || 0) * 100).toFixed(0)}%`,
        status: 'completed'
      });
    });
  }
  
  // 6. 最终结论
  const finalDecision = data.detailed_analysis?.final_decision || data.final_decision;
  if (finalDecision) {
    const isFinalDecisionObject = 'final_vote' in finalDecision;
    const vote = isFinalDecisionObject
      ? finalDecision.final_vote
      : finalDecision.vote;
    const text = isFinalDecisionObject
      ? finalDecision.suggestion || ''
      : finalDecision.reason || '';
    steps.push({
      id: `step-${stepId++}`,
      type: 'conclusion',
      title: '最终决策',
      content: vote ? `${vote} - ${text || '综合分析后的建议'}` : '完成分析',
      status: 'completed'
    });
  }
  
  return steps;
}

// Auto-scroll to bottom
watch([messages, loading], async () => {
  await nextTick();
  messagesEndRef.value?.scrollIntoView({ behavior: 'smooth' });
});

// 从 markdown 代码块中提取 JSON
function extractJsonFromMarkdown(text: string): any | null {
  try {
    // 尝试直接解析
    return JSON.parse(text);
  } catch {
    // 尝试从 markdown 代码块中提取
    const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch && jsonMatch[1]) {
      try {
        return JSON.parse(jsonMatch[1]);
      } catch {
        return null;
      }
    }
    return null;
  }
}

function parseResponse(data: ChatResponse): { 
  content: string; 
  data?: AnalysisData; 
  isClarification?: boolean;
  raw?: any;
  toolChain?: any[];
  agentName?: string;
  stock?: string;
  allToolChains?: any[];
  finalDecision?: any;
} {
  // 保存原始数据供调试
  const raw = { ...data };
  
  // 需要澄清的情况
  if (data.status === 'waiting_for_clarification' || data.parse_result?.status === 'clarification_needed') {
    const clarification = data.parse_result?.clarification;
    let message = '请提供更多信息';
    
    if (typeof clarification === 'object' && clarification?.message) {
      message = clarification.message;
      if (clarification.options?.length) {
        message += '\n\n选项：' + clarification.options.join(' / ');
      }
    } else if (typeof clarification === 'string') {
      message = clarification;
    }
    
    return { content: message, isClarification: true, raw };
  }

  // 双Agent投票模式 - 优先检查
  if (data.all_tool_chains || data.final_decision || data.detailed_analysis) {
    // 使用 detailed_analysis 作为数据源（包含两个Agent的完整决策）
    let analysisData: AnalysisData | undefined = data.detailed_analysis;
    
    return {
      content: data.result || '',
      data: analysisData,
      allToolChains: data.all_tool_chains,
      finalDecision: data.final_decision,
      stock: data.stock,
      raw
    };
  }

  // 单Agent模式 - 解析 result 字段
  if (data.result) {
    const parsed = extractJsonFromMarkdown(data.result);
    if (parsed && (parsed.decisions || parsed.portfolio_suggestion)) {
      return { 
        content: '', 
        data: parsed,
        toolChain: data.tool_chain,
        agentName: data.agent_name,
        stock: data.stock,
        raw 
      };
    }
    // 不是 JSON，当作纯文本
    return { 
      content: data.result, 
      toolChain: data.tool_chain,
      agentName: data.agent_name,
      stock: data.stock,
      raw 
    };
  }

  // 兜底
  return { 
    content: '暂无分析结果', 
    toolChain: data.tool_chain,
    agentName: data.agent_name,
    stock: data.stock,
    raw 
  };
}

const sendMessage = async () => {
  if (!userInput.value.trim() || loading.value) return;

  const input = userInput.value.trim();
  messages.value.push({ role: 'user', content: input, timestamp: new Date() });
  userInput.value = '';
  loading.value = true;
  needsClarification.value = false;
  feedbackMessage.value = '';
  reviewSummary.value = '';
  feedbackError.value = null;
  showCorrection.value = false;
  correctionText.value = '';
  
  // 重置思维链 + 进度
  thinkingSteps.value = [];
  showThinkingChain.value = false;
  chatProgress.value = { stage: 'start', agents: [], startedAt: Date.now() };

  const handleChatResponse = (data: ChatResponse) => {
    threadId.value = data.thread_id;

    if (data.success) {
      const parsed = parseResponse(data);
      needsClarification.value = parsed.isClarification || false;
      
      // 生成思维链步骤
      thinkingSteps.value = generateThinkingSteps(data);
      showThinkingChain.value = thinkingSteps.value.length > 0;
      
      messages.value.push({
        role: 'assistant',
        content: parsed.content,
        data: parsed.data,
        isClarification: parsed.isClarification,
        timestamp: new Date(),
        raw: parsed.raw,
        toolChain: parsed.toolChain,
        agentName: parsed.agentName,
        stock: parsed.stock,
        allToolChains: parsed.allToolChains,
        finalDecision: parsed.finalDecision
      });
    } else {
      messages.value.push({
        role: 'assistant',
        content: '请求失败: ' + (data.error || '未知错误'),
        timestamp: new Date()
      });
    }
  };

  const pushError = (text: string) => {
    messages.value.push({ role: 'assistant', content: text, timestamp: new Date() });
  };

  // 优先走 SSE 流式接口：实时展示分析进度
  const tryStream = async (): Promise<boolean> => {
    let res: Response;
    try {
      res = await apiFetch(`${API_BASE}/api/v1/chat-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_input: input,
          thread_id: threadId.value,
          user_id: userId.value,
        })
      });
    } catch {
      return false; // 网络层失败 → 回退旧接口
    }
    if (!res.ok || !res.body) return false;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // 解析 SSE：事件块以空行分隔，块内 event:/data: 两行
    const handleBlock = (block: string): boolean => {
      const lines = block.split('\n');
      let event = '';
      let data = '';
      for (const line of lines) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (!event || !data) return true; // 继续读
      try {
        const payload = JSON.parse(data);
        if (event === 'progress') {
          applyProgressEvent(payload);
        } else if (event === 'result') {
          handleChatResponse(payload as ChatResponse);
          return false; // 结束
        } else if (event === 'error') {
          pushError('请求失败: ' + (payload.error || '未知错误'));
          return false;
        }
      } catch {
        // 忽略无法解析的块
      }
      return true;
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE 标准行尾是 \r\n（sse_starlette），先归一化再按空行分块
        const text = buffer.replace(/\r\n/g, '\n');
        const blocks = text.split('\n\n');
        buffer = blocks.pop() || '';
        for (const block of blocks) {
          if (!handleBlock(block)) return true;
        }
      }
      if (buffer.trim()) handleBlock(buffer.replace(/\r\n/g, '\n'));
    } finally {
      reader.releaseLock();
    }
    return true;
  };

  try {
    const streamed = await tryStream();
    if (!streamed) {
      // 回退：旧版非流式接口
      const res = await apiFetch(`${API_BASE}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_input: input,
          thread_id: threadId.value,
          user_id: userId.value,
        })
      });
      handleChatResponse(await res.json());
    }
  } catch (err: any) {
    pushError(friendlyError(err));
  } finally {
    loading.value = false;
    chatProgress.value = null;
  }
};

const clearChat = () => {
  messages.value = [];
  threadId.value = null;
  needsClarification.value = false;
  userInput.value = '';
  thinkingSteps.value = [];
  showThinkingChain.value = false;
  feedbackMessage.value = '';
  reviewSummary.value = '';
  feedbackError.value = null;
  correctionText.value = '';
  showCorrection.value = false;
};

const submitFeedback = async (feedbackType: 'agree' | 'disagree' | 'correction') => {
  const msg = lastAssistantMessage.value;
  if (!msg || !msg.stock) {
    feedbackMessage.value = '当前回复中未识别到股票，无法提交反馈。';
    return;
  }
  if (feedbackType === 'correction' && !correctionText.value.trim()) {
    feedbackMessage.value = '请填写纠正规则内容后再提交。';
    return;
  }

  feedbackLoading.value = true;
  feedbackMessage.value = '';
  feedbackError.value = null;

  try {
    const payload: any = {
      session_id: threadId.value || '',
      stock_symbol: msg.stock,
      agent_name: msg.agentName || undefined,
      feedback: feedbackType,
      user_id: userId.value,
    };
    if (feedbackAgents.value.length > 0) {
      payload.agent_names = selectedAgents.value.length ? [...selectedAgents.value] : undefined;
      delete payload.agent_name;
    }
    if (feedbackType === 'correction') {
      payload.rule_text = correctionText.value.trim();
    }

    const res = await apiFetch(`${API_BASE}/api/v1/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.success) {
      feedbackMessage.value =
        data.message ||
        (feedbackType === 'agree'
          ? '已记录赞同反馈。'
          : feedbackType === 'disagree'
          ? '已记录不同意反馈。'
          : '已记录纠正反馈，并保存用户规则。');
      if (feedbackType === 'correction') {
        showCorrection.value = false;
      }
    } else {
      feedbackMessage.value = `反馈失败：${data.error || '未知错误'}`;
    }
  } catch (error: any) {
    feedbackError.value = friendlyError(error, '网络请求失败');
  } finally {
    feedbackLoading.value = false;
  }
};

const VOTE_CN: Record<string, string> = {
  STRONG_BUY: '强烈买入', BUY: '买入', HOLD: '持有', REDUCE: '减仓', SELL: '卖出', STRONG_SELL: '强烈卖出',
};
function votesLine(review: any): string {
  const votes = review.agent_votes || {};
  const names = Object.keys(votes);
  if (!names.length) return '';
  const dissent = review.dissenting_agents || [];
  const list = names.map((n) => `${n} ${VOTE_CN[votes[n]] || votes[n]}`).join('，');
  return `投票：${list}\n` + (dissent.length ? `与最终决策方向不同：${dissent.join('、')}\n` : '');
}
function outcomesLine(review: any): string {
  const rows = review.outcomes || [];
  if (!rows.length) return '结果验证：尚未到验证时间（需要分析后若干个交易日）\n';
  const lines = rows.map((r: any) => {
    const verdict = r.correct === 1 ? '判断正确' : r.correct === 0 ? '判断偏离' : '无法判定';
    const excess = r.excess_return != null ? `，超额收益 ${(r.excess_return * 100).toFixed(1)}%` : '';
    return `${r.horizon_days} 日 · ${r.agent}（${VOTE_CN[r.vote] || r.vote}）：${verdict}${excess}`;
  });
  return `结果验证：\n${lines.join('\n')}\n`;
}

const submitReview = async () => {
  const msg = lastAssistantMessage.value;
  if (!msg || !msg.stock) {
    reviewSummary.value = '当前回复中未识别到股票，无法执行复盘。';
    return;
  }

  reviewLoading.value = true;
  reviewSummary.value = '';
  feedbackError.value = null;

  try {
    const res = await apiFetch(`${API_BASE}/api/v1/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: threadId.value || undefined,
        stock_symbol: msg.stock,
        user_id: userId.value,
      }),
    });
    const data = await res.json();
    if (data.success) {
      const review = data.review;
      const historyLines = review.recent_history
        .map(
          (item: any) =>
            `${item.created_at.slice(0, 10)} → ${item.final_decision}` +
            (item.user_feedback ? ` (${item.user_feedback})` : ''),
        )
        .join('\n');
      reviewSummary.value =
        `复盘结果：\n` +
        `股票：${review.stock_symbol}\n` +
        `当前决策：${review.current_decision}\n` +
        `${review.reasoning_summary ? `理由：${review.reasoning_summary}\n` : ''}` +
        `${review.user_feedback ? `用户反馈：${review.user_feedback}\n` : ''}` +
        `${review.inconsistency_warning ? `警告：${review.inconsistency_warning}\n` : ''}` +
        `${votesLine(review)}` +
        `${outcomesLine(review)}` +
        `历史回顾：\n${historyLines}`;
    } else {
      reviewSummary.value = `复盘失败：${data.error || '未知错误'}`;
    }
  } catch (error: any) {
    reviewSummary.value = friendlyError(error, '请求失败');
  } finally {
    reviewLoading.value = false;
  }
};

const quickInput = (text: string) => {
  userInput.value = text;
};
</script>

<template>
  <div class="app">
<!-- Sidebar -->
    <Sidebar 
      @new-chat="clearChat" 
      @quick-input="quickInput"
      @switch-view="(view) => currentView = view"
      :current-view="currentView"
    />

    <!-- Main Content -->
    <main class="main">
      <Header :thread-id="threadId" />

      <!-- 聊天视图 -->
      <template v-if="currentView === 'chat'">
        <!-- Messages Area -->
        <div class="messages">
          <EmptyState 
            v-if="messages.length === 0" 
            @quick-input="quickInput" 
            @send="sendMessage" 
          />
          <div v-else class="messages-container">
            <template v-for="(msg, idx) in messages" :key="idx">
              <!-- 用户消息 -->
              <MessageBubble 
                v-if="msg.role === 'user'"
                :message="msg" 
                :index="idx" 
              />
              
              <!-- AI消息：先显示思维链，再显示回复 -->
              <template v-else>
                <!-- 思维链（只在最后一条AI消息显示） -->
                <div v-if="idx === messages.length - 1 && thinkingSteps.length > 0" 
                     class="thinking-panel">
                  <div class="thinking-panel-header" @click="showThinkingChain = !showThinkingChain">
                    <div class="thinking-toggle">
                      <span class="toggle-icon"></span>
                      <span class="toggle-text">Agent 思维链</span>
                      <span class="step-count">{{ thinkingSteps.length }} 步</span>
                    </div>
                    <div class="expand-hint">
                      <span class="hint-text">{{ showThinkingChain ? '点击收起' : '点击展开' }}</span>
                      <svg class="arrow-icon" :class="{ 'expanded': showThinkingChain }" 
                           viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M19 9l-7 7-7-7"/>
                      </svg>
                    </div>
                  </div>
                  <Transition name="slide-fade">
                    <ThinkingChain 
                      v-show="showThinkingChain"
                      :steps="thinkingSteps" 
                      :is-active="loading"
                    />
                  </Transition>
                </div>
                
                <!-- AI回复 -->
                <MessageBubble 
                  :message="msg" 
                  :index="idx" 
                />
              </template>
            </template>
            
            <LoadingBubble v-if="loading" :progress="chatProgress" />
            <div ref="messagesEndRef" />
          </div>

          <div v-if="feedbackEligible" class="feedback-panel">
            <div class="feedback-panel-header">
              <span>对本次分析进行反馈 / 复盘</span>
              <span class="feedback-user">用户ID: {{ userId }}</span>
            </div>
            <div v-if="feedbackAgents.length" class="feedback-agents">
              <span class="feedback-agents-hint">勾选需要反馈的 Agent（不勾选则只记录历史，不改权重）：</span>
              <label v-for="a in feedbackAgents" :key="a" class="feedback-agent-option">
                <input type="checkbox" :value="a" v-model="selectedAgents" />
                {{ agentLabel(a) }}
              </label>
            </div>
            <div class="feedback-actions">
              <button @click="submitFeedback('agree')" :disabled="feedbackLoading || reviewLoading">
                同意
              </button>
              <button @click="submitFeedback('disagree')" :disabled="feedbackLoading || reviewLoading">
                不同意
              </button>
              <button @click="showCorrection = !showCorrection" :disabled="feedbackLoading || reviewLoading">
                纠正规则
              </button>
              <button @click="submitReview" :disabled="reviewLoading || feedbackLoading">
                复盘历史
              </button>
            </div>
            <div v-if="showCorrection" class="feedback-correction">
              <textarea
                v-model="correctionText"
                placeholder="请输入纠正内容，例如：不要过度解读单日放量"
                rows="3"
              />
              <button
                class="submit-correction"
                @click="submitFeedback('correction')"
                :disabled="feedbackLoading || !correctionText.trim()"
              >
                提交纠正
              </button>
            </div>
            <div class="feedback-status">
              <div v-if="feedbackMessage" class="feedback-message">{{ feedbackMessage }}</div>
              <div v-if="feedbackError" class="feedback-error">{{ feedbackError }}</div>
              <div v-if="reviewSummary" class="review-summary"><pre>{{ reviewSummary }}</pre></div>
            </div>
          </div>

        </div>

        <!-- Input Area -->
        <InputArea
          v-model="userInput"
          :loading="loading"
          :needs-clarification="needsClarification"
          @send="sendMessage"
        />
      </template>
      
      <!-- 行情视图 -->
      <template v-else-if="currentView === 'market'">
        <MarketPanel @analyze="quickAnalyze" />
      </template>

      <!-- 设置视图 -->
      <template v-else-if="currentView === 'settings'">
        <div class="settings-view">
          <SettingsPanel />
        </div>
      </template>

      <!-- 评测中心视图 -->
      <template v-else-if="currentView === 'eval'">
        <EvalPanel />
      </template>

      <template v-else-if="currentView === 'rules'">
        <RulesPanel :user-id="userId" />
      </template>

      <!-- 回测视图 -->
      <template v-else>
        <div class="backtest-view">
          <BacktestPanel />
        </div>
      </template>
    </main>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: #121214;
  color: #e4e4e7;
  overflow: hidden;
}

.app {
  height: 100vh;
  width: 100vw;
  background: #121214;
  display: flex;
  overflow: hidden;
  position: relative;
}

.background {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
}

.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(128px);
}

.orb-1 {
  top: 0;
  left: 25%;
  width: 24rem;
  height: 24rem;
  background: rgba(59, 130, 246, 0.08);
}

.orb-2 {
  bottom: 0;
  right: 25%;
  width: 24rem;
  height: 24rem;
  background: rgba(6, 182, 212, 0.08);
}

.orb-3 {
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 50rem;
  height: 50rem;
  background: rgba(16, 185, 129, 0.04);
  filter: blur(200px);
}

.grid-pattern {
  position: absolute;
  inset: 0;
  opacity: 0.03;
  background-image: 
    linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
  background-size: 50px 50px;
}

.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
  z-index: 10;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
}

.messages::-webkit-scrollbar {
  width: 6px;
}

.messages::-webkit-scrollbar-track {
  background: transparent;
}

.messages::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.1);
  border-radius: 3px;
}

.messages::-webkit-scrollbar-thumb:hover {
  background: rgba(255,255,255,0.2);
}

/* 思维链面板 */
.thinking-panel {
  margin: 0.75rem 0;
  background: rgba(11,11,12,0.92);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 1rem;
  overflow: hidden;
}

.thinking-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.875rem 1rem;
  cursor: pointer;
  transition: background 0.2s;
}

.thinking-panel-header:hover {
  background: rgba(255,255,255,0.02);
}

.thinking-toggle {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.toggle-icon {
  font-size: 1.125rem;
}

.toggle-text {
  font-size: 0.9375rem;
  font-weight: 600;
  color: rgba(255,255,255,0.8);
}

.step-count {
  font-size: 0.6875rem;
  padding: 0.125rem 0.5rem;
  background: rgba(245,165,36,0.2);
  color: #f5a524;
  border-radius: 9999px;
  font-weight: 600;
}

.expand-hint {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.hint-text {
  font-size: 0.75rem;
  color: rgba(255,255,255,0.4);
}

.arrow-icon {
  width: 1rem;
  height: 1rem;
  color: rgba(255,255,255,0.4);
  transition: transform 0.3s ease;
}

.arrow-icon.expanded {
  transform: rotate(180deg);
}

/* 过渡动画 */
.slide-fade-enter-active {
  transition: all 0.3s ease-out;
}

.slide-fade-leave-active {
  transition: all 0.2s ease-in;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

.messages-container {
  max-width: 56rem;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.feedback-panel {
  margin: 1rem auto 0;
  max-width: 56rem;
  background: #0b0b0c;
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 1rem;
  padding: 1rem;
  box-shadow: 0 20px 45px rgba(15, 23, 42, 0.04);
}

.feedback-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.75rem;
  color: #d4d4d8;
  font-weight: 600;
}

.feedback-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.feedback-actions button {
  border: 1px solid #2a2a30;
  padding: 0.6rem 1.1rem;
  border-radius: 6px;
  background: #121214;
  color: #d4d4d8;
  font-size: 0.85rem;
  cursor: pointer;
  transition: transform 0.15s ease, opacity 0.15s ease;
}

.feedback-actions button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.feedback-actions button:hover:not(:disabled) {
  border-color: #f5a524;
  color: #f5a524;
}

.feedback-agents {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px;
  margin: 6px 0 10px; font-size: 13px; color: #8b8f98;
}
.feedback-agent-option { display: inline-flex; align-items: center; gap: 4px; color: #e4e4e7; cursor: pointer; }
.feedback-agents-hint { flex-basis: 100%; }

.feedback-correction {
  margin-bottom: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.feedback-correction textarea {
  width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 0.75rem;
  padding: 0.9rem 1rem;
  resize: vertical;
  font-size: 0.95rem;
  font-family: inherit;
}

.submit-correction {
  align-self: flex-end;
  padding: 0.75rem 1rem;
  border-radius: 9999px;
  background: #10b981;
}

.feedback-status {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.feedback-message,
.review-summary {
  color: #f4f4f5;
  font-size: 0.95rem;
  white-space: pre-wrap;
}

.feedback-error {
  color: #ff6b6b;
  font-size: 0.95rem;
}

.feedback-user {
  color: #8b8f98;
  font-size: 0.85rem;
}

.backtest-view {
  flex: 1;
  overflow-y: auto;
  padding: 0;
}

.backtest-view::-webkit-scrollbar {
  width: 6px;
}

.backtest-view::-webkit-scrollbar-track {
  background: transparent;
}

.backtest-view::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.1);
  border-radius: 3px;
}

.settings-view {
  flex: 1;
  overflow-y: auto;
  padding: 2rem;
  background: #121214;
}
</style>
