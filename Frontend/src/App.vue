<script setup lang="ts">
import { ref, watch, nextTick, computed } from 'vue';
import Sidebar from './components/Sidebar.vue';
import Header from './components/Header.vue';
import MessageBubble from './components/MessageBubble.vue';
import LoadingBubble from './components/LoadingBubble.vue';
import EmptyState from './components/EmptyState.vue';
import InputArea from './components/InputArea.vue';
import ThinkingChain from './components/ThinkingChain.vue';
import BacktestPanel from './components/BacktestPanel.vue';
import type { Message, ChatResponse, AnalysisData } from './types/index';

const API_BASE = 'http://localhost:8000';

const userInput = ref('');
const messages = ref<Message[]>([]);
const loading = ref(false);
const threadId = ref<string | null>(null);
const needsClarification = ref(false);
const messagesEndRef = ref<HTMLElement | null>(null);
const currentView = ref<'chat' | 'backtest'>('chat');

// 思维链步骤
const thinkingSteps = ref<any[]>([]);
const showThinkingChain = ref(false);

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
  
  // 3. 路由决策
  const route = data.parse_result?.intent?.type === 'TECHNICAL_ANALYSIS' ? 'TECHNICAL_NERD' :
                data.parse_result?.intent?.type === 'COMPANY_INFO' ? 'Morefit' : 'ALL';
  steps.push({
    id: `step-${stepId++}`,
    type: 'tool_decision',
    title: '路由决策',
    content: `路由到: ${route === 'ALL' ? '双Agent投票模式' : route + ' Agent'}`,
    status: 'completed'
  });
  
  // 4. 工具调用 - 优先处理 all_tool_chains（双Agent模式）
  if (data.all_tool_chains && data.all_tool_chains.length > 0) {
    console.log('[ThinkingChain] 处理 all_tool_chains:', data.all_tool_chains);
    
    data.all_tool_chains.forEach((agentChain: any, chainIdx: number) => {
      console.log(`[ThinkingChain] Agent ${chainIdx}:`, agentChain.agent, '步骤数:', agentChain.steps?.length);
      
      // Agent 开始分析
      steps.push({
        id: `step-${stepId++}`,
        type: 'thinking',
        title: `${agentChain.agent} 开始分析`,
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
        title: `${agentChain.agent} 分析完成`,
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
  if (data.detailed_analysis?.final_decision || data.final_decision) {
    const final = data.detailed_analysis?.final_decision || data.final_decision;
    steps.push({
      id: `step-${stepId++}`,
      type: 'conclusion',
      title: '最终决策',
      content: final?.vote ? `${final.vote} - ${final.reason || '综合分析后的建议'}` : '完成分析',
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
    if (jsonMatch) {
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
  
  // 重置思维链
  thinkingSteps.value = [];
  showThinkingChain.value = false;

  try {
    const res = await fetch(`${API_BASE}/api/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_input: input,
        thread_id: threadId.value
      })
    });

    const data: ChatResponse = await res.json();
    threadId.value = data.thread_id;

    // 调试日志
    console.log('[API Response] 完整响应:', data);
    console.log('[API Response] all_tool_chains:', data.all_tool_chains);
    console.log('[API Response] tool_chain:', data.tool_chain);

    if (data.success) {
      const parsed = parseResponse(data);
      needsClarification.value = parsed.isClarification || false;
      
      // 生成思维链步骤
      thinkingSteps.value = generateThinkingSteps(data);
      showThinkingChain.value = thinkingSteps.value.length > 0;
      
      console.log('[ThinkingChain] 生成的步骤数:', thinkingSteps.value.length);
      console.log('[ThinkingChain] 步骤详情:', thinkingSteps.value.map(s => ({ type: s.type, title: s.title })));
      
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
  } catch (err: any) {
    messages.value.push({
      role: 'assistant',
      content: '网络错误: ' + err.message,
      timestamp: new Date()
    });
  } finally {
    loading.value = false;
  }
};

const clearChat = () => {
  messages.value = [];
  threadId.value = null;
  needsClarification.value = false;
  userInput.value = '';
  thinkingSteps.value = [];
  showThinkingChain.value = false;
};

const quickInput = (text: string) => {
  userInput.value = text;
};
</script>

<template>
  <div class="app">
    <!-- Background effects -->
    <div class="background">
      <div class="orb orb-1" />
      <div class="orb orb-2" />
      <div class="orb orb-3" />
      <div class="grid-pattern" />
    </div>

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
                      <span class="toggle-icon">🧠</span>
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
            
            <LoadingBubble v-if="loading" />
            <div ref="messagesEndRef" />
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
  background: #020617;
  color: white;
  overflow: hidden;
}

.app {
  height: 100vh;
  width: 100vw;
  background: #020617;
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
  background: rgba(59, 130, 246, 0.1);
}

.orb-2 {
  bottom: 0;
  right: 25%;
  width: 24rem;
  height: 24rem;
  background: rgba(6, 182, 212, 0.1);
}

.orb-3 {
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 50rem;
  height: 50rem;
  background: rgba(16, 185, 129, 0.05);
  filter: blur(200px);
}

.grid-pattern {
  position: absolute;
  inset: 0;
  opacity: 0.02;
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
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

.messages::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.2);
}

/* 思维链面板 */
.thinking-panel {
  margin: 0.75rem 0;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
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
  background: rgba(255, 255, 255, 0.04);
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
  color: rgba(255, 255, 255, 0.9);
}

.step-count {
  font-size: 0.6875rem;
  padding: 0.125rem 0.5rem;
  background: rgba(139, 92, 246, 0.2);
  color: #a78bfa;
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
  color: rgba(255, 255, 255, 0.4);
}

.arrow-icon {
  width: 1rem;
  height: 1rem;
  color: rgba(255, 255, 255, 0.4);
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
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}
</style>
