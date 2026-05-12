<script setup lang="ts">
import { ref, watch, nextTick } from 'vue';

interface Props {
  modelValue: string;
  loading: boolean;
  needsClarification: boolean;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  'update:modelValue': [value: string];
  send: [];
}>();

const textareaRef = ref<HTMLTextAreaElement | null>(null);
const isFocused = ref(false);

// Auto-resize textarea
watch(() => props.modelValue, async () => {
  await nextTick();
  if (textareaRef.value) {
    textareaRef.value.style.height = 'auto';
    textareaRef.value.style.height = `${Math.min(textareaRef.value.scrollHeight, 120)}px`;
  }
});

const handleInput = (e: Event) => {
  const target = e.target as HTMLTextAreaElement;
  emit('update:modelValue', target.value);
};

const handleKeyDown = (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!props.loading && props.modelValue.trim()) {
      emit('send');
    }
  }
};

const handleSend = () => {
  if (!props.loading && props.modelValue.trim()) {
    emit('send');
  }
};
</script>

<template>
  <div class="input-area">
    <div :class="['input-wrapper', { focused: isFocused, clarification: needsClarification }]">
      <!-- Decorative line -->
      <div class="top-line"></div>
      
      <div class="input-content">
        <!-- Icon -->
        <div class="input-icon" :class="{ clarification: needsClarification }">
          {{ needsClarification ? '💬' : '✨' }}
        </div>

        <!-- Textarea -->
        <textarea
          ref="textareaRef"
          :value="modelValue"
          :placeholder="needsClarification 
            ? '请回复以澄清您的问题...' 
            : '输入股票名称或代码，如：贵州茅台、NVDA、600519...'
          "
          :disabled="loading"
          rows="1"
          class="textarea"
          @input="handleInput"
          @keydown="handleKeyDown"
          @focus="isFocused = true"
          @blur="isFocused = false"
        />

        <!-- Send button -->
        <button
          :class="['send-btn', { active: modelValue.trim() && !loading }]"
          :disabled="loading || !modelValue.trim()"
          @click="handleSend"
        >
          <svg v-if="!loading" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="send-icon">
            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
          </svg>
          <div v-else class="spinner">
            <div class="spinner-dot"></div>
            <div class="spinner-dot"></div>
            <div class="spinner-dot"></div>
          </div>
        </button>
      </div>
    </div>
    
    <!-- Hint -->
    <p class="hint">
      {{ needsClarification ? '请提供更多细节以帮助 AI 更准确地分析' : '按 Enter 发送，Shift + Enter 换行 · AI 分析仅供参考' }}
    </p>
  </div>
</template>

<style scoped>
.input-area {
  padding: 1.25rem 1.5rem;
  background: linear-gradient(to top, rgba(248, 250, 252, 0.98), transparent);
}

.input-wrapper {
  position: relative;
  max-width: 56rem;
  margin: 0 auto;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(24px);
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 1.125rem;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.input-wrapper.focused {
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12), 0 20px 50px -15px rgba(59, 130, 246, 0.15);
  border-color: rgba(59, 130, 246, 0.35);
}

.input-wrapper.clarification {
  border-color: rgba(251, 191, 36, 0.3);
}

.input-wrapper.clarification.focused {
  box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.1), 0 20px 50px -15px rgba(251, 191, 36, 0.15);
  border-color: rgba(251, 191, 36, 0.5);
}

.top-line {
  position: absolute;
  top: 0;
  left: 1.25rem;
  right: 1.25rem;
  height: 1px;
  background: linear-gradient(to right, transparent, rgba(59, 130, 246, 0.4), transparent);
  opacity: 0.5;
}

.input-wrapper.clarification .top-line {
  background: linear-gradient(to right, transparent, rgba(251, 191, 36, 0.4), transparent);
}

.input-content {
  display: flex;
  align-items: flex-end;
  gap: 0.875rem;
  padding: 0.875rem;
}

.input-icon {
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 0.625rem;
  background: rgba(59, 130, 246, 0.08);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 1.125rem;
  transition: all 0.2s;
}

.input-icon.clarification {
  background: rgba(251, 191, 36, 0.12);
}

.textarea {
  flex: 1;
  background: transparent;
  color: #1e293b;
  font-size: 0.9375rem;
  line-height: 1.6;
  resize: none;
  outline: none;
  padding: 0.625rem 0;
  min-height: 2.5rem;
  max-height: 7.5rem;
  border: none;
  font-family: inherit;
}

.textarea::placeholder {
  color: rgba(0, 0, 0, 0.35);
}

.textarea:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.send-btn {
  width: 2.75rem;
  height: 2.75rem;
  border-radius: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.05);
  color: rgba(0, 0, 0, 0.35);
  border: none;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  flex-shrink: 0;
}

.send-btn.active {
  background: linear-gradient(135deg, #3b82f6, #06b6d4);
  color: white;
  box-shadow: 0 8px 25px -8px rgba(59, 130, 246, 0.5);
}

.send-btn.active:hover {
  transform: translateY(-1px) scale(1.02);
  box-shadow: 0 12px 35px -10px rgba(59, 130, 246, 0.6);
}

.send-btn:disabled {
  cursor: not-allowed;
}

.send-icon {
  width: 1.125rem;
  height: 1.125rem;
}

.spinner {
  display: flex;
  gap: 3px;
}

.spinner-dot {
  width: 4px;
  height: 4px;
  background: rgba(255, 255, 255, 0.6);
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
}

.spinner-dot:nth-child(2) { animation-delay: 0.2s; }
.spinner-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-4px); opacity: 1; }
}

.hint {
  text-align: center;
  font-size: 0.75rem;
  color: rgba(0, 0, 0, 0.35);
  margin-top: 0.875rem;
  letter-spacing: 0.02em;
}
</style>
