<script setup lang="ts">
import { computed } from 'vue';
import { marked } from 'marked';

interface Props {
  content: string;
}

const props = defineProps<Props>();

// 配置 marked
marked.setOptions({
  breaks: true,
  gfm: true,
});

const renderedHtml = computed(() => {
  return marked.parse(props.content);
});
</script>

<template>
  <div class="markdown-body" v-html="renderedHtml"></div>
</template>

<style scoped>
.markdown-body {
  color: rgba(255, 255, 255, 0.9);
  font-size: 0.9375rem;
  line-height: 1.8;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.markdown-body :deep(h4) {
  color: white;
  margin-top: 1.5rem;
  margin-bottom: 0.75rem;
  font-weight: 600;
}

.markdown-body :deep(h1) {
  font-size: 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding-bottom: 0.5rem;
}

.markdown-body :deep(h2) {
  font-size: 1.25rem;
}

.markdown-body :deep(h3) {
  font-size: 1.125rem;
}

.markdown-body :deep(h4) {
  font-size: 1rem;
}

.markdown-body :deep(p) {
  margin-bottom: 1rem;
}

.markdown-body :deep(strong) {
  color: #60a5fa;
  font-weight: 600;
}

.markdown-body :deep(em) {
  color: #fbbf24;
  font-style: italic;
}

.markdown-body :deep(hr) {
  border: none;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  margin: 1.5rem 0;
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  margin-left: 1.5rem;
  margin-bottom: 1rem;
}

.markdown-body :deep(li) {
  margin-bottom: 0.5rem;
}

.markdown-body :deep(code) {
  background: rgba(255, 255, 255, 0.08);
  padding: 0.2rem 0.4rem;
  border-radius: 0.25rem;
  font-family: monospace;
  font-size: 0.875em;
  color: #fbbf24;
}

.markdown-body :deep(pre) {
  background: rgba(0, 0, 0, 0.3);
  padding: 1rem;
  border-radius: 0.5rem;
  overflow-x: auto;
  margin-bottom: 1rem;
}

.markdown-body :deep(pre code) {
  background: none;
  padding: 0;
  color: rgba(255, 255, 255, 0.8);
}

.markdown-body :deep(blockquote) {
  border-left: 3px solid #60a5fa;
  padding-left: 1rem;
  margin-left: 0;
  margin-bottom: 1rem;
  color: rgba(255, 255, 255, 0.7);
}

.markdown-body :deep(a) {
  color: #60a5fa;
  text-decoration: none;
}

.markdown-body :deep(a:hover) {
  text-decoration: underline;
}

.markdown-body :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1rem;
}

.markdown-body :deep(th),
.markdown-body :deep(td) {
  padding: 0.625rem;
  text-align: left;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.markdown-body :deep(th) {
  color: rgba(255, 255, 255, 0.6);
  font-weight: 600;
}
</style>
