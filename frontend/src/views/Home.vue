<template>
  <div class="flex h-screen w-screen overflow-hidden bg-white">
    <!-- Left Sidebar: Sources -->
    <aside :class="[
      'border-r border-notebook-200 flex flex-col bg-notebook-50 transition-all duration-300 ease-in-out',
      store.leftSidebarCollapsed ? 'w-16' : 'w-80'
    ]">
      <!-- Expanded View -->
      <template v-if="!store.leftSidebarCollapsed">
        <!-- Header with Notebook Name and Collapse Button -->
        <div class="p-4 flex items-center justify-between">
          <h1 class="text-xl font-semibold text-notebook-900 truncate">VibeProject</h1>
          <button 
            @click="store.toggleLeftSidebar"
            class="p-1 hover:bg-notebook-200 rounded-lg transition-colors flex-shrink-0"
            title="Collapse sidebar"
          >
            <component :is="icons.ChevronLeft" :size="18" class="text-notebook-600" />
          </button>
        </div>

        <!-- View Toggle Buttons -->
        <div class="p-4 b">
          <div class="flex items-center gap-2">
            <button
              @click="store.setSidebarView('notebooks')"
              :class="[
                'flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                store.sidebarView === 'notebooks' 
                  ? 'bg-white shadow-sm text-notebook-900' 
                  : 'text-notebook-600 hover:bg-notebook-100'
              ]"
            >
              <component :is="icons.BookOpen" :size="16" class="inline mr-1.5" />
              Notebooks
            </button>
            <button
              @click="store.setSidebarView('sources')"
              :class="[
                'flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                store.sidebarView === 'sources' 
                  ? 'bg-white shadow-sm text-notebook-900' 
                  : 'text-notebook-600 hover:bg-notebook-100'
              ]"
            >
              <component :is="icons.FileText" :size="16" class="inline mr-1.5" />
              Sources
            </button>
          </div>
        </div>

        <!-- Transition Wrapper for View Content -->
        <transition
          mode="out-in"
          :enter-active-class="store.sidebarView === 'notebooks' ? 'transition-all duration-200 ease-out' : 'transition-all duration-200 ease-out'"
          :enter-from-class="store.sidebarView === 'notebooks' ? 'opacity-0 -translate-x-4' : 'opacity-0 translate-x-4'"
          :enter-to-class="'opacity-100 translate-x-0'"
          :leave-active-class="'transition-all duration-150 ease-in'"
          :leave-from-class="'opacity-100 translate-x-0'"
          :leave-to-class="store.sidebarView === 'notebooks' ? 'opacity-0 translate-x-4' : 'opacity-0 -translate-x-4'"
        >
          <!-- Notebooks View -->
          <div v-if="store.sidebarView === 'notebooks'" key="notebooks" class="flex-1 flex flex-col overflow-hidden">
            <!-- Create Notebook Button -->
            <div class="p-4 border-b border-notebook-200">
              <button
                @click="store.createNotebook"
                class="w-full flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
              <component :is="icons.Plus" :size="18" />
              <span class="text-sm font-medium">New Notebook</span>
            </button>
          </div>

          <!-- Notebooks List -->
          <div class="flex-1 overflow-y-auto scrollbar-thin p-4">
            <div v-if="store.notebooks.length === 0" class="text-center py-12">
              <component :is="icons.BookOpen" :size="48" class="mx-auto mb-3 text-notebook-300" />
              <p class="text-sm text-notebook-500">No notebooks yet</p>
              <p class="text-xs text-notebook-400 mt-1">Create one to get started</p>
            </div>

            <div v-else class="space-y-2">
              <div
                v-for="notebook in store.notebooks"
                :key="notebook.id"
                class="relative group"
              >
                <button
                  @click="store.selectNotebook(notebook.id)"
                  class="w-full flex items-start gap-3 p-3 rounded-lg hover:bg-notebook-100 transition-colors text-left"
                  :class="{ 'bg-blue-50 border border-blue-300': store.activeNotebook.id === notebook.id }"
                >
                  <component :is="icons.BookOpen" :size="18" class="text-notebook-500 mt-0.5 flex-shrink-0" />
                  <div class="flex-1 min-w-0">
                    <p class="text-sm font-medium text-notebook-900 truncate">{{ notebook.name }}</p>
                    <p class="text-xs text-notebook-500 mt-0.5">{{ notebook.papers.length }} sources • {{ notebook.createdAt }}</p>
                  </div>
                  <button
                    @click.stop="store.toggleNotebookMenu(notebook.id)"
                    class="p-1 opacity-0 group-hover:opacity-100 hover:bg-notebook-200 rounded transition-opacity"
                  >
                    <component :is="icons.MoreVertical" :size="16" class="text-notebook-600" />
                  </button>
                </button>

                <!-- 3-Dot Menu -->
                <transition
                  enter-active-class="transition ease-out duration-100"
                  enter-from-class="transform opacity-0 scale-95"
                  enter-to-class="transform opacity-100 scale-100"
                  leave-active-class="transition ease-in duration-75"
                  leave-from-class="transform opacity-100 scale-100"
                  leave-to-class="transform opacity-0 scale-95"
                >
                  <div
                    v-if="store.notebookMenuOpen === notebook.id"
                    class="absolute right-2 top-12 bg-white border border-notebook-200 rounded-lg shadow-lg overflow-hidden z-10 min-w-[160px]"
                  >
                    <button
                      @click="handleRenameNotebook(notebook)"
                      class="w-full flex items-center gap-3 px-4 py-2 hover:bg-notebook-50 transition-colors text-left text-sm"
                    >
                      <component :is="icons.Edit" :size="14" class="text-notebook-600" />
                      <span class="text-notebook-900">Rename</span>
                    </button>
                    <button
                      @click="handleDeleteNotebook(notebook.id)"
                      class="w-full flex items-center gap-3 px-4 py-2 hover:bg-red-50 transition-colors text-left text-sm border-t border-notebook-100"
                    >
                      <component :is="icons.Trash2" :size="14" class="text-red-600" />
                      <span class="text-red-600">Delete</span>
                    </button>
                  </div>
                </transition>
              </div>
            </div>
          </div>
          </div>

          <!-- Sources View -->
          <div v-else key="sources" class="flex-1 flex flex-col overflow-hidden">
          <!-- Upload Zone -->
          <div class="p-4 border-b border-notebook-200">
            <div class="border-2 border-dashed border-notebook-300 rounded-lg p-6 text-center hover:border-blue-400 hover:bg-blue-50/50 transition-all cursor-pointer">
              <component :is="icons.Upload" :size="32" class="mx-auto mb-2 text-notebook-400" />
              <p class="text-sm font-medium text-notebook-700">Upload sources</p>
              <p class="text-xs text-notebook-500 mt-1">PDFs, text, or paste URLs</p>
            </div>
          </div>

          <!-- Sources List -->
          <div class="flex-1 overflow-y-auto scrollbar-thin p-4">
            <div v-if="store.activeNotebook.papers.length === 0" class="text-center py-12">
              <component :is="icons.FileText" :size="48" class="mx-auto mb-3 text-notebook-300" />
              <p class="text-sm text-notebook-500">No sources yet</p>
              <p class="text-xs text-notebook-400 mt-1">Upload papers to get started</p>
            </div>
            
            <div v-else class="space-y-2">
              <div 
                v-for="paper in store.activeNotebook.papers" 
                :key="paper.id"
                @click="store.selectSource(paper)"
                class="p-3 border bg-white border-notebook-200 rounded-lg hover:bg-notebook-100 cursor-pointer transition-colors"
                :class="{ 'bg-blue-50 border-blue-300': store.selectedSource?.id === paper.id }"
              >
                <div class="flex items-start gap-2">
                  <component :is="icons.FileText" :size="16" class="text-notebook-500 mt-0.5" />
                  <div class="flex-1 min-w-0">
                    <p class="text-sm font-medium text-notebook-900 truncate">{{ paper.title }}</p>
                    <p class="text-xs text-notebook-500 mt-0.5">{{ paper.authors }}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          </div>
        </transition>

        <!-- User Profile Section -->
        <div class="border-t border-notebook-200 bg-white relative">
          <button 
            @click="store.toggleUserMenu"
            class="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-notebook-50 transition-colors"
          >
            <!-- Avatar -->
            <div :class="[store.user.avatarColor, 'w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold text-sm']">
              {{ store.user.initials }}
            </div>
            
            <!-- User Info -->
            <div class="flex-1 text-left min-w-0">
              <p class="text-sm font-medium text-notebook-900 truncate">{{ store.user.name }}</p>
              <p class="text-xs text-notebook-500 truncate">{{ store.user.email }}</p>
            </div>
            
            <!-- Dropdown Icon -->
            <component :is="icons.ChevronUp" :size="16" class="text-notebook-400 transition-transform" :class="{ 'rotate-180': !store.showUserMenu }" />
          </button>

          <!-- Dropdown Menu -->
          <transition
            enter-active-class="transition ease-out duration-100"
            enter-from-class="transform opacity-0 scale-95"
            enter-to-class="transform opacity-100 scale-100"
            leave-active-class="transition ease-in duration-75"
            leave-from-class="transform opacity-100 scale-100"
            leave-to-class="transform opacity-0 scale-95"
          >
            <div 
              v-if="store.showUserMenu"
              class="absolute bottom-full left-3 right-3 mb-2 bg-white border border-notebook-200 rounded-lg shadow-lg overflow-hidden"
            >
              <button
                @click="store.openSettings"
                class="w-full flex items-center gap-3 px-4 py-3 hover:bg-notebook-50 transition-colors text-left"
              >
                <component :is="icons.Settings" :size="16" class="text-notebook-600" />
                <span class="text-sm text-notebook-900">Settings</span>
              </button>
              <button
                @click="store.logout"
                class="w-full flex items-center gap-3 px-4 py-3 hover:bg-notebook-50 transition-colors text-left border-t border-notebook-100"
              >
                <component :is="icons.LogOut" :size="16" class="text-notebook-600" />
                <span class="text-sm text-notebook-900">Logout</span>
              </button>
            </div>
          </transition>
        </div>
      </template>

      <!-- Collapsed View -->
      <template v-else>
        <!-- Expand Button -->
        <div class="p-4 flex justify-center">
          <button 
            @click="store.toggleLeftSidebar"
            class="p-1 hover:bg-notebook-200 rounded-lg transition-colors"
            title="Expand sidebar"
          >
            <component :is="icons.ChevronRight" :size="18" class="text-notebook-600" />
          </button>
        </div>

        <!-- Spacer -->
        <div class="flex-1"></div>

        <!-- Collapsed User Avatar -->
        <div class="border-t border-notebook-200 p-2 bg-white flex justify-center">
          <div :class="[store.user.avatarColor, 'w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold text-sm cursor-pointer']">
            {{ store.user.initials }}
          </div>
        </div>
      </template>
    </aside>

    <!-- Center: Chat Interface -->
    <main class="flex-1 flex flex-col bg-white">
      <!-- Header -->
      <header class="h-14 flex items-center justify-between px-6">
        <h1 class="text-xl font-semibold text-notebook-900">{{ store.activeNotebook.name }}</h1>
        
        <!-- Show expand button when right panel is hidden -->
        <button 
          v-if="!store.rightPanelVisible"
          @click="store.toggleRightPanel"
          class="p-2 hover:bg-notebook-100 rounded-lg transition-colors"
          title="Show panel"
        >
          <component :is="icons.ChevronLeft" :size="18" class="text-notebook-600" />
        </button>
      </header>

      <!-- Chat Messages Area -->
      <div class="flex-1 overflow-y-auto scrollbar-thin p-6">
        <!-- Welcome State -->
        <div v-if="chatMessages.length === 0" class="max-w-3xl mx-auto">
          <div class="text-center py-12">
            <h2 class="text-4xl font-bold text-notebook-900 mb-3">Research with AI</h2>
            <p class="text-lg text-notebook-600">Upload sources and ask questions to get started</p>
          </div>

          <!-- Feature Cards -->
          <div class="grid grid-cols-2 gap-4 mt-12">
            <div class="p-4 border border-notebook-200 rounded-xl hover:shadow-md transition-shadow cursor-pointer">
              <component :is="icons.Brain" :size="24" class="text-blue-500 mb-2" />
              <h3 class="font-semibold text-notebook-900 mb-1">Smart Analysis</h3>
              <p class="text-sm text-notebook-600">AI-powered insights from your research papers</p>
            </div>
            <div class="p-4 border border-notebook-200 rounded-xl hover:shadow-md transition-shadow cursor-pointer">
              <component :is="icons.Quote" :size="24" class="text-green-500 mb-2" />
              <h3 class="font-semibold text-notebook-900 mb-1">Source Citations</h3>
              <p class="text-sm text-notebook-600">Every answer linked to original sources</p>
            </div>
            <div class="p-4 border border-notebook-200 rounded-xl hover:shadow-md transition-shadow cursor-pointer">
              <component :is="icons.Layers" :size="24" class="text-purple-500 mb-2" />
              <h3 class="font-semibold text-notebook-900 mb-1">Multi-Paper Query</h3>
              <p class="text-sm text-notebook-600">Ask questions across multiple documents</p>
            </div>
            <div class="p-4 border border-notebook-200 rounded-xl hover:shadow-md transition-shadow cursor-pointer">
              <component :is="icons.Sparkles" :size="24" class="text-orange-500 mb-2" />
              <h3 class="font-semibold text-notebook-900 mb-1">Generate Content</h3>
              <p class="text-sm text-notebook-600">Create code, posters, and summaries</p>
            </div>
          </div>
        </div>

        <!-- Chat Messages -->
        <div v-else class="max-w-3xl mx-auto space-y-6">
          <div 
            v-for="(message, index) in chatMessages" 
            :key="index"
            class="flex gap-3"
            :class="message.role === 'user' ? 'justify-end' : 'justify-start'"
          >
            <!-- User Message -->
            <div v-if="message.role === 'user'" class="max-w-[80%] bg-notebook-100 rounded-2xl px-4 py-3">
              <p class="text-sm text-notebook-900">{{ message.content }}</p>
            </div>

            <!-- AI Message -->
            <div v-else class="max-w-[80%] flex gap-3">
              <div class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
                <component :is="icons.Sparkles" :size="16" class="text-white" />
              </div>
              <div class="flex-1">
                <div class="prose prose-sm max-w-none" v-html="renderMarkdown(message.content)"></div>
                
                <!-- Citations -->
                <div v-if="message.citations && message.citations.length > 0" class="flex flex-wrap gap-2 mt-3">
                  <button
                    v-for="citation in message.citations"
                    :key="citation.id"
                    @click="store.selectCitation(citation)"
                    class="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium bg-blue-50 text-blue-700 rounded-md hover:bg-blue-100 transition-colors border border-blue-200"
                  >
                    <component :is="icons.FileText" :size="12" />
                    [{{ citation.id }}] {{ citation.title }}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Fixed Input Bar -->
      <div class=" p-2 bg-white">
        <div class="max-w-3xl mx-auto">
          <div class="flex gap-3 items-center">
            <div class="flex-1 relative flex items-center">
              <textarea
                v-model="inputMessage"
                @keydown.enter.prevent="handleSendMessage"
                placeholder="Ask a question about your sources..."
                rows="1"
                class="w-full px-4 py-3 pr-12 border border-notebook-300 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none scrollbar-thin"
                style="max-height: 120px;"
              ></textarea>
              <button
                @click="handleSendMessage"
                :disabled="!inputMessage.trim()"
                class="absolute right-2 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center hover:bg-blue-600 transition-colors disabled:bg-notebook-300 disabled:cursor-not-allowed"
              >
                <component :is="icons.Send" :size="16" />
              </button>
            </div>
          </div>
          <p class="text-xs text-notebook-500 mt-2 text-center">Press Enter to send, Shift+Enter for new line</p>
        </div>
      </div>
    </main>

    <!-- Right Panel: Notes / Source Preview -->
    <aside :class="[
      'border-l border-notebook-200 flex flex-col bg-notebook-50 transition-all duration-300 ease-in-out overflow-hidden',
      store.rightPanelVisible ? 'w-96' : 'w-0 border-l-0'
    ]">
      <div v-if="store.rightPanelVisible" class="w-96 flex flex-col h-full">
        <!-- Collapse Button -->
        <div class="p-4 flex justify-start">
          <button 
            @click="store.toggleRightPanel"
            class="p-1 hover:bg-notebook-200 rounded-lg transition-colors"
            title="Collapse panel"
          >
            <component :is="icons.ChevronRight" :size="18" class="text-notebook-600" />
          </button>
        </div>

        <!-- Mode Toggle Buttons -->
        <div class="p-4">
          <div class="flex items-center gap-2">
            <button
              @click="store.setRightPanelMode('notes')"
              :class="[
                'flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                store.rightPanelMode === 'notes' 
                  ? 'bg-white shadow-sm text-notebook-900' 
                  : 'text-notebook-600 hover:bg-notebook-100'
              ]"
            >
              <component :is="icons.StickyNote" :size="16" class="inline mr-1.5" />
              Notes
            </button>
            <button
              @click="store.setRightPanelMode('preview')"
              :class="[
                'flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                store.rightPanelMode === 'preview' 
                  ? 'bg-white shadow-sm text-notebook-900' 
                  : 'text-notebook-600 hover:bg-notebook-100'
              ]"
            >
              <component :is="icons.Eye" :size="16" class="inline mr-1.5" />
              Preview
            </button>
          </div>
        </div>

        <!-- Panel Content -->
        <div class="flex-1 overflow-y-auto scrollbar-thin p-4">
            <!-- Notes Mode -->
            <div v-if="store.rightPanelMode === 'notes'">
              <div v-if="store.activeNotebook.notes.length === 0" class="text-center py-12">
                <component :is="icons.StickyNote" :size="48" class="mx-auto mb-3 text-notebook-300" />
                <p class="text-sm text-notebook-500">No notes yet</p>
                <p class="text-xs text-notebook-400 mt-1">Create notes from your research</p>
              </div>
              
              <div v-else class="grid grid-cols-1 gap-3">
                <div
                  v-for="note in store.activeNotebook.notes"
                  :key="note.id"
                  class="p-3 bg-white border border-notebook-200 rounded-lg hover:shadow-sm transition-shadow cursor-pointer"
                >
                  <h4 class="text-sm font-semibold text-notebook-900 mb-1">{{ note.title }}</h4>
                  <p class="text-xs text-notebook-600 line-clamp-3">{{ note.content }}</p>
                  <p class="text-xs text-notebook-400 mt-2">{{ note.date }}</p>
                </div>
              </div>
            </div>

            <!-- Preview Mode -->
            <div v-else-if="store.rightPanelMode === 'preview'">
              <div v-if="!store.selectedSource && !store.selectedCitation" class="text-center py-12">
                <component :is="icons.Eye" :size="48" class="mx-auto mb-3 text-notebook-300" />
                <p class="text-sm text-notebook-500">No preview</p>
                <p class="text-xs text-notebook-400 mt-1">Click a source or citation to preview</p>
              </div>

              <div v-else class="space-y-4">
                <!-- Selected Citation Preview -->
                <div v-if="store.selectedCitation" class="bg-white border border-notebook-200 rounded-lg p-4">
                  <div class="flex items-start justify-between mb-3">
                    <h3 class="text-sm font-semibold text-notebook-900">Citation [{{ store.selectedCitation.id }}]</h3>
                    <button @click="store.selectedCitation = null" class="text-notebook-400 hover:text-notebook-600">
                      <component :is="icons.X" :size="16" />
                    </button>
                  </div>
                  <p class="text-xs text-notebook-600 mb-2">{{ store.selectedCitation.title }}</p>
                  <div class="text-sm text-notebook-800 bg-notebook-50 rounded p-3 border-l-4 border-blue-500">
                    {{ store.selectedCitation.excerpt }}
                  </div>
                </div>

                <!-- Selected Source Preview -->
                <div v-if="store.selectedSource" class="bg-white border border-notebook-200 rounded-lg p-4">
                  <div class="flex items-start justify-between mb-3">
                    <h3 class="text-sm font-semibold text-notebook-900">{{ store.selectedSource.title }}</h3>
                    <button @click="store.selectedSource = null" class="text-notebook-400 hover:text-notebook-600">
                      <component :is="icons.X" :size="16" />
                    </button>
                  </div>
                  <p class="text-xs text-notebook-600 mb-3">{{ store.selectedSource.authors }}</p>
                  <div class="text-sm text-notebook-800 space-y-2">
                    <p class="font-medium">Abstract</p>
                    <p class="text-xs text-notebook-600">{{ store.selectedSource.abstract || 'No abstract available' }}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
    </aside>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useAppStore } from '../stores/app'
import { marked } from 'marked'
import {
  Upload, FileText, Settings, PanelRight, Brain, Quote, Layers, Sparkles,
  Send, StickyNote, Eye, X, ChevronUp, ChevronLeft, ChevronRight, LogOut,
  Plus, BookOpen, MoreVertical, Edit, Trash2
} from 'lucide-vue-next'

const store = useAppStore()

// Icon components
const icons = {
  Upload, FileText, Settings, PanelRight, Brain, Quote, Layers, Sparkles,
  Send, StickyNote, Eye, X, ChevronUp, ChevronLeft, ChevronRight, LogOut,
  Plus, BookOpen, MoreVertical, Edit, Trash2
}

// Chat state
const inputMessage = ref('')
const chatMessages = ref([])

// Notebook handlers
const handleRenameNotebook = (notebook) => {
  const newName = prompt('Enter new notebook name:', notebook.name)
  if (newName && newName.trim()) {
    store.renameNotebook(notebook.id, newName.trim())
  }
}

const handleDeleteNotebook = (id) => {
  if (confirm('Are you sure you want to delete this notebook? This action cannot be undone.')) {
    store.deleteNotebook(id)
  }
}

// Markdown rendering
const renderMarkdown = (text) => {
  return marked(text, { breaks: true, gfm: true })
}

// Handle message sending
const handleSendMessage = () => {
  if (!inputMessage.value.trim()) return

  // Add user message to active notebook
  const userMessage = {
    role: 'user',
    content: inputMessage.value
  }
  store.activeNotebook.messages.push(userMessage)
  chatMessages.value.push(userMessage)

  const userQuestion = inputMessage.value
  inputMessage.value = ''

  // Simulate AI response after a delay
  setTimeout(() => {
    const aiMessage = {
      role: 'assistant',
      content: `Based on the research papers, here's what I found about "${userQuestion}":\n\nThe **Transformer architecture** introduced in "Attention Is All You Need" revolutionized NLP by using self-attention mechanisms instead of recurrence. This allows for better parallelization and capturing long-range dependencies.\n\nKey innovations include:\n- Multi-head attention layers\n- Positional encodings\n- Feed-forward networks\n\nLater work like **BERT** built upon this foundation to create powerful pre-trained models.`,
      citations: [
        { id: 1, title: 'Attention Is All You Need', excerpt: 'The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder...' },
        { id: 2, title: 'BERT', excerpt: 'BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context...' }
      ]
    }
    store.activeNotebook.messages.push(aiMessage)
    chatMessages.value.push(aiMessage)
  }, 1000)
}
</script>

<style scoped>
/* Markdown prose styles */
:deep(.prose) {
  @apply text-notebook-800;
}

:deep(.prose p) {
  @apply mb-3;
}

:deep(.prose strong) {
  @apply font-semibold text-notebook-900;
}

:deep(.prose ul) {
  @apply list-disc list-inside mb-3;
}

:deep(.prose ol) {
  @apply list-decimal list-inside mb-3;
}

:deep(.prose code) {
  @apply bg-notebook-100 px-1 py-0.5 rounded text-sm;
}

/* Auto-resize textarea */
textarea {
  field-sizing: content;
}
</style>
