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
                <div
                  class="w-full flex items-start gap-3 p-3 rounded-lg hover:bg-notebook-100 transition-colors cursor-pointer"
                  :class="{ 'bg-blue-50 border border-blue-300': store.activeNotebook.id === notebook.id }"
                  @click="store.selectNotebook(notebook.id)"
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
                </div>

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
            <label
              :class="[
                'border-2 border-dashed rounded-lg p-6 text-center transition-all cursor-pointer block',
                uploadState === 'uploading' ? 'border-blue-300 bg-blue-50/50 cursor-not-allowed' :
                uploadState === 'success' ? 'border-green-400 bg-green-50/50' :
                uploadState === 'error' ? 'border-red-400 bg-red-50/50' :
                'border-notebook-300 hover:border-blue-400 hover:bg-blue-50/50'
              ]"
            >
              <input type="file" accept=".pdf" class="hidden" @change="handleFileUpload" :disabled="uploadState === 'uploading'" />
              <component :is="icons.Upload" :size="32" :class="['mx-auto mb-2', uploadState === 'uploading' ? 'text-blue-400 animate-pulse' : 'text-notebook-400']" />
              <p class="text-sm font-medium text-notebook-700">
                {{ uploadState === 'uploading' ? 'Extracting content...' : uploadState === 'success' ? `Done! ${tablesIndexed} table${tablesIndexed !== 1 ? 's' : ''}, ${chunksIndexed} text chunk${chunksIndexed !== 1 ? 's' : ''} indexed` : uploadState === 'error' ? 'Upload failed' : 'Upload sources' }}
              </p>
              <p class="text-xs text-notebook-500 mt-1">{{ uploadState === 'error' ? uploadError : 'PDFs only' }}</p>
            </label>
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
                class="relative group"
              >
                <div
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
                    <button
                      @click.stop="store.togglePaperMenu(paper.id)"
                      class="p-1 opacity-0 group-hover:opacity-100 hover:bg-notebook-200 rounded transition-opacity"
                    >
                      <component :is="icons.MoreVertical" :size="16" class="text-notebook-600" />
                    </button>
                  </div>
                </div>

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
                    v-if="store.paperMenuOpen === paper.id"
                    class="absolute right-2 top-12 w-40 bg-white rounded-lg shadow-lg border border-notebook-200 py-1 z-10"
                  >
                    <button
                      @click="handleRenamePaper(paper)"
                      class="w-full px-4 py-2 text-left text-sm text-notebook-700 hover:bg-notebook-50 flex items-center gap-2"
                    >
                      <component :is="icons.Edit" :size="14" />
                      Rename
                    </button>
                    <button
                      @click="handleDeletePaper(paper)"
                      class="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
                    >
                      <component :is="icons.Trash2" :size="14" />
                      Delete
                    </button>
                  </div>
                </transition>
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
        <h1 class="text-xl font-semibold text-notebook-900 truncate">{{ store.activeNotebook.name }}</h1>
      </header>

      <!-- Chat Messages Area -->
      <div ref="messagesContainer" class="flex-1 overflow-y-auto scrollbar-thin p-6">
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

        <!-- Typing Indicator -->
        <div v-if="store.isTyping" class="max-w-3xl mx-auto mt-6 flex gap-3 justify-start">
          <div class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
            <component :is="icons.Sparkles" :size="16" class="text-white" />
          </div>
          <div class="bg-notebook-100 rounded-2xl px-4 py-3 flex items-center gap-1">
            <span class="w-2 h-2 bg-notebook-400 rounded-full animate-bounce" style="animation-delay: 0ms"></span>
            <span class="w-2 h-2 bg-notebook-400 rounded-full animate-bounce" style="animation-delay: 150ms"></span>
            <span class="w-2 h-2 bg-notebook-400 rounded-full animate-bounce" style="animation-delay: 300ms"></span>
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
                @keydown.enter.exact.prevent="handleSendMessage"
                placeholder="Ask a question about your sources..."
                rows="1"
                class="w-full px-4 py-3 pr-12 border border-notebook-300 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none scrollbar-thin"
                style="max-height: 120px;"
              ></textarea>
              <button
                @click="handleSendMessage"
                :disabled="!inputMessage.trim() || store.isTyping"
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

    <!-- Right Panel: Paper Generation Features -->
    <aside :class="[
      'border-l border-notebook-200 flex flex-col bg-notebook-50 transition-all duration-300 ease-in-out',
      store.rightPanelVisible ? 'w-96' : 'w-16'
    ]">
      <!-- Expanded View -->
      <div v-if="store.rightPanelVisible" class="w-96 flex flex-col h-full">
        <!-- Header with Lab Label and Collapse Button -->
        <div class="p-4 flex items-center justify-between">
          <button 
            @click="store.toggleRightPanel"
            class="p-1 hover:bg-notebook-200 rounded-lg transition-colors"
            title="Collapse panel"
          >
            <component :is="icons.ChevronRight" :size="18" class="text-notebook-600" />
          </button>
          <h1 class="text-xl font-semibold text-notebook-900 truncate">Lab</h1>
        </div>

        <!-- Feature Cards Grid -->
        <div class="flex-1 p-4 space-y-4">
          <!-- Row 1: Paper to Code | Paper to Poster -->
          <div class="grid grid-cols-2 gap-4">
            <!-- Paper to Code -->
            <button
              @click="store.openPaperSelector('code')"
              class="p-4 border border-notebook-200 rounded-xl hover:shadow-md transition-all cursor-pointer bg-white text-left"
            >
              <component :is="icons.Code" :size="24" class="text-blue-500 mb-2" />
              <h3 class="font-semibold text-notebook-900 text-sm mb-1">Paper to Code</h3>
              <p class="text-xs text-notebook-600">Generate code from paper</p>
            </button>

            <!-- Paper to Poster -->
            <button
              @click="store.openPaperSelector('poster')"
              class="p-4 border border-notebook-200 rounded-xl hover:shadow-md transition-all cursor-pointer bg-white text-left"
            >
              <component :is="icons.Image" :size="24" class="text-purple-500 mb-2" />
              <h3 class="font-semibold text-notebook-900 text-sm mb-1">Paper to Poster</h3>
              <p class="text-xs text-notebook-600">Convert paper in to conference poster</p>
            </button>
          </div>

          <!-- Row 2: Paper to Web -->
          <div class="grid grid-cols-2 gap-4">
            <button
              @click="store.openPaperSelector('web')"
              class="p-4 border border-notebook-200 rounded-xl hover:shadow-md transition-all cursor-pointer bg-white text-left"
            >
              <component :is="icons.Globe" :size="24" class="text-green-500 mb-2" />
              <h3 class="font-semibold text-notebook-900 text-sm mb-1">Paper to Web</h3>
              <p class="text-xs text-notebook-600">Transform the paper into interactive website</p>
            </button>
          </div>
        </div>
      </div>

      <!-- Collapsed View -->
      <div v-else class="w-16 flex flex-col h-full py-4">
        <!-- Expand Button -->
        <button 
          @click="store.toggleRightPanel"
          class="mb-4 p-2 mx-auto hover:bg-notebook-200 rounded-lg transition-colors"
          title="Expand panel"
        >
          <component :is="icons.ChevronLeft" :size="18" class="text-notebook-600" />
        </button>

        <!-- Feature Icons -->
        <div class="flex-1 flex flex-col items-center gap-3 px-2">
          <!-- Code Icon -->
          <button
            @click="store.openPaperSelector('code')"
            class="p-2 hover:bg-blue-100 rounded-lg transition-colors group"
            title="Paper to Code"
          >
            <component :is="icons.Code" :size="20" class="text-blue-500 group-hover:text-blue-600" />
          </button>

          <!-- Poster Icon -->
          <button
            @click="store.openPaperSelector('poster')"
            class="p-2 hover:bg-purple-100 rounded-lg transition-colors group"
            title="Paper to Poster"
          >
            <component :is="icons.Image" :size="20" class="text-purple-500 group-hover:text-purple-600" />
          </button>

          <!-- Web Icon -->
          <button
            @click="store.openPaperSelector('web')"
            class="p-2 hover:bg-green-100 rounded-lg transition-colors group"
            title="Paper to Web"
          >
            <component :is="icons.Globe" :size="20" class="text-green-500 group-hover:text-green-600" />
          </button>
        </div>
      </div>
    </aside>

    <!-- Paper Selection Modal -->
    <div 
      v-if="store.showPaperSelector"
      class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
      @click.self="store.closePaperSelector"
    >
      <div class="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] flex flex-col">
        <!-- Modal Header -->
        <div class="flex items-center justify-between p-6 border-b border-notebook-200">
          <h2 class="text-xl font-semibold text-notebook-900">Select a Paper</h2>
          <button 
            @click="store.closePaperSelector"
            class="p-1 hover:bg-notebook-100 rounded-lg transition-colors"
          >
            <component :is="icons.X" :size="20" class="text-notebook-600" />
          </button>
        </div>

        <!-- Papers List -->
        <div class="flex-1 overflow-y-auto p-6">
          <div v-if="store.activeNotebook.papers.length === 0" class="text-center py-12">
            <component :is="icons.FileText" :size="48" class="mx-auto mb-3 text-notebook-300" />
            <p class="text-sm text-notebook-500">No papers in this notebook</p>
            <p class="text-xs text-notebook-400 mt-1">Upload papers to get started</p>
          </div>

          <div v-else class="space-y-3">
            <button
              v-for="paper in store.activeNotebook.papers"
              :key="paper.id"
              @click="store.selectPaperForGeneration(paper)"
              class="w-full p-4 border border-notebook-200 rounded-lg hover:bg-notebook-50 hover:border-blue-300 transition-all text-left"
            >
              <div class="flex items-start gap-3">
                <component :is="icons.FileText" :size="20" class="text-notebook-500 mt-0.5 flex-shrink-0" />
                <div class="flex-1 min-w-0">
                  <p class="text-sm font-medium text-notebook-900">{{ paper.title }}</p>
                  <p class="text-xs text-notebook-500 mt-1">{{ paper.authors }}</p>
                </div>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Confirmation Dialog -->
    <div 
      v-if="store.showConfirmation"
      class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
      @click.self="store.cancelGeneration"
    >
      <div class="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6">
        <h2 class="text-xl font-semibold text-notebook-900 mb-4">Confirm Generation</h2>
        <p class="text-sm text-notebook-600 mb-6">
          Generate <span class="font-semibold">{{ 
            store.selectedFeature === 'code' ? 'Code' : 
            store.selectedFeature === 'poster' ? 'Poster' : 
            'Web Page' 
          }}</span> for "<span class="font-semibold">{{ store.selectedPaperForGeneration?.title }}</span>"?
        </p>
        
        <div class="flex gap-3">
          <button
            @click="store.cancelGeneration"
            class="flex-1 px-4 py-2 border border-notebook-300 text-notebook-700 rounded-lg hover:bg-notebook-50 transition-colors font-medium"
          >
            No
          </button>
          <button
            @click="store.confirmGeneration"
            class="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
          >
            Yes
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { useAppStore } from '../stores/app'
import { marked } from 'marked'
import {
  Upload, FileText, Settings, PanelRight, Brain, Quote, Layers, Sparkles,
  Send, X, ChevronUp, ChevronLeft, ChevronRight, LogOut,
  Plus, BookOpen, MoreVertical, Edit, Trash2, Code, Image, Globe
} from 'lucide-vue-next'

const store = useAppStore()

// Icon components
const icons = {
  Upload, FileText, Settings, PanelRight, Brain, Quote, Layers, Sparkles,
  Send, X, ChevronUp, ChevronLeft, ChevronRight, LogOut,
  Plus, BookOpen, MoreVertical, Edit, Trash2, Code, Image, Globe
}

// Chat state
const inputMessage = ref('')
const messagesContainer = ref(null)
const chatMessages = computed(() => store.activeNotebook?.messages ?? [])

// Upload state
const uploadState = ref('idle') // 'idle' | 'uploading' | 'success' | 'error'
const uploadError = ref('')
const tablesIndexed = ref(0)
const chunksIndexed = ref(0)

const handleFileUpload = async (event) => {
  const file = event.target.files[0]
  if (!file) return
  uploadState.value = 'uploading'
  uploadError.value = ''
  tablesIndexed.value = 0
  chunksIndexed.value = 0
  try {
    const result = await store.uploadPaper(file)
    tablesIndexed.value = result?.tables_indexed ?? 0
    chunksIndexed.value = result?.chunks_indexed ?? 0
    uploadState.value = 'success'
    setTimeout(() => { uploadState.value = 'idle' }, 3000)
  } catch (err) {
    uploadState.value = 'error'
    uploadError.value = err?.response?.data?.detail || 'Upload failed. Is the backend running?'
    setTimeout(() => { uploadState.value = 'idle' }, 3000)
  }
  event.target.value = ''
}

// Auto-scroll to bottom when messages update or typing indicator changes
watch([chatMessages, () => store.isTyping], () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}, { deep: true })

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

// Paper handlers
const handleRenamePaper = (paper) => {
  const newTitle = prompt('Enter new paper title:', paper.title)
  if (newTitle && newTitle.trim()) {
    store.renamePaper(paper.id, newTitle.trim())
  }
}

const handleDeletePaper = async (paper) => {
  if (confirm(`Are you sure you want to delete "${paper.title}"? This action cannot be undone.`)) {
    await store.deletePaper(paper.id)
  }
}

// Markdown rendering
const renderMarkdown = (text) => {
  return marked(text, { breaks: true, gfm: true })
}

// Handle message sending
const handleSendMessage = () => {
  if (!inputMessage.value.trim() || store.isTyping) return
  const question = inputMessage.value.trim()
  inputMessage.value = ''
  store.sendMessage(question)
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
