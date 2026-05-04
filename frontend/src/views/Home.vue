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
          <img src="/OpenLab_logo_light.png" alt="OpenLab" class="h-9 w-auto object-contain" />
          <button 
            @click="store.toggleLeftSidebar"
            class="p-1 hover:bg-notebook-200 rounded-lg transition-colors flex-shrink-0"
            title="Collapse sidebar"
          >
            <component :is="icons.ChevronLeft" :size="18" class="text-notebook-600" />
          </button>
        </div>

        <!-- View Toggle Buttons -->
        <div class="">
          <div class="flex items-center mx-3 my-2 bg-notebook-100  rounded-full p-1">
            <button
              @click="store.setSidebarView('notebooks')"
              :class="[
                'flex-1 px-3 py-1.5 text-sm font-medium rounded-full transition-all',
                store.sidebarView === 'notebooks'
                  ? 'bg-white text-notebook-900 shadow-card'
                  : 'text-notebook-600 hover:text-notebook-800'
              ]"
            >
              <component :is="icons.BookOpen" :size="14" class="inline mr-1" />
              Notebooks
            </button>
            <button
              @click="store.setSidebarView('sources')"
              :class="[
                'flex-1 px-3 py-1.5 text-sm font-medium rounded-full transition-all',
                store.sidebarView === 'sources'
                  ? 'bg-white text-notebook-900 shadow-card'
                  : 'text-notebook-600 hover:text-notebook-800'
              ]"
            >
              <component :is="icons.FileText" :size="14" class="inline mr-1" />
              Sources
            </button>
          </div>
        </div>

        <!-- Transition Wrapper for View Content -->
        <div class="flex-1 overflow-hidden flex flex-col bg-white">
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
          <div v-if="store.sidebarView === 'notebooks'" key="notebooks" class="flex-1 flex flex-col overflow-hidden bg-white">
            <!-- Create Notebook Button -->
            <div class="p-4 border-b border-notebook-200 bg-white">
                <button
                @click="handleCreateNotebook"
                class="w-full flex items-center gap-2 px-4 py-2 bg-notebook-800 text-white rounded-lg hover:bg-notebook-900 transition-colors"
              >
              <component :is="icons.Plus" :size="18" />
              <span class="text-sm font-medium">New Notebook</span>
            </button>
          </div>

          <!-- Notebooks List -->
          <div class="flex-1 overflow-y-auto scrollbar-thin p-4 bg-white">
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
                  class="w-full flex items-start gap-3 p-3 rounded-lg hover:bg-notebook-100 transition-colors cursor-pointer border"
                  :class="store.activeNotebook?.id === notebook.id ? 'bg-blue-50 border-blue-300' : 'border-notebook-200'"
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
          <div v-else key="sources" class="flex-1 flex flex-col overflow-hidden bg-white">
          <!-- Upload Zone -->
          <div class="p-4 border-b border-notebook-200 bg-white">
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
                {{ uploadState === 'uploading' ? 'Extracting document...' : uploadState === 'success' ? 'Done!' : uploadState === 'error' ? 'Upload failed' : 'Upload sources' }}
              </p>
              <p class="text-xs text-notebook-500 mt-1">{{ uploadState === 'error' ? uploadError : 'PDFs only' }}</p>
            </label>
          </div>

          <!-- Sources List -->
          <div class="flex-1 overflow-y-auto scrollbar-thin p-4 bg-white">
            <div v-if="(store.activeNotebook?.papers?.length ?? 0) === 0" class="text-center py-12">
              <component :is="icons.FileText" :size="48" class="mx-auto mb-3 text-notebook-300" />
              <p class="text-sm text-notebook-500">No sources yet</p>
              <p class="text-xs text-notebook-400 mt-1">Upload papers to get started</p>
            </div>
            
            <div v-else class="space-y-2">
              <div 
              v-for="paper in store.activeNotebook?.papers ?? []" 
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
        </div>

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
        <h1 class="text-xl font-semibold text-notebook-900 truncate">{{ store.activeNotebook?.name ?? '' }}</h1>
      </header>

      <!-- Chat Messages Area -->
      <div ref="messagesContainer" class="flex-1 overflow-y-auto scrollbar-thin p-6">
        <!-- No notebook selected: atmospheric empty state -->
        <div v-if="!store.activeNotebook" class="h-full flex flex-col items-center justify-center relative overflow-hidden empty-state-root">

          <!-- Dot-grid texture -->
          <div class="absolute inset-0 dot-grid pointer-events-none opacity-70"></div>

          <!-- Ambient glow -->
          <div class="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/3 w-[520px] h-72 rounded-full pointer-events-none glow-blob"></div>
          <div class="absolute bottom-0 right-0 translate-x-1/3 translate-y-1/3 w-64 h-64 rounded-full pointer-events-none glow-blob-accent"></div>

          <!-- Orbital cluster -->
          <div class="relative w-48 h-48 flex items-center justify-center mb-10">
            <!-- Rotating dashed outer ring -->
            <div class="absolute inset-0 rounded-full border border-dashed border-notebook-200 ring-rotate"></div>
            <!-- Static inner ring -->
            <div class="absolute w-[112px] h-[112px] rounded-full border border-notebook-100"></div>

            <!-- Center chip -->
            <div class="relative z-10 w-[60px] h-[60px] bg-brand rounded-[18px] flex items-center justify-center center-pulse">
              <component :is="icons.BookOpen" :size="26" class="text-white" />
            </div>

            <!-- Satellite: top-right -->
            <div class="absolute z-10 w-9 h-9 bg-white rounded-xl border border-notebook-100 sat-chip sat-float-1 flex items-center justify-center" style="top:12px; right:12px;">
              <component :is="icons.FileText" :size="14" class="text-notebook-400" />
            </div>
            <!-- Satellite: bottom-left -->
            <div class="absolute z-10 w-9 h-9 bg-white rounded-xl border border-notebook-100 sat-chip sat-float-2 flex items-center justify-center" style="bottom:12px; left:12px;">
              <component :is="icons.Brain" :size="14" class="text-brand" />
            </div>
            <!-- Satellite: bottom-right -->
            <div class="absolute z-10 w-9 h-9 bg-white rounded-xl border border-notebook-100 sat-chip sat-float-3 flex items-center justify-center" style="bottom:16px; right:8px;">
              <component :is="icons.Sparkles" :size="14" class="text-amber-400" />
            </div>
          </div>

          <!-- Headline + description -->
          <div class="relative z-10 text-center max-w-sm px-6 mb-8">
            <h1 class="font-display font-semibold text-notebook-900 tracking-tight mb-3 empty-headline">
              Where research<br>comes alive
            </h1>
            <p class="text-notebook-400 text-sm leading-relaxed">
              Pick up where you left off, or create a new notebook to start exploring your papers with AI.
            </p>
          </div>

          <!-- CTA -->
          <button
            @click="handleCreateNotebook"
            class="relative z-10 inline-flex items-center gap-2 px-6 py-2.5 bg-brand text-white text-sm font-medium rounded-full hover:bg-brand-hover transition-colors mb-10 cta-glow"
          >
            <component :is="icons.Plus" :size="15" />
            New Notebook
          </button>

          <!-- Capability pills -->
          <div class="relative z-10 flex flex-wrap gap-2 justify-center px-10 max-w-md">
            <span class="es-pill"><component :is="icons.Quote"  :size="11" class="text-brand" />Cited answers</span>
            <span class="es-pill"><component :is="icons.Layers" :size="11" class="text-purple-500" />Multi-paper</span>
            <span class="es-pill"><component :is="icons.Code"   :size="11" class="text-green-500" />Paper to Code</span>
            <span class="es-pill"><component :is="icons.Image"  :size="11" class="text-orange-400" />Paper to Poster</span>
            <span class="es-pill"><component :is="icons.Globe"  :size="11" class="text-sky-500" />Paper to Web</span>
          </div>
        </div>

        <!-- Notebook selected, no messages yet: Welcome State -->
        <div v-else-if="chatMessages.length === 0" class="max-w-3xl mx-auto">
          <div class="text-center py-12">
            <h2 class="font-display text-5xl font-semibold text-notebook-900 mb-3">Research with AI</h2>
            <p class="text-lg text-notebook-600">Upload sources and ask questions to get started</p>
          </div>

          <!-- Feature Cards -->
          <div class="grid grid-cols-2 gap-4 mt-12">
            <div class="p-4 border border-notebook-200 rounded-2xl hover:shadow-brand-glow transition-shadow cursor-pointer">
              <component :is="icons.Brain" :size="24" class="text-brand mb-2" />
              <h3 class="font-semibold text-notebook-900 mb-1">Smart Analysis</h3>
              <p class="text-sm text-notebook-600">AI-powered insights from your research papers</p>
            </div>
            <div class="p-4 border border-notebook-200 rounded-2xl hover:shadow-brand-glow transition-shadow cursor-pointer">
              <component :is="icons.Quote" :size="24" class="text-green-500 mb-2" />
              <h3 class="font-semibold text-notebook-900 mb-1">Source Citations</h3>
              <p class="text-sm text-notebook-600">Every answer linked to original sources</p>
            </div>
            <div class="p-4 border border-notebook-200 rounded-2xl hover:shadow-brand-glow transition-shadow cursor-pointer">
              <component :is="icons.Layers" :size="24" class="text-purple-500 mb-2" />
              <h3 class="font-semibold text-notebook-900 mb-1">Multi-Paper Query</h3>
              <p class="text-sm text-notebook-600">Ask questions across multiple documents</p>
            </div>
            <div class="p-4 border border-notebook-200 rounded-2xl hover:shadow-brand-glow transition-shadow cursor-pointer">
              <component :is="icons.Sparkles" :size="24" class="text-orange-500 mb-2" />
              <h3 class="font-semibold text-notebook-900 mb-1">Generate Content</h3>
              <p class="text-sm text-notebook-600">Create code, posters, and website</p>
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
            <div v-if="message.role === 'user'" class="max-w-[80%] bg-[#f0f0f0] rounded-2xl px-4 py-3">
              <p class="text-sm text-notebook-900">{{ message.content }}</p>
            </div>

            <!-- AI Message -->
            <div v-else class="max-w-[80%] flex gap-3">
              <div class="w-8 h-8 rounded-full bg-brand flex items-center justify-center flex-shrink-0">
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
          <div class="w-8 h-8 rounded-full bg-brand flex items-center justify-center flex-shrink-0">
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
      <div v-if="store.activeNotebook" class=" p-2 bg-white">
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
                class="absolute right-2 w-8 h-8 bg-notebook-800 text-white rounded-full flex items-center justify-center hover:bg-notebook-900 transition-colors disabled:bg-notebook-300 disabled:cursor-not-allowed"
              >
                <component :is="icons.Send" :size="16" />
              </button>
            </div>
          </div>
          <p class="text-xs text-notebook-500 mt-2 text-center">AI can make mistakes. Please double-check responses.</p>
        </div>
      </div>
    </main>

    <!-- Right Panel: Paper Generation Features -->
    <aside :class="[
      'border-l border-notebook-200 flex flex-col bg-notebook-50 transition-all duration-300 ease-in-out overflow-hidden',
      store.rightPanelVisible ? 'w-80' : 'w-16'
    ]">
      <!-- Expanded View -->
      <div v-if="store.rightPanelVisible" class="w-80 flex flex-col h-full">
        <!-- Header with Lab Label and Collapse Button -->
        <div class="p-4 flex items-center justify-between">
          <button 
            @click="store.toggleRightPanel"
            class="p-1 hover:bg-notebook-200 rounded-lg transition-colors"
            title="Collapse panel"
          >
            <component :is="icons.ChevronRight" :size="18" class="text-notebook-600" />
          </button>
          <h1 class="text-xl font-semibold text-notebook-900 truncate">Lab Space</h1>
        </div>

        <!-- Feature Cards Grid -->
        <div class="flex-1 pb-4 pr-4 pl-4 space-y-3">
          <!-- Row 1: Paper to Code | Paper to Poster -->
          <div class="grid grid-cols-2 gap-3">

            <!-- Paper to Code -->
            <div class="relative">
              <!-- Idle -->
              <button
                v-if="store.paper2codeJob.status === 'idle'"
                @click="store.openPaperSelector('code')"
                class="w-full aspect-square rounded-2xl overflow-hidden shadow-brand-glow hover:shadow-elevated transition-all cursor-pointer flex flex-col"
              >
                <div class="h-[40%] flex items-center justify-center bg-gradient-to-br from-brand to-indigo-600">
                  <component :is="icons.Code" :size="24" class="text-white" />
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-start pt-2.5 bg-white text-left">
                  <h3 class="font-semibold text-notebook-900 text-xs mb-0.5">Paper to Code</h3>
                  <p class="text-xs text-notebook-500">Generate code from paper</p>
                </div>
              </button>

              <!-- Running -->
              <div
                v-else-if="store.paper2codeJob.status === 'running'"
                class="w-full aspect-square rounded-2xl overflow-hidden flex flex-col"
              >
                <div class="h-[40%] relative flex items-center justify-center bg-gradient-to-br from-brand/80 to-indigo-600/80">
                  <component :is="icons.Code" :size="24" class="text-white" />
                  <button
                    @click="store.cancelCodeJob()"
                    class="absolute top-2 right-2 p-1 bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
                    title="Cancel generation"
                  >
                    <component :is="icons.X" :size="12" class="text-white" />
                  </button>
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-between py-2.5 bg-white text-left">
                  <h3 class="font-semibold text-notebook-900 text-xs">Paper to Code</h3>
                  <div>
                    <div class="w-full bg-notebook-200 rounded-full h-1.5 mb-1">
                      <div
                        class="bg-brand h-1.5 rounded-full transition-all duration-500"
                        :style="{ width: (store.paper2codeJob.progress * 100) + '%' }"
                      ></div>
                    </div>
                    <p class="text-xs text-notebook-500 truncate">{{ store.paper2codeJob.step }}</p>
                  </div>
                </div>
              </div>

              <!-- Done -->
              <div
                v-else-if="store.paper2codeJob.status === 'done'"
                class="w-full aspect-square rounded-2xl overflow-hidden shadow-brand-glow flex flex-col"
              >
                <div class="h-[40%] flex items-center justify-center bg-gradient-to-br from-brand to-indigo-600">
                  <component :is="icons.Code" :size="24" class="text-white" />
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-between py-2.5 bg-white">
                  <h3 class="font-semibold text-notebook-900 text-xs">Paper to Code</h3>
                  <div class="flex gap-1.5">
                    <button
                      @click="store.downloadCodeResult()"
                      class="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-notebook-800 text-white text-xs rounded-lg hover:bg-notebook-900 transition-colors font-medium"
                    >
                      <component :is="icons.Download" :size="10" />
                    </button>
                    <button
                      @click="store.resetCodeJob()"
                      class="flex-1 flex items-center justify-center px-2 py-1 border border-notebook-200 text-notebook-600 text-xs rounded-lg hover:bg-notebook-100 transition-colors"
                    >New</button>
                  </div>
                </div>
              </div>

              <!-- Error -->
              <div
                v-else-if="store.paper2codeJob.status === 'error'"
                class="w-full aspect-square rounded-2xl overflow-hidden flex flex-col"
              >
                <div class="h-[40%] flex items-center justify-center bg-gradient-to-br from-brand/60 to-indigo-600/60">
                  <component :is="icons.AlertCircle" :size="24" class="text-white" />
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-start pt-2.5 bg-white text-left">
                  <h3 class="font-semibold text-notebook-900 text-xs mb-0.5">Paper to Code</h3>
                  <p class="text-xs text-red-500 truncate mb-1">{{ store.paper2codeJob.error || 'Generation failed' }}</p>
                  <button @click="store.resetCodeJob()" class="text-xs text-brand underline hover:opacity-80 text-left">Retry</button>
                </div>
              </div>
            </div>

            <!-- Paper to Poster -->
            <div class="relative">
              <!-- Idle -->
              <button
                v-if="store.paper2posterJob.status === 'idle'"
                @click="store.openPaperSelector('poster')"
                class="w-full aspect-square rounded-2xl overflow-hidden shadow-brand-glow hover:shadow-elevated transition-all cursor-pointer flex flex-col"
              >
                <div class="h-[40%] flex items-center justify-center bg-gradient-to-br from-purple-500 to-pink-500">
                  <component :is="icons.Image" :size="24" class="text-white" />
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-start pt-2.5 bg-white text-left">
                  <h3 class="font-semibold text-notebook-900 text-xs mb-0.5">Paper to Poster</h3>
                  <p class="text-xs text-notebook-500">Generate poster from paper</p>
                </div>
              </button>

              <!-- Running -->
              <div
                v-else-if="store.paper2posterJob.status === 'running'"
                class="w-full aspect-square rounded-2xl overflow-hidden flex flex-col"
              >
                <div class="h-[40%] relative flex items-center justify-center bg-gradient-to-br from-purple-500/80 to-pink-500/80">
                  <component :is="icons.Image" :size="24" class="text-white" />
                  <button
                    @click="store.cancelPosterJob()"
                    class="absolute top-2 right-2 p-1 bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
                    title="Cancel"
                  >
                    <component :is="icons.X" :size="12" class="text-white" />
                  </button>
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-between py-2.5 bg-white text-left">
                  <h3 class="font-semibold text-notebook-900 text-xs">Paper to Poster</h3>
                  <div>
                    <div class="w-full bg-notebook-200 rounded-full h-1.5 mb-1">
                      <div
                        class="bg-purple-500 h-1.5 rounded-full transition-all duration-500"
                        :style="{ width: (store.paper2posterJob.progress * 100) + '%' }"
                      ></div>
                    </div>
                    <p class="text-xs text-notebook-500 truncate">{{ store.paper2posterJob.step }}</p>
                  </div>
                </div>
              </div>

              <!-- Done -->
              <div
                v-else-if="store.paper2posterJob.status === 'done'"
                class="w-full aspect-square rounded-2xl overflow-hidden shadow-brand-glow flex flex-col"
              >
                <div class="h-[40%] flex items-center justify-center bg-gradient-to-br from-purple-500 to-pink-500">
                  <component :is="icons.Image" :size="24" class="text-white" />
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-between py-2.5 bg-white">
                  <h3 class="font-semibold text-notebook-900 text-xs">Paper to Poster</h3>
                  <div class="flex gap-1.5">
                    <button
                      @click="store.downloadPosterResult()"
                      class="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-notebook-800 text-white text-xs rounded-lg hover:bg-notebook-900 transition-colors font-medium"
                    >
                      <component :is="icons.Download" :size="10" />
                      <!-- <span class="truncate"></span> -->
                    </button>
                    <button
                      @click="store.resetPosterJob()"
                      class="flex-1 flex items-center justify-center px-2 py-1 border border-notebook-200 text-notebook-600 text-xs rounded-lg hover:bg-notebook-100 transition-colors"
                    >New</button>
                  </div>
                </div>
              </div>

              <!-- Error -->
              <div
                v-else-if="store.paper2posterJob.status === 'error'"
                class="w-full aspect-square rounded-2xl overflow-hidden flex flex-col"
              >
                <div class="h-[40%] flex items-center justify-center bg-gradient-to-br from-purple-500/60 to-pink-500/60">
                  <component :is="icons.AlertCircle" :size="24" class="text-white" />
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-start pt-2.5 bg-white text-left">
                  <h3 class="font-semibold text-notebook-900 text-xs mb-0.5">Paper to Poster</h3>
                  <p class="text-xs text-red-500 truncate mb-1">{{ store.paper2posterJob.error || 'Generation failed' }}</p>
                  <button @click="store.resetPosterJob()" class="text-xs text-purple-600 underline hover:opacity-80 text-left">Retry</button>
                </div>
              </div>
            </div>

          </div>

          <!-- Row 2: Paper to Web -->
          <div class="grid grid-cols-2 gap-3">
            <div class="relative">
              <!-- Idle -->
              <button
                v-if="store.paper2webJob.status === 'idle'"
                @click="store.openPaperSelector('web')"
                class="w-full aspect-square rounded-2xl overflow-hidden shadow-brand-glow hover:shadow-elevated transition-all cursor-pointer flex flex-col"
              >
                <div class="h-[40%] flex items-center justify-center bg-gradient-to-br from-emerald-500 to-teal-600">
                  <component :is="icons.Globe" :size="24" class="text-white" />
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-start pt-2.5 bg-white text-left">
                  <h3 class="font-semibold text-notebook-900 text-xs mb-0.5">Paper to Web</h3>
                  <p class="text-xs text-notebook-500">Generate website from paper</p>
                </div>
              </button>

              <!-- Running -->
              <div
                v-else-if="store.paper2webJob.status === 'running'"
                class="w-full aspect-square rounded-2xl overflow-hidden flex flex-col"
              >
                <div class="h-[40%] relative flex items-center justify-center bg-gradient-to-br from-emerald-500/80 to-teal-600/80">
                  <component :is="icons.Globe" :size="24" class="text-white" />
                  <button
                    @click="store.cancelWebJob()"
                    class="absolute top-2 right-2 p-1 bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
                    title="Cancel"
                  >
                    <component :is="icons.X" :size="12" class="text-white" />
                  </button>
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-between py-2.5 bg-white text-left">
                  <h3 class="font-semibold text-notebook-900 text-xs">Paper to Web</h3>
                  <div>
                    <div class="w-full bg-notebook-200 rounded-full h-1.5 mb-1">
                      <div
                        class="bg-emerald-500 h-1.5 rounded-full transition-all duration-500"
                        :style="{ width: (store.paper2webJob.progress * 100) + '%' }"
                      ></div>
                    </div>
                    <p class="text-xs text-notebook-500 truncate">{{ store.paper2webJob.step }}</p>
                  </div>
                </div>
              </div>

              <!-- Done -->
              <div
                v-else-if="store.paper2webJob.status === 'done'"
                class="w-full aspect-square rounded-2xl overflow-hidden shadow-brand-glow flex flex-col"
              >
                <div class="h-[40%] flex items-center justify-center bg-gradient-to-br from-emerald-500 to-teal-600">
                  <component :is="icons.Globe" :size="24" class="text-white" />
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-between py-2.5 bg-white">
                  <h3 class="font-semibold text-notebook-900 text-xs">Paper to Web</h3>
                  <div class="flex gap-1.5">
                    <button
                      @click="store.downloadWebResult()"
                      class="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-notebook-800 text-white text-xs rounded-lg hover:bg-notebook-900 transition-colors font-medium"
                    >
                      <component :is="icons.Download" :size="10" />
                    </button>
                    <button
                      @click="store.resetWebJob()"
                      class="flex-1 flex items-center justify-center px-2 py-1 border border-notebook-200 text-notebook-600 text-xs rounded-lg hover:bg-notebook-100 transition-colors"
                    >New</button>
                  </div>
                </div>
              </div>

              <!-- Error -->
              <div
                v-else-if="store.paper2webJob.status === 'error'"
                class="w-full aspect-square rounded-2xl overflow-hidden flex flex-col"
              >
                <div class="h-[40%] flex items-center justify-center bg-gradient-to-br from-emerald-500/60 to-teal-600/60">
                  <component :is="icons.AlertCircle" :size="24" class="text-white" />
                </div>
                <div class="h-[60%] px-3 flex flex-col justify-start pt-2.5 bg-white text-left">
                  <h3 class="font-semibold text-notebook-900 text-xs mb-0.5">Paper to Web</h3>
                  <p class="text-xs text-red-500 truncate mb-1">{{ store.paper2webJob.error || 'Generation failed' }}</p>
                  <button @click="store.resetWebJob()" class="text-xs text-emerald-600 underline hover:opacity-80 text-left">Retry</button>
                </div>
              </div>
            </div>
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
            v-if="store.paper2codeJob.status === 'idle' || store.paper2codeJob.status === 'error'"
            @click="store.paper2codeJob.status === 'error' ? store.resetCodeJob() : store.openPaperSelector('code')"
            class="p-2 hover:bg-blue-100 rounded-lg transition-colors group"
            title="Paper to Code"
          >
            <component :is="icons.Code" :size="20" class="text-blue-500 group-hover:text-blue-600" />
          </button>
          <button
            v-else-if="store.paper2codeJob.status === 'running'"
            class="p-2 rounded-lg cursor-default"
            title="Generating code…"
          >
            <component :is="icons.Code" :size="20" class="text-blue-300 animate-pulse" />
          </button>
          <button
            v-else-if="store.paper2codeJob.status === 'done'"
            @click="store.downloadCodeResult()"
            class="p-2 flex items-center justify-center hover:bg-green-100 rounded-lg transition-colors group"
            title="Download generated code"
          >
            <component :is="icons.Download" :size="20" class="text-green-500 group-hover:text-green-600" />
          </button>

          <!-- Poster Icon -->
          <button
            v-if="store.paper2posterJob.status === 'idle' || store.paper2posterJob.status === 'error'"
            @click="store.paper2posterJob.status === 'error' ? store.resetPosterJob() : store.openPaperSelector('poster')"
            class="p-2 hover:bg-purple-100 rounded-lg transition-colors group"
            title="Paper to Poster"
          >
            <component :is="icons.Image" :size="20" class="text-purple-500 group-hover:text-purple-600" />
          </button>
          <button
            v-else-if="store.paper2posterJob.status === 'running'"
            class="p-2 rounded-lg cursor-default"
            title="Generating poster\u2026"
          >
            <component :is="icons.Image" :size="20" class="text-purple-300 animate-pulse" />
          </button>
          <button
            v-else-if="store.paper2posterJob.status === 'done'"
            @click="store.downloadPosterResult()"
            class="p-2 flex items-center justify-center hover:bg-green-100 rounded-lg transition-colors group"
            title="Download generated poster"
          >
            <component :is="icons.Download" :size="20" class="text-green-500 group-hover:text-green-600" />
          </button>

          <!-- Web Icon -->
          <button
            v-if="store.paper2webJob.status === 'idle' || store.paper2webJob.status === 'error'"
            @click="store.paper2webJob.status === 'error' ? store.resetWebJob() : store.openPaperSelector('web')"
            class="p-2 hover:bg-green-100 rounded-lg transition-colors group"
            title="Paper to Web"
          >
            <component :is="icons.Globe" :size="20" class="text-green-500 group-hover:text-green-600" />
          </button>
          <button
            v-else-if="store.paper2webJob.status === 'running'"
            class="p-2 rounded-lg cursor-default"
            title="Generating website\u2026"
          >
            <component :is="icons.Globe" :size="20" class="text-green-300 animate-pulse" />
          </button>
          <button
            v-else-if="store.paper2webJob.status === 'done'"
            @click="store.downloadWebResult()"
            class="p-2 flex items-center justify-center hover:bg-green-100 rounded-lg transition-colors group"
            title="Download generated website"
          >
            <component :is="icons.Download" :size="20" class="text-green-500 group-hover:text-green-600" />
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
          <div v-if="(store.activeNotebook?.papers?.length ?? 0) === 0" class="text-center py-12">
            <component :is="icons.FileText" :size="48" class="mx-auto mb-3 text-notebook-300" />
            <p class="text-sm text-notebook-500">No papers in this notebook</p>
            <p class="text-xs text-notebook-400 mt-1">Upload papers to get started</p>
          </div>

          <div v-else class="space-y-3">
            <button
              v-for="paper in store.activeNotebook?.papers ?? []"
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
            class="flex-1 px-4 py-2 bg-notebook-800 text-white rounded-lg hover:bg-notebook-900 transition-colors font-medium"
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
  Plus, BookOpen, MoreVertical, Edit, Trash2, Code, Image, Globe, Download, AlertCircle
} from 'lucide-vue-next'

const store = useAppStore()

// Icon components
const icons = {
  Upload, FileText, Settings, PanelRight, Brain, Quote, Layers, Sparkles,
  Send, X, ChevronUp, ChevronLeft, ChevronRight, LogOut,
  Plus, BookOpen, MoreVertical, Edit, Trash2, Code, Image, Globe, Download, AlertCircle
}

// Chat state
const inputMessage = ref('')
const messagesContainer = ref(null)
const chatMessages = computed(() => store.activeNotebook?.messages ?? [])

// Upload state
const uploadState = ref('idle') // 'idle' | 'uploading' | 'success' | 'error'
const uploadError = ref('')

const handleFileUpload = async (event) => {
  const file = event.target.files[0]
  if (!file) return
  uploadState.value = 'uploading'
  uploadError.value = ''
  try {
    await store.uploadPaper(file)
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
const handleCreateNotebook = () => {
  const name = prompt('Enter notebook name:')
  if (name && name.trim()) {
    store.createNotebook(name.trim())
  }
}

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

/* ---- Empty state (no notebook selected) ---- */

.empty-state-root {
  animation: es-fade-rise 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}

@keyframes es-fade-rise {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}

.dot-grid {
  background-image: radial-gradient(circle, #d1d5db 1px, transparent 1px);
  background-size: 28px 28px;
}

.glow-blob {
  background: radial-gradient(ellipse at center, rgba(20, 86, 240, 0.07) 0%, transparent 68%);
}

.glow-blob-accent {
  background: radial-gradient(ellipse at center, rgba(139, 92, 246, 0.07) 0%, transparent 68%);
}

.ring-rotate {
  animation: ring-spin 28s linear infinite;
}

@keyframes ring-spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}

.center-pulse {
  box-shadow: rgba(20, 86, 240, 0.22) 0 0 0 8px, rgba(20, 86, 240, 0.09) 0 0 0 18px;
  animation: center-breathe 4s ease-in-out infinite;
}

@keyframes center-breathe {
  0%, 100% {
    box-shadow: rgba(20, 86, 240, 0.22) 0 0 0 8px,  rgba(20, 86, 240, 0.09) 0 0 0 18px;
  }
  50% {
    box-shadow: rgba(20, 86, 240, 0.32) 0 0 0 10px, rgba(20, 86, 240, 0.13) 0 0 0 24px;
  }
}

.sat-chip {
  box-shadow: rgba(0, 0, 0, 0.07) 0 3px 8px;
}

.sat-float-1 { animation: sat-bob 5.2s ease-in-out infinite 0s; }
.sat-float-2 { animation: sat-bob 6.8s ease-in-out infinite 1.1s; }
.sat-float-3 { animation: sat-bob 5.6s ease-in-out infinite 0.55s; }

@keyframes sat-bob {
  0%, 100% { transform: translateY(0); }
  50%       { transform: translateY(-6px); }
}

.empty-headline {
  font-size: 2.75rem;
  line-height: 1.1;
}

.cta-glow {
  box-shadow: rgba(20, 86, 240, 0.22) 0 4px 20px;
}

.cta-glow:hover {
  box-shadow: none;
}

.es-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 11px;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 999px;
  font-size: 0.7rem;
  color: #5f5f5f;
  box-shadow: rgba(0, 0, 0, 0.04) 0 1px 3px;
}
</style>
