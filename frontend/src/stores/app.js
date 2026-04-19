import { defineStore } from 'pinia'
import { ref } from 'vue'
import apiClient from '../api/client'

export const useAppStore = defineStore('app', () => {
  // Notebooks and active notebook
  const notebooks = ref([])
  const activeNotebook = ref(null)
  const sidebarView = ref('notebooks') // 'notebooks' or 'sources'
  const notebookMenuOpen = ref(null)
  const paperMenuOpen = ref(null)

  // Legacy state
  const papers = ref([])
  const loading = ref(false)
  const error = ref(null)

  // UI State
  const leftSidebarCollapsed = ref(false)
  const rightPanelVisible = ref(true)
  const selectedCitation = ref(null)
  const selectedSource = ref(null)

  // Paper Generation Feature State
  const showPaperSelector = ref(false)
  const selectedFeature = ref(null) // 'code', 'poster', or 'web'
  const showConfirmation = ref(false)
  const selectedPaperForGeneration = ref(null)

  // Paper2Code job state
  const paper2codeJob = ref({
    status: 'idle',   // 'idle' | 'running' | 'done' | 'error'
    progress: 0,
    step: '',
    jobId: null,
    paperId: null,
    error: null,
  })
  let _pollInterval = null

  // Paper2Poster job state
  const paper2posterJob = ref({
    status: 'idle',   // 'idle' | 'running' | 'done' | 'error'
    progress: 0,
    step: '',
    jobId: null,
    paperId: null,
    error: null,
  })
  let _posterPollInterval = null

  // Chat state (from activeNotebook)
  const messages = ref([])
  const notes = ref([])
  const isTyping = ref(false)

  // User state — null until authenticated
  const user = ref(null)
  const showUserMenu = ref(false)

  // --- Auth Helpers ---

  function _computeInitials(name) {
    if (!name) return '?'
    const parts = name.trim().split(/\s+/)
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
  }

  // --- Auth Actions ---

  async function initApp() {
    // Phase 1: authenticate — failure wipes user and aborts
    try {
      const meRes = await apiClient.get('/auth/me')
      const u = meRes.data
      user.value = {
        name: u.full_name || u.username,
        email: u.email,
        initials: _computeInitials(u.full_name || u.username),
        avatarColor: 'bg-gradient-to-br from-blue-500 to-purple-600'
      }
    } catch {
      user.value = null
      return
    }
    // Phase 2: load notebooks — failure is non-fatal (user stays logged in)
    try {
      await loadNotebooks()
    } catch {
      notebooks.value = []
      activeNotebook.value = null
    }
  }

  async function login(email, password) {
    const res = await apiClient.post('/auth/login', { email, password })
    localStorage.setItem('token', res.data.access_token)
    await initApp()
  }

  async function register(username, email, password) {
    const res = await apiClient.post('/auth/register', { username, email, password })
    localStorage.setItem('token', res.data.access_token)
    await initApp()
  }

  function logout() {
    localStorage.removeItem('token')
    user.value = null
    notebooks.value = []
    activeNotebook.value = null
    showUserMenu.value = false
    window.location.replace('/login')
  }

  // --- Notebook Loading ---

  async function loadNotebooks() {
    const res = await apiClient.get('/notebooks')
    // Backend returns { "notebooks": [...] }, not a bare array
    notebooks.value = res.data.notebooks ?? res.data
    activeNotebook.value = notebooks.value.length > 0 ? notebooks.value[0] : null
  }

  // --- Legacy / Misc Actions ---

  function sendMessage(question) {
    if (!activeNotebook.value || !question.trim()) return
    activeNotebook.value.messages.push({ role: 'user', content: question })
    isTyping.value = true

    apiClient.post(`/notebooks/${activeNotebook.value.id}/chat`, { question })
      .then(res => {
        const { content, citations } = res.data
        activeNotebook.value.messages.push({ role: 'assistant', content, citations })
      })
      .catch(() => {
        activeNotebook.value.messages.push({
          role: 'assistant',
          content: 'Sorry, there was an error contacting the server. Make sure the backend is running.',
          citations: []
        })
      })
      .finally(() => {
        isTyping.value = false
      })
  }

  async function uploadPaper(file) {
    if (!activeNotebook.value) return
    const formData = new FormData()
    formData.append('file', file)
    const res = await apiClient.post(
      `/notebooks/${activeNotebook.value.id}/papers/upload`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    const paper = res.data.paper
    activeNotebook.value.papers.push(paper)
    return res.data
  }

  async function checkHealth() {
    try {
      loading.value = true
      const response = await apiClient.get('/health')
      return response.data
    } catch (err) {
      error.value = err.message
      throw err
    } finally {
      loading.value = false
    }
  }

  function toggleLeftSidebar() {
    leftSidebarCollapsed.value = !leftSidebarCollapsed.value
  }

  function toggleRightPanel() {
    rightPanelVisible.value = !rightPanelVisible.value
  }

  function selectCitation(citation) {
    selectedCitation.value = citation
    rightPanelVisible.value = true
  }

  function selectSource(source) {
    selectedSource.value = source
    rightPanelVisible.value = true
  }

  function toggleUserMenu() {
    showUserMenu.value = !showUserMenu.value
  }

  function openSettings() {
    showUserMenu.value = false
    // TODO: Navigate to settings page
  }

  // --- Notebook Management ---

  function setSidebarView(view) {
    sidebarView.value = view
  }

  function selectNotebook(id) {
    const notebook = notebooks.value.find(n => n.id === id)
    if (notebook) {
      activeNotebook.value = notebook
      sidebarView.value = 'sources'
      notebookMenuOpen.value = null
    }
  }

  async function createNotebook() {
    const res = await apiClient.post('/notebooks', { name: 'Untitled Notebook' })
    notebooks.value.unshift(res.data)
    selectNotebook(res.data.id)
  }

  async function renameNotebook(id, newName) {
    await apiClient.patch(`/notebooks/${id}`, { name: newName })
    const notebook = notebooks.value.find(n => n.id === id)
    if (notebook) {
      notebook.name = newName
      notebookMenuOpen.value = null
    }
  }

  async function deleteNotebook(id) {
    await apiClient.delete(`/notebooks/${id}`)
    const index = notebooks.value.findIndex(n => n.id === id)
    if (index !== -1) {
      notebooks.value.splice(index, 1)
      if (activeNotebook.value?.id === id) {
        activeNotebook.value = notebooks.value.length > 0 ? notebooks.value[0] : null
      }
      notebookMenuOpen.value = null
    }
  }

  function toggleNotebookMenu(id) {
    notebookMenuOpen.value = notebookMenuOpen.value === id ? null : id
  }

  // --- Paper Generation Actions ---

  function openPaperSelector(feature) {
    selectedFeature.value = feature
    showPaperSelector.value = true
  }

  function closePaperSelector() {
    showPaperSelector.value = false
    selectedFeature.value = null
  }

  function selectPaperForGeneration(paper) {
    selectedPaperForGeneration.value = paper
    showPaperSelector.value = false
    showConfirmation.value = true
  }

  function confirmGeneration() {
    if (selectedFeature.value === 'code' && selectedPaperForGeneration.value) {
      const paper = selectedPaperForGeneration.value
      showConfirmation.value = false
      selectedPaperForGeneration.value = null
      selectedFeature.value = null

      paper2codeJob.value = {
        status: 'running',
        progress: 0,
        step: 'Starting…',
        jobId: null,
        paperId: paper.id,
        error: null,
      }

      apiClient.post(`/notebooks/${activeNotebook.value.id}/papers/${paper.id}/generate/code`)
        .then(res => {
          const jobId = res.data.job_id
          paper2codeJob.value.jobId = jobId

          _pollInterval = setInterval(() => {
            apiClient.get(`/generate/code/${jobId}/status`)
              .then(r => {
                const { status, progress, step, error } = r.data
                paper2codeJob.value.progress = progress
                paper2codeJob.value.step = step
                if (status === 'done' || status === 'error' || status === 'cancelled') {
                  paper2codeJob.value.status = status
                  paper2codeJob.value.error = error
                  clearInterval(_pollInterval)
                  _pollInterval = null
                }
              })
              .catch(() => {
                paper2codeJob.value.status = 'error'
                paper2codeJob.value.error = 'Failed to poll status.'
                clearInterval(_pollInterval)
                _pollInterval = null
              })
          }, 2000)
        })
        .catch(err => {
          paper2codeJob.value.status = 'error'
          paper2codeJob.value.error = err?.response?.data?.detail || 'Failed to start generation.'
        })
      return
    }

    if (selectedFeature.value === 'poster' && selectedPaperForGeneration.value) {
      const paper = selectedPaperForGeneration.value
      showConfirmation.value = false
      selectedPaperForGeneration.value = null
      selectedFeature.value = null

      paper2posterJob.value = {
        status: 'running',
        progress: 0,
        step: 'Starting…',
        jobId: null,
        paperId: paper.id,
        error: null,
      }

      apiClient.post(`/notebooks/${activeNotebook.value.id}/papers/${paper.id}/generate/poster`)
        .then(res => {
          const jobId = res.data.job_id
          paper2posterJob.value.jobId = jobId

          _posterPollInterval = setInterval(() => {
            apiClient.get(`/generate/poster/${jobId}/status`)
              .then(r => {
                const { status, progress, step, error } = r.data
                paper2posterJob.value.progress = progress
                paper2posterJob.value.step = step
                if (status === 'done' || status === 'error' || status === 'cancelled') {
                  paper2posterJob.value.status = status
                  paper2posterJob.value.error = error
                  clearInterval(_posterPollInterval)
                  _posterPollInterval = null
                }
              })
              .catch(() => {
                paper2posterJob.value.status = 'error'
                paper2posterJob.value.error = 'Failed to poll status.'
                clearInterval(_posterPollInterval)
                _posterPollInterval = null
              })
          }, 2000)
        })
        .catch(err => {
          paper2posterJob.value.status = 'error'
          paper2posterJob.value.error = err?.response?.data?.detail || 'Failed to start generation.'
        })
      return
    }

    // Placeholder for other features (web)
    console.log(`Generating ${selectedFeature.value} for paper: ${selectedPaperForGeneration.value?.title}`)
    showConfirmation.value = false
    selectedPaperForGeneration.value = null
    selectedFeature.value = null
  }

  function cancelGeneration() {
    showConfirmation.value = false
    showPaperSelector.value = true
  }

  function cancelCodeJob() {
    const jobId = paper2codeJob.value.jobId
    if (!jobId) {
      resetCodeJob()
      return
    }
    apiClient.post(`/generate/code/${jobId}/cancel`)
      .catch(() => {}) // best-effort
      .finally(() => resetCodeJob())
  }

  async function downloadCodeResult() {
    const jobId = paper2codeJob.value.jobId
    if (!jobId) return
    const res = await apiClient.get(`/generate/code/${jobId}/download`, { responseType: 'blob' })
    const url = URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = url
    a.download = `paper2code_${jobId}.zip`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  function resetCodeJob() {
    if (_pollInterval) {
      clearInterval(_pollInterval)
      _pollInterval = null
    }
    paper2codeJob.value = { status: 'idle', progress: 0, step: '', jobId: null, paperId: null, error: null }
  }

  function cancelPosterJob() {
    const jobId = paper2posterJob.value.jobId
    if (!jobId) {
      resetPosterJob()
      return
    }
    apiClient.post(`/generate/poster/${jobId}/cancel`)
      .catch(() => {}) // best-effort
      .finally(() => resetPosterJob())
  }

  async function downloadPosterResult() {
    const jobId = paper2posterJob.value.jobId
    if (!jobId) return
    const res = await apiClient.get(`/generate/poster/${jobId}/download`, { responseType: 'blob' })
    const url = URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = url
    a.download = `paper2poster_${jobId}.pptx`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  function resetPosterJob() {
    if (_posterPollInterval) {
      clearInterval(_posterPollInterval)
      _posterPollInterval = null
    }
    paper2posterJob.value = { status: 'idle', progress: 0, step: '', jobId: null, paperId: null, error: null }
  }

  // --- Paper Management Actions ---

  function togglePaperMenu(paperId) {
    paperMenuOpen.value = paperMenuOpen.value === paperId ? null : paperId
  }

  function renamePaper(paperId, newTitle) {
    if (activeNotebook.value) {
      const paper = activeNotebook.value.papers.find(p => p.id === paperId)
      if (paper) {
        paper.title = newTitle
        paperMenuOpen.value = null
      }
    }
  }

  async function deletePaper(paperId) {
    if (activeNotebook.value) {
      const index = activeNotebook.value.papers.findIndex(p => p.id === paperId)
      if (index !== -1) {
        try {
          await apiClient.delete(`/notebooks/${activeNotebook.value.id}/papers/${paperId}`)
        } catch (e) {
          console.warn('Backend delete failed:', e)
        }
        activeNotebook.value.papers.splice(index, 1)
        paperMenuOpen.value = null
        if (selectedSource.value?.id === paperId) {
          selectedSource.value = null
        }
      }
    }
  }

  return {
    // State
    notebooks,
    activeNotebook,
    sidebarView,
    notebookMenuOpen,
    paperMenuOpen,
    papers,
    loading,
    error,
    leftSidebarCollapsed,
    rightPanelVisible,
    selectedCitation,
    selectedSource,
    messages,
    notes,
    isTyping,
    user,
    showUserMenu,
    showPaperSelector,
    selectedFeature,
    showConfirmation,
    selectedPaperForGeneration,
    paper2codeJob,
    paper2posterJob,
    // Actions
    initApp,
    login,
    register,
    loadNotebooks,
    checkHealth,
    toggleLeftSidebar,
    toggleRightPanel,
    selectCitation,
    selectSource,
    toggleUserMenu,
    logout,
    openSettings,
    setSidebarView,
    selectNotebook,
    createNotebook,
    renameNotebook,
    deleteNotebook,
    toggleNotebookMenu,
    togglePaperMenu,
    renamePaper,
    deletePaper,
    openPaperSelector,
    closePaperSelector,
    selectPaperForGeneration,
    confirmGeneration,
    cancelGeneration,
    sendMessage,
    uploadPaper,
    downloadCodeResult,
    resetCodeJob,
    cancelCodeJob,
    cancelPosterJob,
    downloadPosterResult,
    resetPosterJob,
  }
})
