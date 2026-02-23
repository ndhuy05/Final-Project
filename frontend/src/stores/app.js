import { defineStore } from 'pinia'
import { ref } from 'vue'
import apiClient from '../api/client'

export const useAppStore = defineStore('app', () => {
  // Notebooks and active notebook
  const notebooks = ref([
    {
      id: 1,
      name: 'ML Research Papers',
      createdAt: '2026-02-01',
      papers: [
        { id: 1, title: 'Attention Is All You Need', authors: 'Vaswani et al.', abstract: 'We propose a new simple network architecture...' },
        { id: 2, title: 'BERT: Pre-training of Deep Bidirectional Transformers', authors: 'Devlin et al.', abstract: 'We introduce a new language representation model...' }
      ],
      messages: [],
      notes: [
        { id: 1, title: 'Transformer Architecture', content: 'Key innovation: self-attention mechanism that processes entire sequences in parallel...', date: '2 days ago' },
        { id: 2, title: 'BERT Training', content: 'Uses masked language modeling and next sentence prediction for pre-training...', date: '1 week ago' }
      ]
    },
    {
      id: 2,
      name: 'Thesis Literature Review',
      createdAt: '2026-02-05',
      papers: [],
      messages: [],
      notes: []
    },
    {
      id: 3,
      name: 'Computer Vision Papers',
      createdAt: '2026-02-07',
      papers: [],
      messages: [],
      notes: []
    }
  ])
  const activeNotebook = ref(notebooks.value[0])
  const sidebarView = ref('notebooks') // 'notebooks' or 'sources'
  const notebookMenuOpen = ref(null)
  const paperMenuOpen = ref(null)

  // Legacy state (now computed from activeNotebook)
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

  // Chat state (now from activeNotebook)
  const messages = ref([])
  const notes = ref([])
  const isTyping = ref(false)

  // Mock AI response bank
  const mockResponses = [
    {
      keywords: ['transformer', 'attention', 'self-attention'],
      content: `The **Transformer architecture** introduced in "Attention Is All You Need" (Vaswani et al., 2017) is a sequence-to-sequence model built entirely on attention mechanisms.\n\nKey components:\n- **Multi-Head Self-Attention**: Allows the model to attend to different parts of the input simultaneously\n- **Positional Encodings**: Inject sequence order since there's no recurrence\n- **Feed-Forward Layers**: Applied identically to each position\n\nThis design enables full parallelization during training, which was a major breakthrough over RNNs.`
    },
    {
      keywords: ['bert', 'pre-training', 'bidirectional'],
      content: `**BERT** (Bidirectional Encoder Representations from Transformers) by Devlin et al. is a landmark pre-training approach.\n\nBERT is trained with two objectives:\n1. **Masked Language Modeling (MLM)**: Randomly masks 15% of tokens and predicts them\n2. **Next Sentence Prediction (NSP)**: Classifies whether two sentences are consecutive\n\nThe bidirectional nature allows BERT to capture context from both left and right, making it far more powerful than unidirectional models like GPT-1.`
    },
    {
      keywords: ['vision', 'image', 'cnn', 'convolutional', 'resnet', 'vit'],
      content: `Computer vision has seen dramatic progress through deep learning.\n\n**Key milestones:**\n- **AlexNet (2012)**: First deep CNN to win ImageNet, using ReLU and dropout\n- **ResNet (2015)**: Introduced skip connections to train very deep networks (up to 152 layers)\n- **Vision Transformer (ViT, 2020)**: Applied the Transformer architecture directly to image patches\n\nModern vision models increasingly borrow from NLP, with ViT and its variants now outperforming CNNs on many benchmarks.`
    },
    {
      keywords: ['train', 'training', 'optimize', 'loss', 'gradient'],
      content: `Training deep neural networks involves several key considerations:\n\n- **Loss Function**: Measures prediction error (e.g., cross-entropy for classification)\n- **Optimizer**: SGD, Adam, and AdamW are most common. Adam uses adaptive learning rates per parameter\n- **Regularization**: Dropout, weight decay, and batch normalization prevent overfitting\n- **Learning Rate Scheduling**: Warmup followed by decay is standard in Transformer training\n\nPaper-specific training details are usually found in the "Experiments" section.`
    },
    {
      keywords: [],
      content: `Based on the papers in this notebook, here is a summary of what I found:\n\nThe research covers advanced topics in machine learning and AI. The authors present empirical results demonstrating significant improvements over prior baselines.\n\n**Common themes across papers:**\n- Architecture innovations that improve efficiency or accuracy\n- Large-scale pre-training followed by fine-tuning on downstream tasks\n- Ablation studies validating each component's contribution\n\nWould you like me to dive deeper into a specific aspect?`
    }
  ]

  function getMockResponse(question) {
    const q = question.toLowerCase()
    const match = mockResponses.find(r => r.keywords.some(k => q.includes(k)))
    const response = match || mockResponses[mockResponses.length - 1]
    const papers = activeNotebook.value?.papers ?? []
    const citations = papers.slice(0, 2).map((p, i) => ({
      id: i + 1,
      title: p.title,
      excerpt: p.abstract || 'See paper for details.'
    }))
    return { content: response.content, citations }
  }

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
    return paper
  }

  // User state
  const user = ref({
    name: 'John Doe',
    email: 'john.doe@example.com',
    initials: 'JD',
    avatarColor: 'bg-gradient-to-br from-blue-500 to-purple-600'
  })
  const showUserMenu = ref(false)

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

  function setRightPanelMode(mode) {
    rightPanelMode.value = mode
    if (!rightPanelVisible.value) {
      rightPanelVisible.value = true
    }
  }

  function selectCitation(citation) {
    selectedCitation.value = citation
    rightPanelMode.value = 'preview'
    rightPanelVisible.value = true
  }

  function selectSource(source) {
    selectedSource.value = source
    rightPanelMode.value = 'preview'
    rightPanelVisible.value = true
  }

  function toggleUserMenu() {
    showUserMenu.value = !showUserMenu.value
  }

  function logout() {
    console.log('Logout clicked')
    showUserMenu.value = false
    // TODO: Implement actual logout logic
  }

  function openSettings() {
    console.log('Settings clicked')
    showUserMenu.value = false
    // TODO: Navigate to settings page
  }

  // Notebook management
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

  function createNotebook() {
    const newId = Math.max(...notebooks.value.map(n => n.id)) + 1
    const newNotebook = {
      id: newId,
      name: 'Untitled Notebook',
      createdAt: new Date().toISOString().split('T')[0],
      papers: [],
      messages: [],
      notes: []
    }
    notebooks.value.unshift(newNotebook)
    selectNotebook(newId)
  }

  function renameNotebook(id, newName) {
    const notebook = notebooks.value.find(n => n.id === id)
    if (notebook) {
      notebook.name = newName
      notebookMenuOpen.value = null
    }
  }

  function deleteNotebook(id) {
    const index = notebooks.value.findIndex(n => n.id === id)
    if (index !== -1) {
      notebooks.value.splice(index, 1)
      // If deleting active notebook, switch to first available
      if (activeNotebook.value.id === id && notebooks.value.length > 0) {
        activeNotebook.value = notebooks.value[0]
      }
      notebookMenuOpen.value = null
    }
  }

  function toggleNotebookMenu(id) {
    notebookMenuOpen.value = notebookMenuOpen.value === id ? null : id
  }

  // Paper Generation Actions
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
    // Placeholder for future generation logic
    console.log(`Generating ${selectedFeature.value} for paper: ${selectedPaperForGeneration.value.title}`)
    showConfirmation.value = false
    selectedPaperForGeneration.value = null
    selectedFeature.value = null
  }

  function cancelGeneration() {
    showConfirmation.value = false
    showPaperSelector.value = true
  }

  // Paper Management Actions
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

  function deletePaper(paperId) {
    if (activeNotebook.value) {
      const index = activeNotebook.value.papers.findIndex(p => p.id === paperId)
      if (index !== -1) {
        activeNotebook.value.papers.splice(index, 1)
        paperMenuOpen.value = null
        // Clear selected source if it was deleted
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
    // Actions
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
    uploadPaper
  }
})
