# VibeProject - NotebookLM-Inspired Research Paper Q&A System

A modern web application for managing, querying, and analyzing research papers. Features a clean NotebookLM-inspired UI with notebook management, paper organization, AI-powered chat, and paper generation tools (Code, Poster, Web).

## 🎯 Project Status

**Frontend: ✅ Fully Functional** | **Backend: 🚧 In Progress**

The frontend is complete with a production-ready UI, mock data, and all interactive features. The backend integration (FastAPI, PostgreSQL, Qdrant) is planned for future implementation.

## ✨ Implemented Features

### 📚 Notebook Management System
- **Multiple Notebooks**: Create, rename, delete notebooks (ChatGPT-style sidebar)
- **Isolated Data**: Each notebook has its own sources, chat history, and notes
- **Smart Navigation**: Toggle between "Your Notebooks" and "Sources" views
- **Default View**: Opens to notebooks list on startup

### 📄 Document Management
- **Sources Tab**: View, organize, and manage papers per notebook
- **Paper Actions**: Rename and delete papers via 3-dot menu
- **Upload Zone**: Drag-and-drop interface (UI ready, upload logic pending)
- **Paper Preview**: Click citations or sources to view excerpts

### 💬 Chat Interface
- **AI Q&A**: Chat interface for asking questions about papers
- **Markdown Support**: Rich text rendering with `marked.js`
- **Citation Badges**: Clickable [1], [2] badges linked to sources
- **Message History**: Persistent per-notebook chat history
- **Welcome Cards**: Feature highlights on empty state

### 🧪 Lab (Generation Features)
- **Paper to Code**: Generate code from paper algorithms
- **Paper to Poster**: Create conference posters
- **Paper to Web**: Generate web pages
- **Paper Selection Modal**: Choose source papers from notebook
- **Confirmation Flow**: Yes/No dialog before generation

### 🎨 UI/UX
- **Three-Column Layout**: Left sidebar (notebooks/sources) | Center (chat) | Right sidebar (Lab)
- **Collapsible Sidebars**: Smooth animations (64px collapsed, full width expanded)
- **User Profile**: Avatar with initials, dropdown menu (Settings/Logout)
- **Google-Style Aesthetic**: Clean whites, soft grays, subtle shadows
- **Inter Font**: Professional typography (Google Sans alternative)
- **Responsive Design**: Adapts to sidebar states

## 🛠️ Tech Stack

### Frontend (Production Ready)
- **Vue 3** (Composition API with `<script setup>`)
- **Vite 7.3.1** - Lightning-fast dev server
- **Pinia** - Centralized state management
- **Vue Router** - SPA routing
- **Tailwind CSS v3** - Utility-first styling
- **Lucide Vue Next** - Icon system
- **Marked.js** - Markdown rendering
- **Inter Font** - Typography (via Google Fonts)

### Backend (Planned)
- **FastAPI** - Python web framework
- **Uvicorn** - ASGI server
- **SQLAlchemy** - ORM for PostgreSQL
- **Alembic** - Database migrations
- **Pydantic** - Data validation

### Databases (Planned)
- **PostgreSQL** - Relational data (users, notebooks, papers)
- **Qdrant** - Vector database for semantic search
- **ColPali (HuggingFace)** - Document embeddings

## 📁 Project Structure

```
VibeProject/
├── frontend/                      # Vue 3 Application
│   ├── src/
│   │   ├── views/
│   │   │   └── Home.vue          # Main UI (3-column layout, ~640 lines)
│   │   ├── stores/
│   │   │   └── app.js            # Pinia store (state + actions, ~260 lines)
│   │   ├── router/
│   │   │   └── index.js          # Vue Router config
│   │   ├── App.vue               # Root component
│   │   ├── main.js               # App entry point
│   │   └── style.css             # Tailwind + custom styles
│   ├── index.html                # HTML entry + Inter font
│   ├── tailwind.config.js        # Tailwind config (custom colors)
│   ├── package.json              # Dependencies
│   └── vite.config.js            # Vite configuration
├── backend/                       # FastAPI Application (structure ready)
│   ├── app/
│   │   ├── main.py               # FastAPI app + health endpoint
│   │   └── __init__.py
│   ├── requirements.txt          # Python dependencies
│   └── venv/                     # Python virtual environment
├── .gitignore                    # Comprehensive ignore rules
└── README.md                     # This file
```

## 🗂️ Key Files Explained

### Frontend Core Files

#### **`frontend/src/stores/app.js`** (Pinia Store)
Central state management for the entire application:
- **Notebooks Array**: Mock data with 3 notebooks (ML Research, Thesis Review, Computer Vision)
- **Active Notebook**: Currently selected notebook reference
- **Sidebar States**: `leftSidebarCollapsed`, `rightPanelVisible`, `sidebarView` (notebooks/sources)
- **Paper Generation**: Modal states for Lab features
- **User State**: Profile info with initials-based avatar
- **Actions**: CRUD for notebooks/papers, UI toggles, generation workflows

**Key State Variables:**
```javascript
notebooks: [{ id, name, createdAt, papers[], messages[], notes[] }]
activeNotebook: ref(notebooks[0])
sidebarView: 'notebooks' | 'sources'
paperMenuOpen, notebookMenuOpen: Track dropdown states
showPaperSelector, showConfirmation: Modal visibility
```

#### **`frontend/src/views/Home.vue`** (Main Component)
The entire UI in one component (~640 lines):
- **Lines 1-237**: Left sidebar (notebooks list, sources, user profile, collapsed view)
- **Lines 260-374**: Center chat interface (messages, input bar, welcome cards)
- **Lines 376-467**: Right sidebar "Lab" (generation features, collapsed icons)
- **Lines 470-545**: Modals (paper selector, confirmation dialog)
- **Lines 588-665**: Script setup (handlers, icons, markdown rendering)

**Key Sections:**
- Collapsible sidebars with transition animations
- Notebook/paper 3-dot menus with rename/delete
- Tab switching with directional slide animations
- Feature cards (Code, Poster, Web) in grid layout

#### **`frontend/tailwind.config.js`**
Custom Tailwind configuration:
- **Font Family**: Inter as default sans-serif
- **Custom Colors**: `notebook-50` through `notebook-900` (gray palette)
- **Content Paths**: Configured for Vue files

#### **`frontend/src/style.css`**
Global styles:
- Tailwind directives
- Custom scrollbar utilities (`.scrollbar-thin`)
- Base styles for HTML/body

## 🎨 Design System

### Color Palette
```javascript
notebook-50:  '#f9fafb'  // Lightest background
notebook-100: '#f3f4f6'  // Hover states
notebook-200: '#e5e7eb'  // Borders
notebook-300: '#d1d5db'  // Disabled text
notebook-600: '#4b5563'  // Secondary text
notebook-900: '#111827'  // Primary text
```

### Feature Colors
- **Code Generation**: Blue (`text-blue-500`)
- **Poster Generation**: Purple (`text-purple-500`)
- **Web Generation**: Green (`text-green-500`)

### Typography (Inter Font)
- **Headings**: `text-xl font-semibold` (20px, 600 weight)
- **Body**: `text-sm` (14px, 400 weight)
- **Labels**: `text-xs` (12px, 400 weight)
- **Buttons**: `text-sm font-medium` (14px, 500 weight)

### Component Patterns
- **Active Tab**: `bg-white shadow-sm text-notebook-900`
- **Inactive Tab**: `text-notebook-600 hover:bg-notebook-100`
- **Cards**: `border border-notebook-200 rounded-lg hover:shadow-md`
- **Modals**: `rounded-2xl shadow-2xl` with backdrop `bg-black/50`

## 🚀 Getting Started

### Prerequisites
- **Node.js 20.19+ or 22.12+** (v22.3.0 currently used)
- **npm** (comes with Node.js)
- Python 3.9+ (for backend, when implemented)

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# App will run at http://localhost:5173 or 5174/5175 if port is in use
```

### Current Development URLs
- **Frontend**: http://localhost:5173 (or next available port)
- **Backend**: Not yet running (planned: http://localhost:8000)
- **API Docs**: Planned at http://localhost:8000/docs

### Mock Data
The app runs entirely with mock data defined in `frontend/src/stores/app.js`:
- 3 pre-populated notebooks
- Sample papers with metadata
- Demo chat messages
- Example notes

## 🏗️ Architecture Decisions

### State Management
- **Centralized Pinia Store**: All state in `app.js` for simplicity
- **Notebook-Centric**: Each notebook is self-contained with its own data
- **No Persistence**: Currently in-memory (localStorage/backend planned)

### Component Structure
- **Single-File Component**: Entire UI in `Home.vue` (could be split later)
- **Composition API**: Modern Vue 3 with `<script setup>`
- **Template-Driven**: Minimal logic in templates, handlers in script

### Styling Approach
- **Tailwind Utility Classes**: No CSS modules or scoped styles
- **Custom Color System**: `notebook-*` prefix for brand colors
- **No Component Library**: Pure Tailwind + Lucide icons

### Animation Strategy
- **Vue Transitions**: For modals and tab switching
- **Tailwind Transitions**: For hovers and sidebar collapse
- **Duration**: 300ms for sidebars, 100-200ms for modals

### Data Flow
```
User Action → Event Handler → Pinia Action → State Update → Reactive UI Update
```

Example: Click notebook → `selectNotebook()` → `activeNotebook.value = notebook` → UI re-renders

## 🔧 Development Notes

### Known Issues
- **Node Version Warning**: App works but recommends Node 20.19+ or 22.12+
- **Port Conflicts**: Vite auto-increments if 5173 is taken
- **No Persistence**: Refresh loses all changes (by design for now)

### Technical Debt
- Home.vue is large (~640 lines) - could split into components
- Mock data in store - needs backend API integration
- No error handling for API calls (no API yet)
- No tests written yet

### Future Backend Integration Points
1. **Notebooks API**: CRUD endpoints replacing mock data
2. **Papers API**: Upload, process, embed documents
3. **Chat API**: LLM integration for Q&A
4. **Generation API**: Code/Poster/Web generation logic
5. **Auth API**: User authentication and sessions

## 📝 Key Patterns & Conventions

### Naming Conventions
- **Components**: PascalCase (`Home.vue`)
- **State Variables**: camelCase (`activeNotebook`)
- **Actions**: camelCase verbs (`selectNotebook`, `toggleSidebar`)
- **CSS Classes**: kebab-case (Tailwind standard)

### Icon Usage
All icons from Lucide Vue Next:
```javascript
import { Code, Image, Globe, FileText, ... } from 'lucide-vue-next'
```

### Event Handling
- Use `@click.stop` to prevent event bubbling (e.g., 3-dot menus)
- Use `@click.self` for modal backdrop close

### Conditional Rendering
- `v-if` for modals and major DOM changes
- `v-show` avoided (prefer `v-if` for cleaner code)
- `:class` for dynamic styling

## 🎯 Next Steps (Backend Implementation)

1. **Database Models**: Define SQLAlchemy models for notebooks, papers, messages
2. **API Endpoints**: Implement FastAPI routes matching frontend expectations
3. **Vector Search**: Set up Qdrant for semantic paper search
4. **LLM Integration**: Connect to GPT/Claude for chat functionality
5. **File Processing**: PDF parsing and text extraction
6. **Authentication**: User accounts and sessions
7. **Generation Logic**: Implement actual Code/Poster/Web generation

## 🧪 Testing Strategy (Future)

- **Frontend**: Vitest for component tests
- **Backend**: Pytest for API tests
- **E2E**: Playwright for full user flows
- **Load Testing**: Locust for performance testing

## 📚 Learning Resources

- [Vue 3 Docs](https://vuejs.org/)
- [Pinia Docs](https://pinia.vuejs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Lucide Icons](https://lucide.dev/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

## 🤝 Contributing

Currently in active development. Contribution guidelines TBD.

## 📄 License

TBD

---

**Last Updated**: February 2026  
**Status**: Frontend complete, backend pending  
**Contact**: TBD
