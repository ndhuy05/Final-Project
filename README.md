# VibeProject - Research Paper Q&A System

A NotebookLM-like application for querying and analyzing research papers with advanced features including multi-paper queries, code generation from papers, and conference poster generation.

## ✅ Setup Status

- [x] Project structure initialized
- [x] Backend (FastAPI) structure created  
- [x] Frontend (Vue 3 + Vite) configured
- [x] Python virtual environment created
- [x] Backend dependencies installed
- [ ] PostgreSQL database models
- [ ] Qdrant vector database setup

## Tech Stack

### Frontend
- **Vue 3** - Progressive JavaScript framework
- **Vite** - Next-generation frontend tooling
- **Vue Router** - Official router for Vue.js
- **Pinia** - State management
- **Axios** - HTTP client
- **TailwindCSS** - Utility-first CSS framework

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **SQLAlchemy** - SQL toolkit and ORM
- **Alembic** - Database migration tool
- **Pydantic** - Data validation

### Databases
- **PostgreSQL** - Primary relational database
- **Qdrant** - Vector database for embeddings

### ML/AI
- **ColPali** - Document embeddings (HuggingFace)
- **LLM Integration** - Flexible provider support

## Project Structure

```
VibeProject/
├── backend/          # FastAPI backend application
│   ├── app/         # Main application code
│   ├── alembic/     # Database migrations
│   └── tests/       # Backend tests
├── frontend/         # Vue 3 + Vite frontend
│   ├── src/         # Source code
│   └── public/      # Static assets
└── README.md        # This file
```

## Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- **PostgreSQL 14+**
- **Qdrant** (Docker or cloud instance)

## Getting Started

### Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/MacOS
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Database Setup

```bash
# Create PostgreSQL database
createdb vibeproject

# Run migrations
cd backend
alembic upgrade head
```

### Qdrant Setup

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant
```

## Development

- Backend API: http://localhost:8000
- Frontend: http://localhost:5173
- API Documentation: http://localhost:8000/docs

## Features (Planned)

- 📄 Multi-paper query system
- 🤖 AI-powered Q&A on research papers
- 💻 Code generation from paper algorithms
- 🎨 Conference poster generation
- 🔍 Advanced citation handling
- 📚 Paper library management

## Environment Variables

Create a `.env` file in the backend directory:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/vibeproject
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## License

TBD

## Contributing

TBD
