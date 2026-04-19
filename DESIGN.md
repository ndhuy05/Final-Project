# VibeProject — System Design

---

## 1. System Architecture

```mermaid
flowchart TD
    Browser(["Browser"])

    subgraph FE["FRONTEND  ·  Vue 3 + Pinia + Vite"]
        direction TB
        VR["Vue Router\nbeforeEach — auth guard\ntoken check → /login"]
        STORE["Pinia Store — app.js\nAll state · All API calls"]
        AXIOS["Axios Client — client.js\nBearer token injection\n401 → clear token → /login"]
        VR --> STORE --> AXIOS
    end

    subgraph BE["BACKEND  ·  FastAPI  ·  /api/v1"]
        direction TB
        CORS_MW["CORS Middleware\nlocalhost:5173"]
        JWT_DEP["get_current_user\nJWT Bearer dependency"]
        subgraph ROUTERS["Routers"]
            direction LR
            RA["/auth\nregister · login · me"]
            RN["/notebooks\nGET POST PATCH DELETE"]
            RP["/papers\nupload · list · delete · chunks"]
            RC["/chat\nPOST  agentic RAG"]
            RG["/generate\ncode  start · status · cancel · download"]
            RPO["/poster\nposter  start · status · cancel · download"]
        end
        CORS_MW --> JWT_DEP --> ROUTERS
    end

    subgraph AGENTS["AGENT LAYER"]
        direction LR
        EXA["ExtractionAgent\nVLM page OCR\nchunking · embedding\nvector indexing"]
        PLA["PlannerAgent\nLLM action planning\nretrieve / read_metadata"]
        ANA["AnsweringAgent\nVLM answer generation\nN±1 page image context"]
        CDA["CodeAgent\nPaper2Code pipeline\nplan→analyze→generate→ZIP\nbackground thread"]
        POA["PosterAgent\nPaper2Poster pipeline\nsubprocess orchestrator\nJSON-lines stdout"]
    end

    subgraph SVCS["SERVICE LAYER"]
        direction LR
        S_AUTH["auth_service\nbcrypt · JWT sign/verify"]
        S_EMB["embedding_service\nOpenRouter batch embed"]
        S_QDR["qdrant_service\nupsert · cosine search\nscroll · delete"]
        S_RNK["reranker_service\nBAI/bge cross-encoder\nlocal model"]
        S_MEM["memory_store\nSQLAlchemy ORM facade"]
        S_PDF["pdf_service\nPyMuPDF → PIL pages"]
    end

    subgraph DATA["DATA LAYER"]
        direction LR
        DB[("SQLite\nusers · notebooks\npapers · chat")]
        QD[("Qdrant\nnb_{id} collections\nper-notebook vectors")]
        DISK[("Disk\nuploads/ · images/\npaper2code_outputs/\npaper2poster_outputs/")]
    end

    OR["OpenRouter API\nLLM  ·  VLM  ·  Embeddings"]
    SUBP["poster_pipeline/\nrun_poster_job.py\nsubprocess"]

    Browser -->|"HTTP"| FE
    AXIOS -->|"HTTPS · Bearer JWT"| BE

    RA --> S_AUTH --> DB
    RN --> S_MEM --> DB
    RP --> S_PDF & EXA & S_MEM
    RC --> PLA & ANA
    RG --> CDA
    RPO --> POA

    EXA -->|"VLM extraction"| OR
    EXA --> S_EMB & S_QDR
    PLA -->|"LLM planning"| OR
    PLA --> S_EMB & S_QDR & S_RNK
    ANA -->|"VLM answering"| OR
    CDA -->|"LLM code gen"| OR
    CDA --> S_QDR & DISK
    POA --> S_QDR
    POA -->|"subprocess"| SUBP
    SUBP -->|"LLM poster gen"| OR
    SUBP --> DISK

    S_EMB --> OR
    S_QDR --> QD
    S_PDF --> DISK
```

---

## 2. Entity Relationship Diagram

```mermaid
erDiagram
    USERS {
        string  id            PK
        string  email         UK
        string  username      UK
        string  password_hash
        string  full_name
        boolean is_active
        datetime deleted_at
        datetime created_at
        datetime updated_at
    }

    NOTEBOOKS {
        string  id                 PK
        string  user_id            FK
        string  title
        string  description
        string  color_tag
        int     paper_count_cached
        datetime deleted_at
        datetime created_at
        datetime updated_at
    }

    PAPERS {
        string  id                PK
        string  notebook_id       FK
        string  original_filename
        string  storage_path
        float   file_size_mb
        int     page_count
        int     total_chunks
        string  title
        string  authors
        string  year
        string  venue
        text    abstract
        text    description
        json    metadata_json
        datetime deleted_at
        datetime created_at
        datetime updated_at
    }

    CHAT_SESSIONS {
        string  id          PK
        string  notebook_id FK
        string  title
        string  summary
        datetime deleted_at
        datetime created_at
        datetime updated_at
    }

    CHAT_MESSAGES {
        string  id               PK
        string  chat_session_id  FK
        string  user_id          FK
        string  role
        text    content
        string  message_type
        text    citations_json
        datetime created_at
    }

    GENERATION_JOBS {
        string  id            PK
        string  paper_id      FK
        string  job_type
        string  status
        float   progress
        string  output_path
        text    error_message
        datetime started_at
        datetime completed_at
        datetime created_at
        datetime updated_at
    }

    USERS         ||--o{ NOTEBOOKS      : "owns (user_id → users.id CASCADE)"
    NOTEBOOKS     ||--o{ PAPERS         : "contains (notebook_id → notebooks.id CASCADE)"
    NOTEBOOKS     ||--o{ CHAT_SESSIONS  : "has (notebook_id → notebooks.id CASCADE)"
    CHAT_SESSIONS ||--o{ CHAT_MESSAGES  : "contains (chat_session_id → chat_sessions.id CASCADE)"
    USERS         |o--o{ CHAT_MESSAGES  : "authored (user_id → users.id SET NULL)"
    PAPERS        ||--o{ GENERATION_JOBS : "generates (paper_id → papers.id CASCADE)"
```

---

## 3. Class Diagram

```mermaid
%%{init: {"classDiagram": {"curve": "linear"}}}%%
classDiagram

    %% ─── ORM / Domain Models ────────────────────────────────────────────────
    class User {
        +String id
        +String email
        +String username
        +String password_hash
        +String full_name
        +Boolean is_active
        +DateTime deleted_at
        +register(email, username, password_hash, full_name)$ User
        +login() bool
        +logout() void
        +delete() void
    }

    class Notebook {
        +String id
        +String user_id
        +String title
        +String description
        +String color_tag
        +Int paper_count_cached
        +DateTime deleted_at
        +rename(new_title) void
        +delete() void
        +get_papers() List
        +get_chat_sessions() List
    }

    class Paper {
        +String id
        +String notebook_id
        +String original_filename
        +String storage_path
        +Float file_size_mb
        +Int page_count
        +Int total_chunks
        +String title
        +String authors
        +String year
        +String venue
        +Text abstract
        +Text description
        +JSON metadata_json
        +DateTime deleted_at
        +get_metadata() dict
        +rename(new_title) void
        +delete() void
    }

    class ChatSession {
        +String id
        +String notebook_id
        +String title
        +String summary
        +DateTime deleted_at
        +send_message(role, content, citations_json) ChatMessage
        +get_history() List
        +delete() void
    }

    class ChatMessage {
        +String id
        +String chat_session_id
        +String user_id
        +String role
        +Text content
        +String message_type
        +Text citations_json
        +DateTime created_at
    }

    class GenerationJob {
        +String id
        +String paper_id
        +String job_type
        +String status
        +Float progress
        +String output_path
        +Text error_message
        +DateTime started_at
        +DateTime completed_at
        +get_status() dict
        +cancel() void
        +download() FileResponse
    }

    %% ─── ORM associations ───────────────────────────────────────────────────
    User        "1" --> "0..*" Notebook      : owns
    Notebook    "1" --> "0..*" Paper         : contains
    Notebook    "1" --> "0..*" ChatSession   : has
    ChatSession "1" --> "0..*" ChatMessage   : contains
    User        "1" --> "0..*" ChatMessage   : authored
    Paper       "1" --> "0..*" GenerationJob : generates

    %% ─── Agent Base Class ───────────────────────────────────────────────────
    class GenerationAgent {
        <<abstract>>
        +String model
        +String job_type
        +Dict _jobs
        +get_job(job_id) dict
        +get_progress(job_id) float
        +cancel(job_id) void
        +run() str
        -_new_job() str
        -_update(job_id) void
        -_is_cancelled(job_id) bool
    }

    %% ─── Generation Agents ──────────────────────────────────────────────────
    class CodeAgent {
        +String job_type
        +run(notebook_id, paper_id, paper_title) str
        -_fetch_paper_text(notebook_id, paper_id) str
        -plan(paper_content, output_dir, job_id) dict
        -analyze(paper_content, plan, output_dir, job_id) dict
        -generate_code(plan, analysis, output_dir, job_id) dict
        -package_zip(output_dir, zip_base) str
    }

    class PosterAgent {
        +String job_type
        +Lock _lock
        +String _running_job_id
        +is_busy()$ bool
        +run(paper_id, pdf_path, notebook_id) str
        +cancel(job_id) void
    }

    class WebAgent {
        +String job_type
        +run() str
        +plan(paper) dict
        +generate_app(plan) dict
        +package_bundle(files) str
    }

    GenerationAgent <|-- CodeAgent
    GenerationAgent <|-- PosterAgent
    GenerationAgent <|-- WebAgent

    %% ─── Chat / RAG Agents ──────────────────────────────────────────────────
    class ExtractionAgent {
        +String model
        +extract_pages(pages, extract_metadata) List
        +chunk_text(text) List
        +embed_chunks(chunks) List
        +index_to_vector_store(paper_id, page_num, chunks, vectors) void
        -_encode_image(path) str
        -_extract_json(raw) dict
        -_parse_batch_response(raw, pages, extract_metadata) List
    }

    class PlannerAgent {
        +String model
        +plan_actions(question, papers) List
        +read_metadata(paper) dict
        +retrieve(query, paper_id, top_k) List
    }

    class AnsweringAgent {
        +String model
        +generate_answer(question, image_paths, results) str
        +format_citations(chunks) List
    }

    %% ─── Agent–domain dependencies ──────────────────────────────────────────
    Paper         ..> ExtractionAgent  : processed by on upload
    ChatSession   ..> PlannerAgent     : uses for action planning
    ChatSession   ..> AnsweringAgent   : uses for answer generation
    GenerationJob ..> GenerationAgent  : managed by
```

---

## 4. Sequence Diagrams

### 4.1 Register

```mermaid
sequenceDiagram
    actor User
    participant Login as Login.vue
    participant Store as Pinia Store
    participant Axios as Axios Client
    participant AuthAPI as /auth router
    participant AuthSvc as auth_service
    participant DB as SQLite

    User ->> Login: Fill form → click "Create account"
    Login ->> Store: register(username, email, password)
    Store ->> Axios: POST /auth/register
    Axios ->> AuthAPI: {username, email, password}
    AuthAPI ->> DB: SELECT — check email + username uniqueness
    DB -->> AuthAPI: no conflict
    AuthAPI ->> AuthSvc: hash_password(password)
    AuthSvc -->> AuthAPI: bcrypt hash
    AuthAPI ->> DB: INSERT users
    DB -->> AuthAPI: User row
    AuthAPI ->> AuthSvc: create_access_token({sub: user.id})
    AuthSvc -->> AuthAPI: signed JWT
    AuthAPI -->> Axios: 201 {access_token, user}
    Axios -->> Store: response
    Store ->> Store: user.value = {name, email, initials}\nlocalStorage["token"] = access_token
    Store ->> Store: loadNotebooks()
    Store -->> Login: router.push("/")
```

---

### 4.2 Login

```mermaid
sequenceDiagram
    actor User
    participant Login as Login.vue
    participant Store as Pinia Store
    participant Axios as Axios Client
    participant AuthAPI as /auth router
    participant AuthSvc as auth_service
    participant DB as SQLite

    User ->> Login: Fill form → click "Sign in"
    Login ->> Store: login(email, password)
    Store ->> Axios: POST /auth/login
    Axios ->> AuthAPI: {email, password}
    AuthAPI ->> DB: SELECT User WHERE email = ?
    DB -->> AuthAPI: User row
    AuthAPI ->> AuthSvc: verify_password(plain, password_hash)
    AuthSvc -->> AuthAPI: true
    AuthAPI ->> AuthAPI: user.login() → is_active && deleted_at is None
    AuthAPI ->> AuthSvc: create_access_token({sub: user.id})
    AuthSvc -->> AuthAPI: signed JWT
    AuthAPI -->> Axios: 200 {access_token, user}
    Axios -->> Store: response
    Store ->> Store: store token + user
    Store ->> Store: loadNotebooks()
    Store -->> Login: router.push("/")
```

---

### 4.3 Agentic RAG Chat

```mermaid
sequenceDiagram
    actor User
    participant Home as Home.vue
    participant Store as Pinia Store
    participant Axios as Axios Client
    participant ChatAPI as /chat router
    participant Planner as PlannerAgent
    participant Answerer as AnsweringAgent
    participant Embed as embedding_service
    participant Qdrant as qdrant_service
    participant Reranker as reranker_service
    participant OR as OpenRouter API

    User ->> Home: Type question → send
    Home ->> Store: sendMessage(question)
    Store ->> Axios: POST /notebooks/{id}/chat  {question, top_k:50}
    Axios ->> ChatAPI: request + Bearer JWT
    ChatAPI ->> ChatAPI: validate notebook ownership
    ChatAPI ->> ChatAPI: memory_store.get_papers(notebook_id)

    Note over ChatAPI,OR: Step 1 — Plan actions

    ChatAPI ->> Planner: plan_actions(question, papers)
    Planner ->> OR: LLM prompt — which actions to take?
    OR -->> Planner: [{action:"retrieve", query:"..."}, {action:"read_metadata", paper_id:"..."}]
    Planner -->> ChatAPI: action list

    Note over ChatAPI,OR: Step 2 — Execute actions in parallel (asyncio.gather)

    par retrieve action
        ChatAPI ->> Embed: embed_text(query)
        Embed ->> OR: embeddings API
        OR -->> Embed: dense vector
        Embed -->> ChatAPI: query vector
        ChatAPI ->> Qdrant: search(notebook_id, vector, top_k=50, paper_id?)
        Qdrant -->> ChatAPI: top-50 candidate chunks
        ChatAPI ->> Reranker: rerank(query, chunks, top_k=5)
        Reranker -->> ChatAPI: top-5 reranked chunks
    and read_metadata action
        ChatAPI ->> ChatAPI: memory_store.get_paper(notebook_id, paper_id)
        ChatAPI ->> ChatAPI: prepend metadata block to question context
    end

    Note over ChatAPI,OR: Step 3 — Collect page images and generate answer

    ChatAPI ->> ChatAPI: dedup results by (paper_id, page_num)
    ChatAPI ->> ChatAPI: _collect_images(paper_id, best_page)\nload pages N-1, N, N+1
    ChatAPI ->> Answerer: generate_answer(question_with_context, image_paths, top_results)
    Answerer ->> OR: VLM prompt — [page images + metadata + chunks + question]
    OR -->> Answerer: markdown answer with (Page X) citations
    Answerer -->> ChatAPI: answer string
    ChatAPI ->> ChatAPI: format_citations(chunks) → [{id, title, page, excerpt, score}]
    ChatAPI -->> Axios: 200 {content, citations, query_type}

    Axios -->> Store: response
    Store ->> Store: append assistant message + citations to activeNotebook
    Store -->> Home: reactive update
    Home -->> User: Render answer + citation cards
```

---

### 4.4 Paper2Code Generation

```mermaid
sequenceDiagram
    actor User
    participant Home as Home.vue
    participant Store as Pinia Store
    participant Axios as Axios Client
    participant GenAPI as /generate router
    participant Agent as CodeAgent
    participant Qdrant as qdrant_service
    participant OR as OpenRouter API
    participant Disk as Disk

    User ->> Home: Select paper → click "Generate Code"
    Home ->> Store: confirmGeneration()
    Store ->> Axios: POST /notebooks/{nb}/papers/{p}/generate/code
    Axios ->> GenAPI: request + Bearer JWT
    GenAPI ->> GenAPI: validate notebook + paper ownership
    GenAPI ->> Agent: run(notebook_id, paper_id, paper_title)
    Agent ->> Agent: _new_job() → job_id\nstatus = "running"
    Agent ->> Agent: threading.Thread(target=_run_pipeline).start()
    GenAPI -->> Axios: 200 {job_id}
    Axios -->> Store: paper2codeJob.jobId = job_id
    Store ->> Store: setInterval(poll, 2000)

    loop Poll every 2 seconds
        Store ->> Axios: GET /generate/code/{job_id}/status
        Axios ->> GenAPI: request
        GenAPI ->> Agent: get_job(job_id)
        Agent -->> GenAPI: {status, progress, step}
        GenAPI -->> Axios: {status, progress, step}
        Axios -->> Store: update paper2codeJob state
        Store -->> Home: progress bar updates
    end

    Note over Agent,Disk: Background thread execution

    Agent ->> Qdrant: get_all_page_texts(notebook_id, paper_id)
    Qdrant -->> Agent: {page_num: text} dict

    rect rgb(224, 242, 254)
        Note over Agent,OR: Stage 1 — Plan  (4 sequential LLM calls)
        Agent ->> Agent: _is_cancelled(job_id)
        Agent ->> OR: LLM — overall repo plan
        Agent ->> OR: LLM — architecture breakdown
        Agent ->> OR: LLM — task and logic list
        Agent ->> OR: LLM — config.yaml
        Agent ->> Agent: _update(job_id, progress=0.33, step="Planning")
    end

    rect rgb(220, 252, 231)
        Note over Agent,OR: Stage 2 — Analyze  (per-file LLM calls)
        loop For each file in task list
            Agent ->> Agent: _is_cancelled(job_id)
            Agent ->> OR: LLM — analyze module logic
        end
        Agent ->> Agent: _update(job_id, progress=0.66, step="Analyzing")
    end

    rect rgb(254, 249, 219)
        Note over Agent,OR: Stage 3 — Generate  (per-file LLM calls)
        loop For each file in dependency order
            Agent ->> Agent: _is_cancelled(job_id)
            Agent ->> OR: LLM — generate file content
            Agent ->> Disk: write file to output_repo/
        end
        Agent ->> Agent: package_zip() — shutil.make_archive
        Agent ->> Disk: write .zip to paper2code_outputs/
        Agent ->> Agent: _update(job_id, status="done", progress=1.0)
    end

    Store ->> Store: poll detects status = "done"\nclearInterval — stop polling
    Store -->> Home: show Download button

    User ->> Home: Click "Download"
    Home ->> Store: downloadCodeResult()
    Store ->> Axios: GET /generate/code/{job_id}/download  responseType: blob
    Axios ->> GenAPI: request
    GenAPI ->> Disk: FileResponse(output_path)
    Disk -->> Axios: ZIP binary blob
    Axios -->> Store: blob response
    Store ->> Store: URL.createObjectURL(blob)\ncreate invisible a → .click() → URL.revokeObjectURL

    alt User cancels during generation
        User ->> Home: Click "Cancel"
        Home ->> Store: cancelCodeJob()
        Store ->> Axios: POST /generate/code/{job_id}/cancel
        Axios ->> GenAPI: request
        GenAPI ->> Agent: cancel(job_id)
        Agent ->> Agent: _jobs[job_id]["cancelled"] = True
        Note over Agent: Thread checks _is_cancelled()\nbetween LLM calls → raises InterruptedError → exits
        Store ->> Store: resetCodeJob() — clear state + interval
    end
```
