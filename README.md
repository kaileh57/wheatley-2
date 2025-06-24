# Wheatley 2.0

An autonomous AI assistant with MCP (Model Context Protocol) integration, featuring proactive agents, advanced memory systems, and intelligent task routing.

## Features

- **Autonomous Agents**: Think-Act-Observe loop for complex task execution
- **Advanced Memory**: Context-aware memory system with embeddings
- **MCP Integration**: Extensible through Model Context Protocol servers
- **Intelligent Routing**: Automatic classification of simple vs complex tasks
- **Privacy-First**: Local processing with optional cloud features
- **Simple Deployment**: Pure Python, no Docker required

## Architecture

### Core Components

- **Backend**: FastAPI server with SQLite database
- **Agent Manager**: Autonomous agents with sandbox isolation
- **Memory System**: OpenAI embeddings with vector similarity search
- **Task Router**: Gemini-based classification for optimal routing
- **MCP Client**: Integration with external MCP servers

### Technology Stack

- **AI Models**: Claude 4 (reasoning), Gemini 2.5 (search/simple queries), OpenAI (embeddings)
- **Backend**: Python 3.11+, FastAPI, SQLAlchemy, SQLite
- **Authentication**: JWT with bcrypt password hashing
- **MCP**: Model Context Protocol for tool integrations

## Quick Start

### 1. Installation

```bash
cd wheatley-2.0
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Edit .env with your API keys:
# - ANTHROPIC_API_KEY (Claude)
# - GEMINI_API_KEY (Google)
# - OPENAI_API_KEY (embeddings)
# - SECRET_KEY (generate random)
# - USER_PASSWORD_HASH (see below)
```

Generate password hash:
```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt()).decode())"
```

### 3. Install MCP Servers (Optional)

For agent functionality, install MCP servers:
```bash
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-fetch
npm install -g @modelcontextprotocol/server-github
```

### 4. Run the Server

```bash
cd backend
python main.py
```

Server will start at `http://localhost:8000`

## API Usage

### Authentication

```bash
# Get access token
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user&password=your_password"
```

### Simple Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather today?"}'
```

### Complex Task (Creates Agent)

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Research quantum computing and create a summary report"}'
```

### Monitor Agents

```bash
# List all agents
curl http://localhost:8000/agents \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get specific agent status
curl http://localhost:8000/agents/{agent_id} \
  -H "Authorization: Bearer YOUR_TOKEN"

# Stop an agent
curl -X DELETE http://localhost:8000/agents/{agent_id} \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Memory System

```bash
# Get user profile
curl http://localhost:8000/memory/profile \
  -H "Authorization: Bearer YOUR_TOKEN"

# Search memories
curl -X POST http://localhost:8000/memory/search \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning projects"}'
```

## Project Structure

```
wheatley-2.0/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── config.py            # Configuration
│   ├── auth.py              # Authentication
│   ├── models.py            # Database models
│   ├── agent_manager.py     # Agent system
│   ├── memory_system.py     # Advanced memory
│   ├── mcp_client.py        # MCP integration
│   └── task_router.py       # Task routing
├── perplexity-mcp-server/   # Optional Perplexity integration
├── sandboxes/               # Agent workspaces
├── data/                    # Database and embeddings
├── requirements.txt
└── .env.example
```

## How It Works

### Task Flow

1. **Query Processing**: User submits query via API
2. **Memory Context**: System retrieves relevant memories and user profile
3. **Task Classification**: Gemini classifies as simple or complex
4. **Routing**:
   - **Simple**: Direct Gemini response with Google Search grounding
   - **Complex**: Create autonomous agent with MCP tools
5. **Memory Update**: Store interaction and extracted insights

### Agent Execution

1. **Think**: Agent analyzes task and context
2. **Act**: Execute actions using MCP tools
3. **Observe**: Process results and update memory
4. **Loop**: Continue until task completion

### Memory System

- **Short-term**: Recent conversations
- **Long-term**: Embedded memories with similarity search
- **User Profile**: Learned preferences and facts
- **Context Retrieval**: Relevant memories for each query

## MCP Integration

Wheatley 2.0 uses the Model Context Protocol for extensibility:

- **Filesystem**: File operations in agent sandboxes
- **GitHub**: Repository interactions
- **Fetch**: Web content retrieval
- **Perplexity**: Advanced research capabilities (included)

## Development

### Adding New MCP Servers

1. Install the MCP server
2. Add server configuration in `agent_manager.py`
3. Agents can automatically connect based on task analysis

### Extending Memory System

The memory system can be extended with additional categories and extraction logic in `memory_system.py`.

### Custom Agent Behaviors

Modify the think-act-observe loop in `agent_manager.py` for specialized agent behaviors.

## Security

- JWT authentication with bcrypt password hashing
- Agent sandbox isolation
- Environment variable configuration
- No sensitive data in logs

## Monitoring

- Structured logging
- Agent status tracking
- Memory usage monitoring
- API endpoint metrics

## Limitations

- Single user system
- Simplified MCP client (production should use official SDK)
- Basic agent coordination
- Local SQLite database

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details