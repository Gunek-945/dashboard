# MayaCode User Dashboard - Full Stack Application

A comprehensive user dashboard application with React frontend and Python FastAPI backend, designed to help users manage their integration process with AI assistance.

## Project Structure

```
mayacode-dashboard-fullstack/
├── client/                  # React frontend application
│   ├── components/          # React components
│   ├── services/           # Frontend API services
│   ├── package.json        # Frontend dependencies
│   ├── vite.config.ts      # Vite configuration
│   ├── tsconfig.json       # TypeScript configuration
│   ├── index.html          # HTML template
│   └── App.tsx             # Main React application
├── server/                 # Python FastAPI backend
│   ├── data/               # Sample data
│   │   ├── __init__.py    # Package initializer
│   │   └── user_data.py   # User data definitions
│   ├── main.py            # FastAPI application
│   ├── start_fastapi.py   # Startup script
│   ├── requirements.txt   # Python dependencies
│   └── .env              # Environment variables
└── README.md
```

## Features

### Frontend
- **React 19** with TypeScript
- **Progressive Loading**: Initially shows 2-3 entries, loads more on demand
- **Multi-language Support**: English, Spanish, German
- **AI Chat Integration**: Google Gemini-powered assistant
- **Responsive Design**: Mobile-first approach
- **Task Management**: Track progress on various integration tasks
- **Document Management**: Organize and manage important documents

### Backend
- **Python FastAPI** RESTful API
- **Progressive Loading**: API supports initial load + pagination
- **CORS Support**: Configured for frontend integration
- **Async Support**: Full async/await support for better performance
- **Auto Documentation**: Built-in Swagger/OpenAPI documentation
- **Type Safety**: Pydantic models for request/response validation

## Quick Start

### 1. Backend Setup (FastAPI - Recommended)

```bash
# Navigate to server directory
cd server

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with:
PORT=3001
NODE_ENV=development
CORS_ORIGIN=http://localhost:5173
API_VERSION=v1
HOST=0.0.0.0

# Start the FastAPI server
python start_fastapi.py
# OR
python main.py
# OR using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 3001 --reload
```

The backend will run on `http://localhost:3001`
- API Documentation: `http://localhost:3001/docs` (Swagger UI)
- Alternative docs: `http://localhost:3001/redoc`

### 2. Frontend Setup

```bash
# Navigate to client directory
cd client

# Install dependencies
npm install

# Set up environment variables
# Create .env.local file with:
VITE_API_BASE_URL=http://localhost:3001/api/v1
VITE_GEMINI_API_KEY=your_gemini_api_key_here

# Start the frontend development server
npm run dev
```

The frontend will run on `http://localhost:5173`

## API Endpoints

### Base URL: `http://localhost:3001/api/v1`

- **GET** `/user/profile` - Get user profile data
- **GET** `/user/actions` - Get user actions with progressive loading
- **GET** `/user/actions/{action_id}` - Get specific action details
- **GET** `/user/documents` - Get user documents with progressive loading
- **GET** `/user/dashboard-summary` - Get dashboard overview data
- **GET** `/health` - Health check endpoint

### Progressive Loading

The API implements progressive loading:
- Initial load: Returns first 2-3 items (`loadMore=false`)
- Load more: Returns paginated results (`loadMore=true`)
- Supports filtering by status and type

Example:
```bash
# Initial load (first 3 actions)
curl "http://localhost:3001/api/v1/user/actions"

# Load more (next page)
curl "http://localhost:3001/api/v1/user/actions?page=2&loadMore=true"

# Filter by status
curl "http://localhost:3001/api/v1/user/actions?status=In%20Progress"
```

## Environment Variables

### Backend (.env)
```env
PORT=3001
NODE_ENV=development
CORS_ORIGIN=http://localhost:5173
API_VERSION=v1
HOST=0.0.0.0
```

### Frontend (.env.local)
```env
VITE_API_BASE_URL=http://localhost:3001/api/v1
VITE_GEMINI_API_KEY=your_gemini_api_key_here
```

## Data Flow

1. **Initial Load**: Frontend fetches dashboard summary and first 2-3 items
2. **Progressive Loading**: User can load more items as needed
3. **Filtering**: Backend handles status-based filtering
4. **Real-time Updates**: AI chat provides contextual assistance

## Development

### Adding New Features

1. **Backend**: Add endpoints in `server/main.py`
2. **Frontend**: Add API calls in `client/services/api.ts`
3. **Components**: Create/update React components in `client/components/`
4. **Types**: Update TypeScript interfaces in `client/types.ts`

### Testing

```bash
# Test backend endpoints
curl http://localhost:3001/health
curl http://localhost:3001/api/v1/user/profile

# View API documentation
# Visit http://localhost:3001/docs in your browser

# Frontend testing (from client directory)
cd client
npm run build
npm run preview
```

## Key Features Implemented

✅ **Progressive Loading**: Shows 2-3 entries initially, loads more on demand  
✅ **FastAPI Backend**: Modern Python API with auto-documentation  
✅ **Frontend Integration**: React app connected to backend  
✅ **Error Handling**: Comprehensive error handling on both ends  
✅ **CORS Configuration**: Proper cross-origin setup  
✅ **Environment Configuration**: Flexible environment-based setup  
✅ **Type Safety**: Full TypeScript frontend + Pydantic backend validation  

## Next Steps

- Database integration (PostgreSQL/MongoDB)
- User authentication (JWT)
- Real-time notifications (WebSocket)
- File upload functionality
- Advanced filtering and search
- Unit and integration tests

## Migration from Node.js to FastAPI

The backend has been migrated from Node.js/Express to Python FastAPI while maintaining the exact same API interface. Benefits include:

- **Better Performance**: Async support and efficient request handling
- **Auto Documentation**: Built-in Swagger/OpenAPI docs at `/docs`
- **Type Safety**: Pydantic models for request/response validation
- **Modern Python**: Latest async/await patterns
- **Same Interface**: No frontend changes required

### Legacy Node.js Backend

The original Node.js backend files are still present but have been replaced by the FastAPI implementation:
- `server.js` → `main.py`
- `routes/userRoutes.js` → endpoints in `main.py`
- `data/userData.js` → `data/user_data.py`
- `middleware/` → FastAPI middleware

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure backend CORS_ORIGIN matches frontend URL
2. **API Connection**: Verify VITE_API_BASE_URL in frontend .env.local
3. **Port Conflicts**: Change PORT in backend .env if 3001 is occupied
4. **Python Environment**: Ensure virtual environment is activated

### Logs

- Backend logs: Check terminal running FastAPI server
- Frontend logs: Check browser console and terminal running frontend dev server
- API docs: Visit `http://localhost:3001/docs` for interactive testing
