# Tests Directory

This directory contains all test files for the Enterprise RAG Bot project.

## 📋 Test Files

### Python Tests

#### Agent System Tests
- **`test_agent_system.py`** - Tests for the multi-agent routing system
  - Agent selection logic
  - Intent detection
  - Agent coordination

#### Authentication Tests
- **`test_token_auth.py`** - Token-based authentication tests
  - Token validation
  - Session management
  - Security checks

#### Cluster Operations Tests
- **`test_cluster_list.py`** - Cluster listing and management tests
  - Cluster creation
  - Cluster querying
  - Cluster operations

#### API Endpoint Tests
- **`test_endpoint_list.py`** - API endpoint functionality tests
  - Endpoint routing
  - Response validation
  - Error handling

#### Persistence Tests
- **`test_postgres_persistence.py`** - PostgreSQL persistence layer tests
  - Database operations
  - Data integrity
  - Connection management

- **`test_memori_persistence.py`** - Memory-based persistence tests
  - In-memory storage
  - Session persistence
  - Data retention

- **`manual_test_persistence.py`** - Manual testing script for persistence
  - Interactive testing
  - Debug utilities

### Shell Script Tests

- **`test_backend.sh`** - Backend server testing script
  - Server startup validation
  - API health checks
  - Integration tests

- **`test_backend_standalone.sh`** - Standalone backend tests
  - Isolated component testing
  - Service validation

- **`test_openai_endpoints.sh`** - OpenAI API endpoint tests
  - LLM integration tests
  - Embedding service tests
  - API compatibility checks

### Test Data

- **`test_sessions.db`** - Test database for session management
  - Used by persistence tests
  - Contains test session data

## 🚀 Running Tests

### Python Tests

```bash
# Run all Python tests
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot
python -m pytest tests/

# Run specific test file
python tests/test_agent_system.py

# Run with verbose output
python -m pytest tests/ -v
```

### Shell Script Tests

```bash
# Make scripts executable
chmod +x tests/*.sh

# Run backend tests
./tests/test_backend.sh

# Run OpenAI endpoint tests
./tests/test_openai_endpoints.sh
```

## 📝 Test Coverage

### Current Coverage Areas

- ✅ Agent system routing and selection
- ✅ Authentication and token validation
- ✅ Cluster operations
- ✅ API endpoints
- ✅ Database persistence (PostgreSQL)
- ✅ Memory persistence
- ✅ Backend server functionality
- ✅ OpenAI API integration

### Areas for Additional Testing

- 🔄 Frontend component tests
- 🔄 End-to-end integration tests
- 🔄 Load and performance tests
- 🔄 Security and penetration tests

## 🧪 Test Guidelines

### Writing New Tests

1. **Naming Convention**: Use `test_*.py` or `test_*.sh` format
2. **Organization**: Group related tests in the same file
3. **Documentation**: Add docstrings explaining test purpose
4. **Assertions**: Use clear, descriptive assertion messages
5. **Cleanup**: Ensure tests clean up after themselves

### Test Structure

```python
def test_feature_name():
    """
    Test description: What this test validates
    """
    # Setup
    # ... preparation code ...
    
    # Execute
    # ... test execution ...
    
    # Assert
    # ... validation ...
    
    # Cleanup (if needed)
    # ... cleanup code ...
```

## 🐛 Debugging Tests

### Common Issues

1. **Database Connection Errors**
   - Check if PostgreSQL is running
   - Verify connection credentials
   - Ensure test database exists

2. **API Endpoint Failures**
   - Verify backend server is running
   - Check port availability
   - Review API logs

3. **Authentication Failures**
   - Validate test tokens
   - Check token expiration
   - Verify API keys

### Debug Mode

```bash
# Run with debug logging
LOGLEVEL=DEBUG python tests/test_agent_system.py

# Run with pytest debug output
python -m pytest tests/ -vv --log-cli-level=DEBUG
```

## 📊 Test Results

Test results and logs are typically stored in:
- Console output
- `test_sessions.db` (for persistence tests)
- Application logs in `outputs/` directory

## 🔗 Related Documentation

- [Testing Status](../metadata/TESTING_STATUS.md) - Current testing status
- [Implementation Status](../metadata/IMPLEMENTATION_STATUS.md) - Feature implementation status
- [Architecture](../metadata/ARCHITECTURE.md) - System architecture for context

---

*For questions about tests or to report test failures, please refer to the main project documentation in the `metadata/` folder.*

