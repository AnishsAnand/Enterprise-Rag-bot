# 🔑 Dynamic API Credentials from OpenWebUI

## Overview

The system now supports per-user API credentials stored in the database, retrieved from OpenWebUI users instead of hardcoded environment variables. This allows each user to have their own Tata Communications API credentials.

## ✅ What Was Changed

### 1. **User Model Extended** (`app/models/user.py`)
- Added `api_auth_email` and `api_auth_password` fields to store user-specific API credentials

### 2. **User Credentials Service** (`app/services/user_credentials_service.py`)
- New service to retrieve and update user API credentials from the database
- Falls back to environment variables if user credentials are not found

### 3. **API Executor Service Updated** (`app/services/api_executor_service.py`)
- Modified to accept credentials dynamically per user
- Token caching is now per-user (different users can have different tokens)
- Automatically retrieves credentials from database when `user_id` is provided
- Falls back to environment variables for backward compatibility

### 4. **Execution Agent Updated** (`app/agents/execution_agent.py`)
- Now passes `user_id` from context to `execute_operation`
- Enables automatic credential retrieval for each user

### 5. **New API Endpoints** (`app/api/routes/user_credentials.py`)
- `PUT /api/user/credentials` - Update API credentials for current user
- `GET /api/user/credentials` - Check if user has credentials configured

## 🚀 How to Use

### Step 1: Set Up User Credentials

Users can set their API credentials via the API endpoint:

```bash
# Update credentials for the current user
curl -X PUT http://localhost:8000/api/user/credentials \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "api_auth_email": "your-email@tatacommunications.onmicrosoft.com",
    "api_auth_password": "your-password"
  }'
```

### Step 2: Check Credentials Status

```bash
# Check if user has credentials configured
curl -X GET http://localhost:8000/api/user/credentials \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Step 3: Use the System

When a user makes API calls through OpenWebUI:
1. The system automatically retrieves their credentials from the database
2. Uses those credentials to authenticate with the Tata Communications API
3. Caches the auth token per user for efficiency
4. Falls back to environment variables if user credentials are not found

## 🔄 Backward Compatibility

The system maintains backward compatibility:
- If `user_id` is not provided, uses environment variables (`API_AUTH_EMAIL`, `API_AUTH_PASSWORD`)
- If user credentials are not found in database, falls back to environment variables
- Existing code continues to work without changes

## 📊 Response Format

### Successful Login Response
```json
{
  "statusCode": 200,
  "accessToken": "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICIzZzBSOUo3X0VWVWtsdEY4V2FUZ3kyMXZLZ1pHckg2QWJ0c3ZfbjVfcVpjIn0..."
}
```

### Failed Login Response
```json
{
  "statusCode": 500,
  "accessToken": "Failed to generate token after retries"
}
```

## 🔐 Security Notes

1. **Password Storage**: Passwords are stored in plain text in the database. Consider encrypting them in production.
2. **Token Caching**: Tokens are cached per user for 8 minutes to reduce API calls
3. **Fallback**: Environment variables are still used as fallback for backward compatibility

## 🛠️ Database Migration

The User model has been extended. Run database migrations to add the new columns:

```python
# The columns will be added automatically when the app starts
# Or manually via Alembic if you're using migrations
```

## 📝 Example Flow

1. User logs into OpenWebUI with username "john_doe"
2. User sets their API credentials via `/api/user/credentials`
3. User asks: "List my Kubernetes clusters"
4. System:
   - Retrieves credentials for "john_doe" from database
   - Uses those credentials to get auth token
   - Makes API call with user's token
   - Returns clusters specific to that user's engagement

## 🎯 Benefits

1. **Multi-User Support**: Each user can have their own API credentials
2. **No Hardcoding**: No need to hardcode credentials in environment variables
3. **User-Specific Access**: Users only see resources from their own engagements
4. **Flexible**: Easy to update credentials per user without restarting the service


