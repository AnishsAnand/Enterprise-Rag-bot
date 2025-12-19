# 👥 User Switching & Security Guide

## Overview

This guide addresses user switching, credential storage security, and logout functionality.

## 🔐 Where Credentials Are Stored

### Database Location
- **Table**: `users` (PostgreSQL/SQLite)
- **Columns**: 
  - `api_auth_email` (String, nullable)
  - `api_auth_password` (String, nullable - **ENCRYPTED**)

### Security Measures

1. **Password Encryption**: 
   - Passwords are encrypted using Fernet (symmetric encryption) before storage
   - Encryption key is stored in environment variable `CREDENTIALS_ENCRYPTION_KEY`
   - If key is not set, a key is generated (warning logged)

2. **Database Security**:
   - Credentials are stored in the same database as user accounts
   - Access is controlled through authentication middleware
   - Only the user themselves (or admins) can update their credentials

### Potential Issues & Solutions

#### Issue 1: Plain Text Storage (FIXED ✅)
**Problem**: Passwords stored in plain text  
**Solution**: Implemented encryption using Fernet symmetric encryption

#### Issue 2: Database Access
**Problem**: If database is compromised, credentials could be exposed  
**Solution**: 
- Use strong database passwords
- Enable SSL/TLS for database connections
- Regular backups with encryption
- Consider using a secrets management system (Vault, AWS Secrets Manager)

#### Issue 3: Encryption Key Management
**Problem**: Encryption key needs to be secure  
**Solution**:
```bash
# Generate a secure key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Add to .env
CREDENTIALS_ENCRYPTION_KEY=your-generated-key-here
```

## 🔄 User Switching

### For OpenWebUI Users

OpenWebUI has its own user management system. To switch users:

1. **Sign Out**:
   - Click on your profile (bottom left)
   - Click "Sign Out"
   - This clears the OpenWebUI session

2. **Sign In as Different User**:
   - Enter different username/password
   - OpenWebUI will create a new session
   - Your backend will receive the new `user_id` in requests

3. **Set API Credentials for New User**:
   ```bash
   # After logging in as new user, set their credentials
   curl -X PUT http://localhost:8000/api/user/credentials \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer NEW_USER_TOKEN" \
     -d '{
       "api_auth_email": "new-user@example.com",
       "api_auth_password": "new-password"
     }'
   ```

### For Direct API Users

If using the backend API directly (not through OpenWebUI):

1. **Login as User 1**:
   ```bash
   curl -X POST http://localhost:8000/api/auth/login \
     -d "username=user1&password=pass1"
   # Returns: {"access_token": "token1", "token_type": "bearer"}
   ```

2. **Use Token for User 1**:
   ```bash
   curl -X GET http://localhost:8000/api/user/credentials \
     -H "Authorization: Bearer token1"
   ```

3. **Logout** (client-side):
   - Remove token from storage
   - Call logout endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/auth/logout \
     -H "Authorization: Bearer token1"
   ```

4. **Login as User 2**:
   ```bash
   curl -X POST http://localhost:8000/api/auth/login \
     -d "username=user2&password=pass2"
   # Returns: {"access_token": "token2", "token_type": "bearer"}
   ```

## 🚪 Logout Issues & Fixes

### Problem: Sign Out Doesn't Work Properly

**Root Cause**: The authentication system was using `fake_users_db` (in-memory) instead of the database, causing session persistence issues.

**Fixed**:
1. ✅ Updated `auth.py` to use database User model
2. ✅ Added proper logout endpoint
3. ✅ JWT tokens are stateless, so logout is primarily client-side

### How Logout Works Now

1. **Client-Side** (Frontend):
   ```javascript
   // Remove token from localStorage
   localStorage.removeItem('token');
   // Redirect to login
   window.location.href = '/login';
   ```

2. **Server-Side** (Optional):
   ```bash
   # Call logout endpoint
   curl -X POST http://localhost:8000/api/auth/logout \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

### For OpenWebUI

OpenWebUI handles logout internally:
- Click "Sign Out" in the user menu
- OpenWebUI clears its session
- Next request to backend will require new authentication

## 🔧 Admin Access Issue

### Problem: Admin Access Persists After Logout

**Root Cause**: 
- JWT tokens are stateless and valid until expiration
- Frontend wasn't properly clearing tokens
- Database wasn't being used for user lookup

**Solution**:
1. ✅ Fixed auth to use database
2. ✅ Added logout endpoint
3. ✅ Frontend should clear tokens on logout

### Testing Logout

1. **Login as admin**:
   ```bash
   curl -X POST http://localhost:8000/api/auth/login \
     -d "username=admin&password=admin123"
   ```

2. **Verify admin access**:
   ```bash
   curl -X GET http://localhost:8000/api/auth/admin-only \
     -H "Authorization: Bearer ADMIN_TOKEN"
   ```

3. **Logout**:
   ```bash
   curl -X POST http://localhost:8000/api/auth/logout \
     -H "Authorization: Bearer ADMIN_TOKEN"
   ```

4. **Try to use old token** (should fail):
   ```bash
   curl -X GET http://localhost:8000/api/auth/admin-only \
     -H "Authorization: Bearer ADMIN_TOKEN"
   # Should return 401 Unauthorized
   ```

## 📋 User Management Workflow

### Creating Multiple Users

1. **Register User 1**:
   ```bash
   curl -X POST http://localhost:8000/api/auth/register \
     -H "Content-Type: application/json" \
     -d '{"username": "user1", "password": "pass1"}'
   ```

2. **Register User 2**:
   ```bash
   curl -X POST http://localhost:8000/api/auth/register \
     -H "Content-Type: application/json" \
     -d '{"username": "user2", "password": "pass2"}'
   ```

3. **Set Credentials for Each User**:
   ```bash
   # User 1
   curl -X PUT http://localhost:8000/api/user/credentials \
     -H "Authorization: Bearer USER1_TOKEN" \
     -d '{"api_auth_email": "user1@example.com", "api_auth_password": "pass1"}'
   
   # User 2
   curl -X PUT http://localhost:8000/api/user/credentials \
     -H "Authorization: Bearer USER2_TOKEN" \
     -d '{"api_auth_email": "user2@example.com", "api_auth_password": "pass2"}'
   ```

4. **Switch Between Users**:
   - Logout from current user
   - Login as different user
   - Each user's API calls will use their own credentials

## 🔒 Security Best Practices

1. **Encryption Key**:
   ```bash
   # Generate and store securely
   export CREDENTIALS_ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
   ```

2. **Database Security**:
   - Use strong passwords
   - Enable SSL/TLS
   - Restrict network access
   - Regular backups

3. **Token Management**:
   - Tokens expire after 60 minutes (configurable)
   - Clear tokens on logout
   - Use HTTPS in production

4. **Password Storage**:
   - Never log passwords
   - Encrypt before storage
   - Use secure key management

## 🐛 Troubleshooting

### Issue: "User not found" after logout/login
**Solution**: Ensure database is initialized and user exists in database

### Issue: Credentials not working after user switch
**Solution**: Each user must set their own credentials via `/api/user/credentials`

### Issue: Admin access persists
**Solution**: 
1. Clear browser localStorage
2. Clear cookies
3. Restart browser
4. Login again

### Issue: Encryption errors
**Solution**: 
1. Set `CREDENTIALS_ENCRYPTION_KEY` in `.env`
2. Ensure key is consistent across restarts
3. If key changes, existing encrypted passwords will need to be re-entered


