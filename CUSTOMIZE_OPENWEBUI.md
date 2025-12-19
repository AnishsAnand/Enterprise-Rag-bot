# Customizing OpenWebUI - Branding & Signup

## 1. Customize the Landing Page & Branding

### Option A: Via Admin Panel (After Login)
1. Login as admin
2. Click profile icon → "Admin Settings"
3. Go to "Interface" or "Settings"
4. Customize:
   - **App Name**: Change from "Open WebUI" to "Vayu Maya"
   - **Logo**: Upload your Tata Communications logo
   - **Welcome Message**: Change "Discover wonders" to your message
   - **Theme**: Change colors, dark/light mode

### Option B: Via Environment Variables (Before Starting)
```bash
docker run -d \
  --name enterprise-rag-openwebui \
  -p 3000:8080 \
  -e WEBUI_NAME="Vayu Maya - AI Cloud Assistant" \
  -e WEBUI_LOGO_URL="https://your-domain.com/logo.png" \
  -e DEFAULT_LOCALE="en-US" \
  -e WEBUI_BANNER_TEXT="Welcome to Vayu Maya - Your AI Cloud Assistant" \
  ...other settings...
```

### Custom Branding Examples:
```bash
# Tata Communications Branding
-e WEBUI_NAME="Vayu Maya"
-e WEBUI_BANNER_TEXT="Tata Communications AI Cloud Assistant"
-e WEBUI_LOGO_URL="https://www.tatacommunications.com/logo.png"

# Custom Welcome Message
-e WEBUI_LANDING_PAGE_MESSAGE="Welcome to Vayu Maya - Discover cloud solutions powered by AI"
```

## 2. Enable Signup After Admin Creation

### The Problem:
After the first user (admin) signs up, OpenWebUI **disables signup by default** for security.

### The Solution:
Admin must manually enable signup for others.

### Steps to Enable Signup:

#### Step 1: Login as Admin
- Go to `http://localhost:3000/`
- Login with your admin credentials

#### Step 2: Enable Signup
1. Click your **profile icon** (top right)
2. Click **"Admin Settings"** or **"Settings"**
3. Find **"General"** or **"Authentication"** section
4. Look for **"Enable Signup"** toggle
5. **Turn it ON**
6. Save settings

#### Step 3: Test
- Logout
- You should now see a **"Sign Up"** or **"Create Account"** link on the login page

### Alternative: Enable Signup via API

If you can't find the setting in the UI, enable it via API:

```bash
# First, login and get your admin token
TOKEN=$(curl -s -X POST http://localhost:3000/api/v1/auths/signin \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@tatacommunications.com","password":"YourPassword"}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['token'])")

# Then enable signup
curl -X POST http://localhost:3000/api/v1/configs/update \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"ENABLE_SIGNUP": true}'
```

## 3. Customize Signup Page

### Add Custom Fields or Messages

You can customize what users see during signup:

```bash
# Require email confirmation
-e ENABLE_SIGNUP_PASSWORD_CONFIRMATION=true

# Custom signup message
-e WEBUI_SIGNUP_MESSAGE="Sign up with your Tata Communications email for full access"

# Require admin approval for new users
-e ENABLE_ADMIN_APPROVAL=true
```

### Restrict Signup by Email Domain

Only allow certain email domains to signup:

```bash
# Only allow @tatacommunications.com emails
-e ALLOWED_EMAIL_DOMAINS="tatacommunications.com"

# Allow multiple domains
-e ALLOWED_EMAIL_DOMAINS="tatacommunications.com,tata.com"
```

## 4. Complete Customization Example

Here's a fully customized OpenWebUI setup:

```bash
docker stop enterprise-rag-openwebui
docker rm enterprise-rag-openwebui

docker run -d \
  --name enterprise-rag-openwebui \
  -p 3000:8080 \
  \
  # Branding
  -e WEBUI_NAME="Vayu Maya" \
  -e WEBUI_BANNER_TEXT="Tata Communications AI Cloud Assistant" \
  -e WEBUI_LOGO_URL="https://www.tatacommunications.com/logo.png" \
  \
  # Authentication
  -e WEBUI_AUTH=true \
  -e ENABLE_SIGNUP=true \
  -e ENABLE_SIGNUP_PASSWORD_CONFIRMATION=true \
  -e ALLOWED_EMAIL_DOMAINS="tatacommunications.com" \
  \
  # Backend Connection
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8001/api/v1" \
  -e OPENAI_API_KEY="secure-openwebui-api-key-2024" \
  \
  # Features
  -e ENABLE_RAG=true \
  -e DEFAULT_MODELS="Vayu Maya" \
  -e DEFAULT_USER_ROLE=user \
  \
  --add-host=host.docker.internal:host-gateway \
  -v openwebui-data:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

## 5. Signup Flow After Customization

### For Admin:
1. First user to register → Becomes admin automatically
2. Admin enables signup in settings
3. Admin can customize branding

### For New Users:
1. Visit `http://localhost:3000/`
2. See custom landing page with your branding
3. Click "Sign Up" (now visible after admin enabled it)
4. Register with email (restricted to @tatacommunications.com if configured)
5. Get appropriate access based on email domain

## 6. Current Issue: Signup Not Visible After Admin Creation

This is **normal behavior** - OpenWebUI disables signup after first user for security.

### Quick Fix:

**Option A**: Enable via Admin Panel (Recommended)
1. Login as admin
2. Admin Settings → Enable Signup toggle → ON

**Option B**: Restart with ENABLE_SIGNUP=true
```bash
# This forces signup to be enabled on startup
docker restart enterprise-rag-openwebui
```

**Option C**: Use the API command above to enable it

## Summary

| Customization | How to Do It |
|---------------|--------------|
| **App Name** | `-e WEBUI_NAME="Vayu Maya"` |
| **Logo** | `-e WEBUI_LOGO_URL="url"` |
| **Landing Message** | `-e WEBUI_BANNER_TEXT="message"` |
| **Enable Signup** | Admin Settings → Enable Signup |
| **Restrict Email** | `-e ALLOWED_EMAIL_DOMAINS="domain.com"` |
| **Theme/Colors** | Admin Panel → Interface Settings |

## Next Steps

1. **Create admin account** (if not done)
2. **Login as admin**
3. **Go to Admin Settings**
4. **Enable "Signup"**
5. **Customize branding** (name, logo, colors)
6. **Test**: Logout and verify signup link appears

Would you like me to:
1. Show you exactly where to find the signup toggle in admin settings?
2. Restart OpenWebUI with custom branding now?
3. Enable signup via API for you?

