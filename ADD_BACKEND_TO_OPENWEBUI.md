# 🔧 How To Add Your Backend to OpenWebUI (Admin Panel)

## You're Now Logged In As Admin!

**Email**: admin@localhost  
**Password**: Admin@123

## Step-by-Step Guide

### Step 1: Access Admin Settings

1. Click your **profile icon** or **gear icon** (⚙️) in the top right
2. Look for **"Admin Panel"** or **"Admin Settings"**
3. Click it

### Step 2: Find Connections/APIs Section

Look for one of these menu items:
- **Settings** → **Connections**
- **Settings** → **External APIs**  
- **Settings** → **Models**
- **Workspace** → **Connections**
- **Admin** → **Settings** → **Connections**

### Step 3: Add OpenAI-Compatible API

Once you find the Connections/APIs section:

1. Look for **"Add Connection"** or **"Add API"** or **"OpenAI API"**
2. Click it
3. Fill in:
   - **Name**: `Vayu Maya Backend` (or any name)
   - **API Base URL**: `http://host.docker.internal:8001/api/v1`
   - **API Key**: `sk-dummy-key`
   - **Type**: OpenAI Compatible
   - **Enable**: ✅ Yes/On

4. Click **"Save"** or **"Add"**

### Step 4: Test Connection

1. Look for **"Test Connection"** button
2. Click it
3. You should see: ✅ **Connected** or **2 models found**

### Step 5: Make Models Public

1. Go to **Settings** → **Models**
2. You should now see:
   - Vayu Maya
   - Vayu Maya v2
3. For each model:
   - Click on it
   - Look for **"Visibility"** or **"Access Control"**
   - Set to **"Public"** or **"All Users"**
   - Save

### Step 6: Test with Regular User

1. **Logout** from admin
2. **Login** as regular user: `cdemoipc@gmail.com`
3. Click **"Select a model"**
4. You should see both models! ✅

## If You Can't Find "Connections" Section

Try these alternatives:

### Option A: Admin Panel → Settings

1. Admin Panel
2. Settings
3. Scroll down to find **"OpenAI"** or **"External APIs"** section
4. Add your backend URL there

### Option B: Workspace Settings

1. Click **"Workspace"** in sidebar
2. Look for **"Connections"** or **"Integrations"**
3. Add OpenAI API there

### Option C: Models Section

1. Admin Panel → Settings → **Models**
2. Look for **"Add Model Source"** or **"External Models"**
3. Add your backend there

## What To Look For

You're looking for a form with fields like:
- ✅ API Base URL / Endpoint URL
- ✅ API Key
- ✅ Model Provider / Type
- ✅ Enable/Active toggle

## Common Section Names in OpenWebUI

- **Connections**
- **External APIs**
- **OpenAI API Settings**
- **Model Sources**
- **Integrations**
- **API Keys**

## If Still Can't Find It

### Take a Screenshot

1. Go to Admin Panel
2. Click through all the Settings sections
3. Take screenshots of the menu
4. Share with me and I'll tell you exactly where to click

### Or Check OpenWebUI Version

```bash
docker logs enterprise-rag-openwebui | grep "v0\."
```

Different versions have different UI layouts.

## Alternative: Add via Database (If UI doesn't work)

If you absolutely can't find the UI option, I can add it directly to the database for you. Just let me know!

## Summary

**Goal**: Register `http://host.docker.internal:8001/api/v1` as an OpenAI-compatible API source in OpenWebUI

**Where**: Admin Panel → Settings → Connections/External APIs

**What**: Add API Base URL + API Key

**Result**: Models will appear for all users

## Next Steps

1. ✅ Login as admin (admin@localhost / Admin@123)
2. 🔍 Find Connections/APIs section in Admin Panel
3. ➕ Add your backend URL
4. ✅ Test and verify models appear
5. 🌐 Make models public for all users

**Let me know what you see in the Admin Panel and I'll guide you to the exact location!**

