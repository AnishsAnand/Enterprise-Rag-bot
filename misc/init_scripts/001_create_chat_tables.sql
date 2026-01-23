-- Migration: Create Chat Tables for OpenWebUI-compatible API
-- Date: 2024
-- Description: Creates tables for chat persistence, tags, and folders

-- ===================== Chats Table =====================
-- Main chat storage - compatible with OpenWebUI schema
CREATE TABLE IF NOT EXISTS chats (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    title TEXT NOT NULL DEFAULT 'New Chat',
    -- JSON field containing messages and conversation history
    -- Structure: { "history": { "messages": {...}, "currentId": "..." }, "title": "...", "models": [...] }
    chat JSONB NOT NULL DEFAULT '{}',
    -- Timestamps as epoch integers (OpenWebUI style)
    created_at BIGINT NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT,
    updated_at BIGINT NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT,
    -- Sharing
    share_id TEXT UNIQUE,
    -- Organization
    archived BOOLEAN NOT NULL DEFAULT FALSE,
    pinned BOOLEAN DEFAULT FALSE,
    -- Metadata (tags, etc.)
    meta JSONB NOT NULL DEFAULT '{}',
    folder_id TEXT
);

-- Indexes for chats
CREATE INDEX IF NOT EXISTS idx_chat_user_id ON chats(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_folder_id ON chats(folder_id);
CREATE INDEX IF NOT EXISTS idx_chat_user_pinned ON chats(user_id, pinned);
CREATE INDEX IF NOT EXISTS idx_chat_user_archived ON chats(user_id, archived);
CREATE INDEX IF NOT EXISTS idx_chat_updated_user ON chats(updated_at, user_id);

-- ===================== Chat Files Table =====================
-- Tracks files attached to chat messages
CREATE TABLE IF NOT EXISTS chat_files (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    chat_id VARCHAR(36) NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    message_id VARCHAR(255),
    file_id VARCHAR(255) NOT NULL,
    created_at BIGINT NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT,
    updated_at BIGINT NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT
);

CREATE INDEX IF NOT EXISTS idx_chat_file_chat_id ON chat_files(chat_id);

-- ===================== Tags Table =====================
-- User tags for organizing chats
CREATE TABLE IF NOT EXISTS tags (
    id VARCHAR(255) PRIMARY KEY,  -- tag_name normalized (lowercase, underscores)
    name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    created_at BIGINT NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT
);

CREATE INDEX IF NOT EXISTS idx_tag_user_id ON tags(user_id);

-- ===================== Folders Table =====================
-- Folders for organizing chats
CREATE TABLE IF NOT EXISTS folders (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    parent_id VARCHAR(36),
    created_at BIGINT NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT,
    updated_at BIGINT NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT
);

CREATE INDEX IF NOT EXISTS idx_folder_user_id ON folders(user_id);
CREATE INDEX IF NOT EXISTS idx_folder_parent_id ON folders(parent_id);

-- ===================== Sample Data (Optional) =====================
-- Uncomment to insert a sample chat for testing

-- INSERT INTO chats (id, user_id, title, chat, created_at, updated_at, meta)
-- VALUES (
--     'sample-chat-001',
--     'default_user',
--     'Welcome Chat',
--     '{
--         "title": "Welcome Chat",
--         "history": {
--             "messages": {
--                 "msg-001": {
--                     "id": "msg-001",
--                     "role": "user",
--                     "content": "Hello!",
--                     "timestamp": 1704067200
--                 },
--                 "msg-002": {
--                     "id": "msg-002",
--                     "role": "assistant",
--                     "content": "Hello! How can I help you today?",
--                     "timestamp": 1704067201
--                 }
--             },
--             "currentId": "msg-002"
--         },
--         "models": ["default"]
--     }',
--     EXTRACT(EPOCH FROM NOW())::BIGINT,
--     EXTRACT(EPOCH FROM NOW())::BIGINT,
--     '{"tags": []}'
-- );

-- ===================== Grant Permissions (if needed) =====================
-- GRANT ALL PRIVILEGES ON TABLE chats TO your_user;
-- GRANT ALL PRIVILEGES ON TABLE chat_files TO your_user;
-- GRANT ALL PRIVILEGES ON TABLE tags TO your_user;
-- GRANT ALL PRIVILEGES ON TABLE folders TO your_user;
