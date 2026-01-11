-- File: init_scripts/00-create-users-table.sql
-- Creates the complete users table with ALL required columns
-- This REPLACES your existing 00-create-users-table.sql

-- ===================== Users Table =====================

CREATE TABLE IF NOT EXISTS users (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- Authentication & Identity
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    
    -- Profile Information
    full_name VARCHAR(255),
    avatar_url VARCHAR(500),
    bio TEXT,
    
    -- User Role & Permissions
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    
    -- Account Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- User Preferences
    theme VARCHAR(50) DEFAULT 'light',
    language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    
    -- Notification Settings
    notifications_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    email_notifications BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Security & Login Tracking
    last_login TIMESTAMP WITH TIME ZONE,
    login_count INTEGER NOT NULL DEFAULT 0,
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps - CRITICAL: Both columns must exist!
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT users_role_check CHECK (role IN ('admin', 'user', 'viewer', 'editor'))
);

-- ===================== Indexes for Performance =====================

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- ===================== Trigger for updated_at =====================

-- Create trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to users table
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ===================== Verification =====================

DO $$
DECLARE
    col_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_name = 'users';
    
    RAISE NOTICE '================================';
    RAISE NOTICE '✅ Users Table Created';
    RAISE NOTICE '================================';
    RAISE NOTICE 'Total columns: %', col_count;
    RAISE NOTICE 'Expected: 21 columns';
    
    IF col_count >= 21 THEN
        RAISE NOTICE '✅ All columns present!';
    ELSE
        RAISE NOTICE '⚠️  Warning: Expected 21 columns, found %', col_count;
    END IF;
    
    RAISE NOTICE '================================';
END $$;