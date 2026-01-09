-- File: init_scripts/02-add-missing-columns.sql
-- EMERGENCY FIX: Add missing columns to existing users table
-- Run this ONLY if you already have a users table with missing columns
-- This is idempotent - safe to run multiple times

-- ===================== Add Missing Columns =====================

-- Add updated_at (THE CRITICAL ONE!)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE users ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP;
        RAISE NOTICE '✅ Added updated_at column';
    ELSE
        RAISE NOTICE '⏭️  updated_at already exists';
    END IF;
END $$;

-- Add full_name
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'full_name'
    ) THEN
        ALTER TABLE users ADD COLUMN full_name VARCHAR(255);
        RAISE NOTICE '✅ Added full_name column';
    ELSE
        RAISE NOTICE '⏭️  full_name already exists';
    END IF;
END $$;

-- Add avatar_url
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'avatar_url'
    ) THEN
        ALTER TABLE users ADD COLUMN avatar_url VARCHAR(500);
        RAISE NOTICE '✅ Added avatar_url column';
    ELSE
        RAISE NOTICE '⏭️  avatar_url already exists';
    END IF;
END $$;

-- Add bio
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'bio'
    ) THEN
        ALTER TABLE users ADD COLUMN bio TEXT;
        RAISE NOTICE '✅ Added bio column';
    ELSE
        RAISE NOTICE '⏭️  bio already exists';
    END IF;
END $$;

-- Add theme
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'theme'
    ) THEN
        ALTER TABLE users ADD COLUMN theme VARCHAR(50) DEFAULT 'light';
        RAISE NOTICE '✅ Added theme column';
    ELSE
        RAISE NOTICE '⏭️  theme already exists';
    END IF;
END $$;

-- Add language
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'language'
    ) THEN
        ALTER TABLE users ADD COLUMN language VARCHAR(10) DEFAULT 'en';
        RAISE NOTICE '✅ Added language column';
    ELSE
        RAISE NOTICE '⏭️  language already exists';
    END IF;
END $$;

-- Add timezone
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'timezone'
    ) THEN
        ALTER TABLE users ADD COLUMN timezone VARCHAR(50) DEFAULT 'UTC';
        RAISE NOTICE '✅ Added timezone column';
    ELSE
        RAISE NOTICE '⏭️  timezone already exists';
    END IF;
END $$;

-- Add notifications_enabled
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'notifications_enabled'
    ) THEN
        ALTER TABLE users ADD COLUMN notifications_enabled BOOLEAN NOT NULL DEFAULT TRUE;
        RAISE NOTICE '✅ Added notifications_enabled column';
    ELSE
        RAISE NOTICE '⏭️  notifications_enabled already exists';
    END IF;
END $$;

-- Add email_notifications
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'email_notifications'
    ) THEN
        ALTER TABLE users ADD COLUMN email_notifications BOOLEAN NOT NULL DEFAULT FALSE;
        RAISE NOTICE '✅ Added email_notifications column';
    ELSE
        RAISE NOTICE '⏭️  email_notifications already exists';
    END IF;
END $$;

-- Add failed_login_attempts
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'failed_login_attempts'
    ) THEN
        ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER NOT NULL DEFAULT 0;
        RAISE NOTICE '✅ Added failed_login_attempts column';
    ELSE
        RAISE NOTICE '⏭️  failed_login_attempts already exists';
    END IF;
END $$;

-- Add locked_until
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'locked_until'
    ) THEN
        ALTER TABLE users ADD COLUMN locked_until TIMESTAMP WITH TIME ZONE;
        RAISE NOTICE '✅ Added locked_until column';
    ELSE
        RAISE NOTICE '⏭️  locked_until already exists';
    END IF;
END $$;

-- ===================== Fix Column Types =====================

-- Ensure columns have correct types
DO $$
BEGIN
    -- Fix username type
    BEGIN
        ALTER TABLE users ALTER COLUMN username TYPE VARCHAR(255);
        RAISE NOTICE '✅ Updated username type';
    EXCEPTION WHEN others THEN
        RAISE NOTICE '⏭️  username type already correct';
    END;
    
    -- Fix email type
    BEGIN
        ALTER TABLE users ALTER COLUMN email TYPE VARCHAR(255);
        RAISE NOTICE '✅ Updated email type';
    EXCEPTION WHEN others THEN
        RAISE NOTICE '⏭️  email type already correct';
    END;
    
    -- Fix hashed_password type
    BEGIN
        ALTER TABLE users ALTER COLUMN hashed_password TYPE VARCHAR(255);
        RAISE NOTICE '✅ Updated hashed_password type';
    EXCEPTION WHEN others THEN
        RAISE NOTICE '⏭️  hashed_password type already correct';
    END;
    
    -- Fix created_at to include timezone
    BEGIN
        ALTER TABLE users ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE;
        RAISE NOTICE '✅ Updated created_at to include timezone';
    EXCEPTION WHEN others THEN
        RAISE NOTICE '⏭️  created_at type already correct';
    END;
    
    -- Fix last_login to include timezone
    BEGIN
        ALTER TABLE users ALTER COLUMN last_login TYPE TIMESTAMP WITH TIME ZONE;
        RAISE NOTICE '✅ Updated last_login to include timezone';
    EXCEPTION WHEN others THEN
        RAISE NOTICE '⏭️  last_login type already correct';
    END;
END $$;

-- ===================== Create/Update Trigger =====================

-- Create trigger function if doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop and recreate trigger
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

RAISE NOTICE '✅ Created/updated trigger for updated_at';

-- ===================== Create Missing Indexes =====================

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

RAISE NOTICE '✅ Created/verified indexes';

-- ===================== Verification =====================

DO $$
DECLARE
    col_count INTEGER;
    required_cols TEXT[] := ARRAY[
        'id', 'username', 'email', 'hashed_password', 'full_name', 'avatar_url',
        'bio', 'role', 'is_active', 'is_verified', 'theme', 'language', 'timezone',
        'notifications_enabled', 'email_notifications', 'last_login', 'login_count',
        'failed_login_attempts', 'locked_until', 'created_at', 'updated_at'
    ];
    missing_cols TEXT[];
    col TEXT;
BEGIN
    -- Check for missing columns
    SELECT ARRAY(
        SELECT unnest(required_cols)
        EXCEPT
        SELECT column_name::TEXT
        FROM information_schema.columns
        WHERE table_name = 'users'
    ) INTO missing_cols;
    
    -- Count total columns
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_name = 'users';
    
    RAISE NOTICE '';
    RAISE NOTICE '================================';
    RAISE NOTICE '✅ Migration Complete';
    RAISE NOTICE '================================';
    RAISE NOTICE 'Total columns: %', col_count;
    RAISE NOTICE 'Expected: 21 columns';
    
    IF array_length(missing_cols, 1) > 0 THEN
        RAISE NOTICE '';
        RAISE WARNING '⚠️  Still missing columns:';
        FOREACH col IN ARRAY missing_cols
        LOOP
            RAISE NOTICE '   - %', col;
        END LOOP;
    ELSE
        RAISE NOTICE '✅ All 21 columns present!';
    END IF;
    RAISE NOTICE '================================';
    RAISE NOTICE '';
END $$;