-- File: init_scripts/01-create-admin.sql
-- Creates default admin user with credentials: admin / admin123
-- This REPLACES your existing 01-create-admin.sql

-- ===================== Create Default Admin User =====================

DO $$
BEGIN
    -- Check if users table exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables 
        WHERE schemaname = 'public' AND tablename = 'users'
    ) THEN
        RAISE NOTICE '⚠️  Users table does not exist yet. Skipping admin creation.';
        RAISE NOTICE '💡 Make sure 00-create-users-table.sql runs first!';
    ELSE
        -- Insert admin user if doesn't exist
        INSERT INTO users (
            username,
            email,
            hashed_password,
            full_name,
            role,
            is_active,
            is_verified,
            theme,
            language,
            timezone,
            notifications_enabled,
            email_notifications,
            login_count,
            failed_login_attempts,
            created_at,
            updated_at
        )
        SELECT
            'admin',
            'admin@example.com',
            -- This is bcrypt hash of 'admin123'
            -- Generated with: python -c "from passlib.context import CryptContext; print(CryptContext(schemes=['bcrypt']).hash('admin123'))"
            '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqPKL4p5ay',
            'System Administrator',
            'admin',
            TRUE,
            TRUE,
            'light',
            'en',
            'UTC',
            TRUE,
            FALSE,
            0,
            0,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        WHERE NOT EXISTS (
            SELECT 1 FROM users WHERE username = 'admin'
        );
        
        -- Check if admin was created
        IF FOUND THEN
            RAISE NOTICE '================================';
            RAISE NOTICE '✅ Default Admin User Created';
            RAISE NOTICE '================================';
            RAISE NOTICE '🔑 Credentials:';
            RAISE NOTICE '   Username: admin';
            RAISE NOTICE '   Password: admin123';
            RAISE NOTICE '';
            RAISE NOTICE '⚠️  CRITICAL: Change these in production!';
            RAISE NOTICE '================================';
        ELSE
            RAISE NOTICE '⏭️  Admin user already exists. Skipping creation.';
        END IF;
    END IF;
END $$;

-- ===================== Verify Admin User =====================

DO $$
DECLARE
    admin_exists BOOLEAN;
    admin_data RECORD;
BEGIN
    -- Check if admin exists
    SELECT EXISTS(
        SELECT 1 FROM users WHERE username = 'admin'
    ) INTO admin_exists;
    
    IF admin_exists THEN
        -- Get admin details
        SELECT 
            username, 
            email, 
            role, 
            is_active, 
            is_verified,
            created_at
        INTO admin_data
        FROM users 
        WHERE username = 'admin';
        
        RAISE NOTICE '';
        RAISE NOTICE '📋 Admin User Details:';
        RAISE NOTICE '   Username: %', admin_data.username;
        RAISE NOTICE '   Email: %', admin_data.email;
        RAISE NOTICE '   Role: %', admin_data.role;
        RAISE NOTICE '   Active: %', admin_data.is_active;
        RAISE NOTICE '   Verified: %', admin_data.is_verified;
        RAISE NOTICE '   Created: %', admin_data.created_at;
        RAISE NOTICE '';
    ELSE
        RAISE WARNING '❌ Admin user was not created!';
    END IF;
END $$;