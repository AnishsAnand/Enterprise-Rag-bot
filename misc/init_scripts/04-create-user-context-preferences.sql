-- Migration: Create user_context_preferences table
-- Purpose: Persistent storage for user's default entity selections
-- (engagement, datacenter, cluster, firewall, BU, environment, zone)

-- Create the table if it doesn't exist
CREATE TABLE IF NOT EXISTS user_context_preferences (
    -- Primary key - user identifier (email)
    user_id VARCHAR(255) PRIMARY KEY,
    
    -- Engagement context
    default_engagement_id INTEGER,
    default_engagement_name VARCHAR(255),
    default_ipc_engagement_id INTEGER,
    
    -- Datacenter/Endpoint context
    default_datacenter_id INTEGER,
    default_datacenter_name VARCHAR(255),
    default_endpoint_ids JSONB,  -- List of endpoint IDs
    
    -- Cluster context
    default_cluster_id INTEGER,
    default_cluster_name VARCHAR(255),
    
    -- Firewall context
    default_firewall_id INTEGER,
    default_firewall_name VARCHAR(255),
    
    -- Business Unit context
    default_business_unit_id INTEGER,
    default_business_unit_name VARCHAR(255),
    
    -- Environment context
    default_environment_id INTEGER,
    default_environment_name VARCHAR(255),
    
    -- Zone context
    default_zone_id INTEGER,
    default_zone_name VARCHAR(255),
    
    -- Flexible storage for additional preferences
    preferences JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_user_context_engagement 
ON user_context_preferences(default_engagement_id);

CREATE INDEX IF NOT EXISTS idx_user_context_datacenter 
ON user_context_preferences(default_datacenter_id);

CREATE INDEX IF NOT EXISTS idx_user_context_user_id 
ON user_context_preferences(user_id);

-- Grant permissions (adjust user as needed)
GRANT ALL PRIVILEGES ON TABLE user_context_preferences TO ragbot;

-- Verify creation
SELECT 'user_context_preferences table created successfully!' as status;
