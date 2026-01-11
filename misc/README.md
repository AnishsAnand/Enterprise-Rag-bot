# Miscellaneous Files Directory

This directory contains Docker Compose configurations and utility scripts that support the Enterprise RAG Bot project.

## 📂 Directory Structure

```
misc/
├── docker/          # Docker Compose configurations
├── scripts/         # Utility scripts
└── README.md        # This file
```

> **Note**: The main Dockerfile and service configs are in the project root:
> - `Dockerfile` - Main container build
> - `docker/` - Nginx and supervisord configs
> - `docker-compose.yml` - Primary orchestration

## 🐳 Docker (`docker/`)

Docker Compose configurations for different deployment scenarios.

### Files

- **`docker-compose.yml`** - Main Docker Compose configuration
  - Defines all services (backend, frontend, Milvus, PostgreSQL, etc.)
  - Network configuration
  - Volume mappings
  - Environment variables

- **`docker-compose.openwebui.yml`** - OpenWebUI-specific Docker Compose
  - OpenWebUI service configuration
  - Integration with main services
  - Additional dependencies (Redis)

### Usage

```bash
# Start all services (from project root)
docker-compose -f misc/docker/docker-compose.yml up -d

# Start with OpenWebUI
docker-compose -f misc/docker/docker-compose.openwebui.yml up -d

# Stop services
docker-compose -f misc/docker/docker-compose.yml down

# View logs
docker-compose -f misc/docker/docker-compose.yml logs -f
```

## 🔧 Scripts (`scripts/`)

Utility scripts for various operations.

### Files

- **`start_with_openwebui.sh`** - OpenWebUI startup script
  - Starts backend services
  - Initializes OpenWebUI
  - Sets up environment
  - Health checks

- **`createcluster.ts`** - TypeScript cluster creation utility
  - Cluster creation automation
  - Configuration validation
  - API interaction
  - Error handling

### Usage

```bash
# Make scripts executable
chmod +x misc/scripts/*.sh

# Start with OpenWebUI
./misc/scripts/start_with_openwebui.sh

# Run cluster creation (requires Node.js/Deno)
ts-node misc/scripts/createcluster.ts
# or
deno run --allow-net misc/scripts/createcluster.ts
```

## 🔗 Integration with Main Project

These files support the main application located in:
- **Backend**: `app/`
- **Frontend**: `user-frontend/`, `angular-frontend/`
- **Configs**: `docker/` (root level)
- **Documentation**: `metadata/`
- **Tests**: `tests/`

## 📋 Configuration Reference

All active configuration files are in the **root `docker/`** folder:

| File | Purpose |
|------|---------|
| `docker/supervisord.conf` | Process manager for running nginx + backends |
| `docker/admin_default.conf` | Nginx config for admin frontend (port 4200) |
| `docker/user_default.conf` | Nginx config for user frontend (port 4201) |
| `docker/supervisord-user.conf` | User-only backend config (alternative) |
| `docker/env.openwebui.template` | Environment template for OpenWebUI integration |

## 🚀 Quick Start

### Using Docker

```bash
# 1. Navigate to project root
cd /path/to/Enterprise-Rag-bot

# 2. Start services
docker-compose -f misc/docker/docker-compose.yml up -d

# 3. Check status
docker-compose -f misc/docker/docker-compose.yml ps
```

### Using Scripts

```bash
# 1. Make scripts executable
chmod +x misc/scripts/*.sh

# 2. Run startup script
./misc/scripts/start_with_openwebui.sh

# 3. Monitor logs
tail -f outputs/*.log
```

## 🐛 Troubleshooting

### Docker Issues

```bash
# Rebuild containers
docker-compose -f misc/docker/docker-compose.yml build --no-cache

# Remove volumes and restart
docker-compose -f misc/docker/docker-compose.yml down -v
docker-compose -f misc/docker/docker-compose.yml up -d

# Check logs
docker-compose -f misc/docker/docker-compose.yml logs [service-name]
```

### Script Issues

- Check file permissions (`chmod +x`)
- Verify dependencies installed
- Review script output/errors
- Check environment variables

## 🔗 Related Documentation

- [Quick Start Guide](../metadata/QUICK_START.md)
- [Deployment Success](../metadata/DEPLOYMENT_SUCCESS.md)
- [Architecture](../metadata/ARCHITECTURE.md)
- [OpenWebUI Integration](../metadata/OPENWEBUI_README.md)

## 📊 File Overview

| Category | Files | Purpose |
|----------|-------|---------|
| Docker Compose | 2 files | Container orchestration variants |
| Scripts | 2 files | Automation and utility operations |

---

*These miscellaneous files are essential for deployment and operation of the Enterprise RAG Bot.*
