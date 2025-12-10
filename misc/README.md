# Miscellaneous Files Directory

This directory contains various configuration files, Docker setups, and utility scripts that support the Enterprise RAG Bot project.

## 📂 Directory Structure

```
misc/
├── docker/          # Docker and container configuration
├── config/          # Configuration files
├── scripts/         # Utility scripts
└── README.md        # This file
```

## 🐳 Docker (`docker/`)

Docker-related files for containerization and deployment.

### Files

- **`docker-compose.yml`** - Main Docker Compose configuration
  - Defines all services (backend, frontend, Milvus, etc.)
  - Network configuration
  - Volume mappings
  - Environment variables

- **`docker-compose.openwebui.yml`** - OpenWebUI-specific Docker Compose
  - OpenWebUI service configuration
  - Integration with main services
  - Additional dependencies

- **`Dockerfile`** - Docker image definition
  - Base image setup
  - Dependencies installation
  - Application configuration
  - Entry point definition

### Usage

```bash
# Start all services
docker-compose -f misc/docker/docker-compose.yml up -d

# Start with OpenWebUI
docker-compose -f misc/docker/docker-compose.openwebui.yml up -d

# Stop services
docker-compose -f misc/docker/docker-compose.yml down

# View logs
docker-compose -f misc/docker/docker-compose.yml logs -f
```

## ⚙️ Configuration (`config/`)

Configuration files for various services and components.

### Files

- **`default.conf`** - Nginx/web server configuration
  - Reverse proxy settings
  - Static file serving
  - Port mappings
  - SSL/TLS configuration (if applicable)

- **`supervisord.conf`** - Supervisor process manager configuration
  - Process definitions
  - Auto-restart policies
  - Log management
  - Service dependencies

- **`env.openwebui.template`** - OpenWebUI environment template
  - Environment variable template
  - Configuration placeholders
  - Setup instructions
  - API key placeholders

### Usage

```bash
# Copy template and configure
cp misc/config/env.openwebui.template .env.openwebui
# Edit .env.openwebui with your values

# Use supervisor configuration
supervisord -c misc/config/supervisord.conf

# Use nginx configuration
nginx -c /path/to/misc/config/default.conf
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

## 📋 File Details

### Docker Files

#### docker-compose.yml
Main orchestration file defining:
- Backend API service
- Frontend service
- Milvus vector database
- MinIO object storage
- etcd for coordination
- Network configuration

#### docker-compose.openwebui.yml
Extended configuration for:
- OpenWebUI service
- Additional dependencies
- Integration settings

#### Dockerfile
Image build instructions for:
- Python environment setup
- Dependency installation
- Application code copying
- Port exposure

### Configuration Files

#### default.conf
Web server configuration for:
- Request routing
- Static file serving
- Proxy settings
- CORS configuration

#### supervisord.conf
Process management for:
- Backend service
- Frontend service
- Worker processes
- Log rotation

#### env.openwebui.template
Environment template for:
- API endpoints
- Authentication tokens
- Service URLs
- Feature flags

### Script Files

#### start_with_openwebui.sh
Startup script that:
- Checks dependencies
- Sets environment variables
- Starts services in order
- Monitors health

#### createcluster.ts
Cluster utility that:
- Validates input
- Makes API calls
- Handles errors
- Reports status

## 🚀 Quick Start

### Using Docker

```bash
# 1. Navigate to project root
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot

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

## 🔗 Integration with Main Project

These files support the main application located in:
- **Backend**: `app/`
- **Frontend**: `user-frontend/`
- **Documentation**: `metadata/`
- **Tests**: `tests/`

## 📝 Maintenance

### Adding New Files

1. Place files in appropriate subdirectory:
   - Docker files → `misc/docker/`
   - Config files → `misc/config/`
   - Scripts → `misc/scripts/`

2. Update this README with:
   - File description
   - Usage instructions
   - Dependencies

3. Update main project documentation if needed

### Updating Existing Files

1. Test changes in development environment
2. Update documentation
3. Notify team of breaking changes
4. Update version/changelog if applicable

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

### Configuration Issues

- Verify file paths in configuration
- Check environment variables
- Validate syntax (use linters)
- Review service logs

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
| Docker | 3 files | Container orchestration and deployment |
| Config | 3 files | Service configuration and environment setup |
| Scripts | 2 files | Automation and utility operations |

---

*These miscellaneous files are essential for deployment, configuration, and operation of the Enterprise RAG Bot. Keep them organized and documented.*

