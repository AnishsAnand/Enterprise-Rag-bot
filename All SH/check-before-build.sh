#!/bin/bash
# Pre-build validation script
# Save as: check-before-build.sh
# Run: bash check-before-build.sh

echo "=============================================="
echo "🔍 Pre-Build Validation Check"
echo "=============================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Check 1: Verify we're in project root
echo "📁 Checking project structure..."
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}❌ ERROR: Dockerfile not found in current directory${NC}"
    echo "   Please run this script from the project root"
    exit 1
fi
echo -e "${GREEN}✅ Dockerfile found${NC}"

# Check 2: Find main.py (admin) and app/user_main.py
echo ""
echo "🔍 Looking for main entry points..."

ADMIN_MAIN_FOUND=false
USER_MAIN_FOUND=false

# Check for main.py (admin backend) in root
if [ -f "app/main.py" ]; then
    echo -e "${GREEN}✅ main.py found in project root (admin backend)${NC}"
    ADMIN_MAIN_FOUND=true
else
    echo -e "${RED}❌ ERROR: main.py not found in project root${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check for user_main.py in app/
if [ -f "app/user_main.py" ]; then
    echo -e "${GREEN}✅ app/user_main.py found (user backend)${NC}"
    USER_MAIN_FOUND=true
else
    echo -e "${RED}❌ ERROR: app/user_main.py not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check 3: Verify config files
echo ""
echo "⚙️  Checking configuration files..."

CONFIG_FILES=(
    "misc/config/supervisord.conf"
    "misc/config/default.conf"
    "misc/scripts/entrypoint.sh"
    "docker-compose.openwebui.yml"
)

for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ MISSING: $file${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check 4: Verify entrypoint.sh is executable
if [ -f "misc/scripts/entrypoint.sh" ]; then
    if [ -x "misc/scripts/entrypoint.sh" ]; then
        echo -e "${GREEN}✅ entrypoint.sh is executable${NC}"
    else
        echo -e "${YELLOW}⚠️  entrypoint.sh is not executable${NC}"
        echo "   Run: chmod +x misc/scripts/entrypoint.sh"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

# Check 5: Verify .env file exists
echo ""
echo "🔐 Checking environment configuration..."

if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env file found in project root${NC}"
    
    # Check for required variables
    REQUIRED_VARS=(
        "OPENAI_API_KEY"
        "API_AUTH_TOKEN"
        "JWT_SECRET_KEY"
        "WEBUI_SECRET_KEY"
    )
    
    for var in "${REQUIRED_VARS[@]}"; do
        if grep -q "^${var}=" .env && ! grep -q "^${var}=REPLACE" .env && ! grep -q "^${var}=$" .env; then
            echo -e "${GREEN}  ✅ ${var} is set${NC}"
        else
            echo -e "${RED}  ❌ ${var} is NOT set or needs to be replaced${NC}"
            ERRORS=$((ERRORS + 1))
        fi
    done
else
    echo -e "${RED}❌ ERROR: .env file not found in project root${NC}"
    echo "   Copy env.openwebui.template to .env and fill in your API keys"
    ERRORS=$((ERRORS + 1))
fi

# Check 6: Verify frontend directories
echo ""
echo "🎨 Checking frontend directories..."

if [ -d "angular-frontend" ]; then
    echo -e "${GREEN}✅ angular-frontend directory found${NC}"
else
    echo -e "${RED}❌ ERROR: angular-frontend directory not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "user-frontend" ]; then
    echo -e "${GREEN}✅ user-frontend directory found${NC}"
else
    echo -e "${RED}❌ ERROR: user-frontend directory not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check 7: Verify app directory
echo ""
echo "🐍 Checking Python application..."

if [ -d "app" ]; then
    echo -e "${GREEN}✅ app directory found${NC}"
else
    echo -e "${RED}❌ ERROR: app directory not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}✅ requirements.txt found${NC}"
else
    echo -e "${RED}❌ ERROR: requirements.txt not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "=============================================="
echo "📊 Validation Summary"
echo "=============================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Ready to build.${NC}"
    echo ""
    echo "Next step:"
    echo "  docker-compose -f docker-compose.openwebui.yml build --no-cache"
    echo ""
    echo "Or use the startup script:"
    echo "  bash scripts/start_with_openwebui.sh"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  ${WARNINGS} warning(s) found, but build should work${NC}"
    echo ""
    echo "You can proceed with:"
    echo "  docker-compose -f docker-compose.openwebui.yml build --no-cache"
    echo ""
    echo "Or use the startup script:"
    echo "  bash scripts/start_with_openwebui.sh"
    exit 0
else
    echo -e "${RED}❌ ${ERRORS} error(s) found. Please fix before building.${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠️  ${WARNINGS} warning(s) also found${NC}"
    fi
    echo ""
    echo "Fix the errors above before running build."
    exit 1
fi