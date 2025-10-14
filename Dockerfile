# Stage 1: Build Angular frontends
FROM node:18 AS admin-frontend-build
WORKDIR /admin-frontend
COPY angular-frontend/ ./
RUN npm install && npm run build --prod

FROM node:18 AS user-frontend-build
WORKDIR /user-frontend
COPY user-frontend/ ./
RUN npm install && npm run build --prod

# Stage 2: Python backend
FROM python:3.11-slim AS backend
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app
COPY app ./app

# Install nginx
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

# Copy built frontends
COPY --from=admin-frontend-build /admin-frontend/dist/enterprise-rag-frontend /var/www/admin
COPY --from=user-frontend-build /user-frontend/dist/user-rag-frontend /var/www/user

# Nginx config
COPY default.conf /etc/nginx/conf.d/default.conf

# Expose backend and frontend ports
EXPOSE 8000 80

# Start both backend and nginx
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & nginx -g "daemon off;"
