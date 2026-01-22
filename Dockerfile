
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git nginx supervisor build-essential netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY app ./app
COPY docker /docker

FROM node:20 AS admin-frontend-build
WORKDIR /angular-frontend
COPY angular-frontend/package*.json ./
RUN npm install --force
COPY angular-frontend/ .
RUN npm run build --prod

FROM node:20 AS user-frontend-build
WORKDIR /user-frontend
COPY user-frontend/package*.json ./
RUN npm install --force
COPY user-frontend/ .
RUN npm run build --prod

FROM base AS final

COPY --from=admin-frontend-build /angular-frontend/dist/enterprise-rag-frontend /var/www/admin
COPY --from=user-frontend-build /user-frontend/dist/user-frontend /var/www/user

COPY docker/supervisord.conf /etc/supervisord.conf
COPY docker/admin_default.conf /etc/nginx/conf.d/admin_default.conf
COPY docker/user_default.conf /etc/nginx/conf.d/user_default.conf

RUN mkdir -p /var/log /var/run/supervisor /app/logs /app/uploads /app/outputs /app/backups \
    && chmod -R 755 /var/log /var/run/supervisor /app/logs /app/uploads /app/outputs /app/backups


RUN chown -R root:root /var/www /app

RUN which supervisord && which nginx && which uvicorn

EXPOSE 4200 4201 8000

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
