#!/bin/bash
# scripts/generate_ssl_certs.sh - Generate self-signed SSL certificates for development

set -e

echo "🔐 Generating self-signed SSL certificates..."

# Create SSL directory if it doesn't exist
mkdir -p nginx/ssl

# Generate self-signed certificate valid for 365 days
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/key.pem \
    -out nginx/ssl/cert.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Set proper permissions
chmod 644 nginx/ssl/cert.pem
chmod 600 nginx/ssl/key.pem

echo "✅ SSL certificates generated successfully!"
echo "📁 Files created:"
echo "   - nginx/ssl/cert.pem (certificate)"
echo "   - nginx/ssl/key.pem (private key)"
echo ""
echo "⚠️  These are self-signed certificates for development only."
echo "    For production, use Let's Encrypt or a trusted CA."
echo ""
echo "To enable HTTPS:"
echo "1. Uncomment the HTTPS server block in nginx/nginx.conf"
echo "2. Restart nginx: docker-compose restart nginx"