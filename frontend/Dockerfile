# Build stage
FROM node:20-slim AS build-stage
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# Same-origin API (nginx proxies /api to the backend)
RUN VITE_API_URL="" npm run build

# Production stage
FROM nginx:stable-alpine AS production-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
