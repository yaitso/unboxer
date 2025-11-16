#!/bin/bash
set -e

APP_NAME="unboxer"
IMAGE_TAG="latest"
REGISTRY_IMAGE="registry.fly.io/${APP_NAME}:${IMAGE_TAG}"

echo "checking if app ${APP_NAME} exists..."
if ! fly apps list | grep -q "^${APP_NAME}"; then
    echo "creating app ${APP_NAME}..."
    fly apps create ${APP_NAME}
else
    echo "app ${APP_NAME} already exists"
fi

echo "authenticating docker with fly registry..."
fly auth docker >/dev/null 2>&1

echo "building docker image..."
docker build \
    --platform linux/amd64 \
    -t ${REGISTRY_IMAGE} \
    -f sandbox.Dockerfile \
    .

echo "pushing image to fly registry..."
docker push ${REGISTRY_IMAGE}

echo "✓ image built and pushed successfully"
echo "  ${REGISTRY_IMAGE}"
