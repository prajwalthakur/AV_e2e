#!/bin/bash
docker build --rm  $@ -t e2e:latest -f "$(dirname "$0")/../../.docker/e2e.Dockerfile" "$(dirname "$0")/../.."