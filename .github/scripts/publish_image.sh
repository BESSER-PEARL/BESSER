#!/usr/bin/env bash
# Build and push one image with buildx, then check the registry serves it.
#
# Usage: publish_image.sh <image:tag> <docker buildx build args...>
#
# buildx pushes an OCI index (the provenance attestation makes it one), the
# format every other push to artefacts.list.lu has. Artifactory keeps a tag's
# index and plain manifest apart and serves the index, so a plain `docker push`
# left :latest on the previous image while reporting success.
set -euo pipefail

image="$1"
shift
meta="$(mktemp)"
trap 'rm -f "$meta"' EXIT

docker buildx build --push --provenance=mode=min --metadata-file "$meta" -t "$image" "$@"

pushed="$(jq -r '."containerimage.digest"' "$meta")"
served="$(docker buildx imagetools inspect "$image" --format '{{json .Manifest}}' | jq -r .digest)"
echo "$image: pushed $pushed, registry serves $served"
if [[ "$pushed" != "$served" ]]; then
  echo "::error::The registry serves $served for $image, not $pushed just pushed. Nothing was restarted."
  exit 1
fi
