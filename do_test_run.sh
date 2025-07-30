# #!/usr/bin/env bash

# # Stop at first error
# set -e

# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# DOCKER_IMAGE_TAG="example-algorithm-track-2-atypical-mitosis-classification-preliminary-eval-phase"

# DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

# INPUT_DIR="${SCRIPT_DIR}/test/input"
# OUTPUT_DIR="${SCRIPT_DIR}/test/output"

# echo "=+= (Re)build the container"
# source "${SCRIPT_DIR}/do_build.sh"

# cleanup() {
#     echo "=+= Cleaning permissions ..."
#     # Ensure permissions are set correctly on the output
#     # This allows the host user (e.g. you) to access and handle these files
#     docker run --rm \
#       --platform=linux/amd64 \
#       --quiet \
#       --volume "$OUTPUT_DIR":/output \
#       --entrypoint /bin/sh \
#       $DOCKER_IMAGE_TAG \
#       -c "chmod -R -f o+rwX /output/* || true"

#     # Ensure volume is removed
#     docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
# }

# # This allows for the Docker user to read
# chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/model"


# if [ -d "${OUTPUT_DIR}/interf0" ]; then
#   # This allows for the Docker user to write
#   chmod -f o+rwX "${OUTPUT_DIR}/interf0"

#   echo "=+= Cleaning up any earlier output"
#   # Use the container itself to circumvent ownership problems
#   docker run --rm \
#       --platform=linux/amd64 \
#       --quiet \
#       --volume "${OUTPUT_DIR}/interf0":/output \
#       --entrypoint /bin/sh \
#       $DOCKER_IMAGE_TAG \
#       -c "rm -rf /output/* || true"
# else
#   mkdir -p -m o+rwX "${OUTPUT_DIR}/interf0"
# fi


# docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

# trap cleanup EXIT

# run_docker_forward_pass() {
#     local interface_dir="$1"

#     echo "=+= Doing a forward pass on ${interface_dir}"

#     ## Note the extra arguments that are passed here:
#     # '--network none'
#     #    entails there is no internet connection
#     # 'gpus all'
#     #    enables access to any GPUs present
#     # '--volume <NAME>:/tmp'
#     #   is added because on Grand Challenge this directory cannot be used to store permanent files
#     # '-volume ../model:/opt/ml/model/":ro'
#     #   is added to provide access to the (optional) tarball-upload locally
#     docker run --rm \
#         --platform=linux/amd64 \
#         --network none \
#         --gpus all \
#         --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
#         --volume "${OUTPUT_DIR}/${interface_dir}":/output \
#         --volume "$DOCKER_NOOP_VOLUME":/tmp \
#         --volume "${SCRIPT_DIR}/model":/opt/ml/model:ro \
#         "$DOCKER_IMAGE_TAG"

#   echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
# }


# run_docker_forward_pass "interf0"



# echo "=+= Save this image for uploading via ./do_save.sh"


#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DOCKER_IMAGE_TAG="example-algorithm-track-2-atypical-mitosis-classification-preliminary-eval-phase"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

cleanup() {
  echo "=+= Cleaning permissions ..."
  docker run --rm --platform=linux/amd64 --quiet \
    --volume "${OUTPUT_DIR}":/output \
    --entrypoint /bin/sh $DOCKER_IMAGE_TAG \
    -c "chmod -R -f o+rwX /output/* || true"
  docker volume rm "${DOCKER_NOOP_VOLUME}" >/dev/null || true
}
trap cleanup EXIT

chmod -R -f o+rX "${INPUT_DIR}" "${SCRIPT_DIR}/model"

mkdir -p -m o+rwX "${OUTPUT_DIR}"
chmod -f o+rwX "${OUTPUT_DIR}"
echo "=+= Cleaning up any earlier output"
docker run --rm --platform=linux/amd64 --quiet \
  --volume "${OUTPUT_DIR}":/output \
  --entrypoint /bin/sh $DOCKER_IMAGE_TAG \
  -c "rm -rf /output/* || true"

docker volume create "${DOCKER_NOOP_VOLUME}" >/dev/null

# detect NVIDIA runtime 
GPU_FLAG=""
if docker info --format '{{.Runtimes.nvidia}}' 2>/dev/null | grep -q 'nvidia'; then
  GPU_FLAG="--gpus all"
  echo "=+= NVIDIA runtime detected – GPU access enabled"
else
  echo "=+= No NVIDIA runtime – running on CPU"
fi

run_docker_forward_pass() {
  local interface_dir="$1"
  echo "=+= Doing a forward pass on ${interface_dir}"

  # Decide whether to request GPUs
  if [ -n "$GPU_FLAG" ]; then
    GPU_OPT=(--gpus all)
  else
    GPU_OPT=()       # empty means nothing expands
  fi

  docker run --rm \
    --platform=linux/amd64 \
    --network none \
    "${GPU_OPT[@]}" \
    -v "${INPUT_DIR}":/input:ro \
    -v "${OUTPUT_DIR}/${interface_dir}":/output \
    -v "${DOCKER_NOOP_VOLUME}":/tmp \
    -v "${SCRIPT_DIR}/model":/opt/ml/model:ro \
    "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}


run_docker_forward_pass "interf0"
echo "=+= Save this image for uploading via ./do_save.sh"
