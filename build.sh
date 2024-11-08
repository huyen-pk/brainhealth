docker-compose -f infrastructure/docker/alzheimer_detection_training.yml \
    build \
    # -e ALZHEIMER_MODELS_REPO_DIR_PATH=$ALZHEIMER_MODELS_REPO_DIR_PATH \
    #             ALZHEIMER_BASE_MODEL_PATH=$ALZHEIMER_BASE_MODEL_PATH \
    #             ALZHEIMER_MODELS_REPO_DIR_PATH=$ALZHEIMER_MODELS_REPO_DIR_PATH