import os
import logging
from docling.datamodel.pipeline_options import PdfPipelineOptions, smolvlm_picture_description

logger = logging.getLogger(__name__)

def get_vision_enabled() -> bool:
    """Check if picture description is enabled via environment variable"""
    enabled = os.getenv("ENABLE_PICTURE_DESCRIPTION", "false").lower()
    return enabled == "true"

def get_vision_pipeline_options() -> PdfPipelineOptions:
    """
    Configure PdfPipelineOptions with Granite Vision for image description.

    Returns:
        PdfPipelineOptions configured for vision processing
    """
    pipeline_options = PdfPipelineOptions()

    # Enable picture description if configured
    vision_enabled = get_vision_enabled()
    pipeline_options.do_picture_description = vision_enabled

    if vision_enabled:
        logger.info("[VISION] Picture description enabled with SmolVLM-256M (optimized for 6GB GPU)")

        # Use SmolVLM preset (HuggingFaceTB/SmolVLM-256M-Instruct)
        pipeline_options.picture_description_options = smolvlm_picture_description

        # Customize prompt if provided
        custom_prompt = os.getenv("PICTURE_DESCRIPTION_PROMPT")
        if custom_prompt:
            pipeline_options.picture_description_options.prompt = custom_prompt
            logger.info(f"[VISION] Using custom prompt: {custom_prompt}")
        else:
            logger.info(f"[VISION] Using default prompt: {smolvlm_picture_description.prompt}")

        # Configure area threshold (minimum image size to describe)
        area_threshold = float(os.getenv("PICTURE_AREA_THRESHOLD", "0.05"))
        pipeline_options.picture_description_options.picture_area_threshold = area_threshold
        logger.info(f"[VISION] Picture area threshold: {area_threshold}")

        # Configure batch size
        batch_size = int(os.getenv("PICTURE_BATCH_SIZE", "4"))
        pipeline_options.picture_description_options.batch_size = batch_size
        logger.info(f"[VISION] Batch size: {batch_size}")

        # Enable image generation and scaling for better quality
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0
        logger.info("[VISION] Image generation enabled with 2x scaling")
    else:
        logger.info("[VISION] Picture description disabled")

    return pipeline_options
