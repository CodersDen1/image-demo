import os
import io
import logging
import asyncio
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import openai

# Load environment variables from .env
load_dotenv()

# Configure production-level logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Configure Cloudinary with explicit transformation settings to minimize compression
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# Configure OpenAI (optional)
openai.api_key = os.environ.get("OPENAI_API_KEY")

app = FastAPI(
    title="Production-Grade Image Enhancer",
    description=(
        "A robust image enhancer that performs adaptive gamma correction, conditional CLAHE, "
        "subtle color adjustments, unsharp masking, and final brightness/contrast tweaks to produce "
        "bright, warm images while preserving quality. The final image is encoded at maximum quality "
        "and uploaded to Cloudinary."
    ),
    version="1.0.0"
)

# Allow CORS for all domains (customize in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

################################################
#             ENHANCEMENT PIPELINE             #
################################################

def calculate_brightness(image: np.ndarray) -> float:
    """Compute average brightness (0-255) from the grayscale version of the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    logger.info(f"Calculated brightness: {brightness:.2f}")
    return brightness

def apply_adaptive_gamma(image: np.ndarray, brightness: float) -> np.ndarray:
    """
    Applies gamma correction using high-precision float32 math.
    For very dark images, uses stronger gamma; for moderately dark, milder adjustments.
    Skips gamma for bright images.
    """
    if brightness < 60:
        gamma_value = 1.8
    elif brightness < 90:
        gamma_value = 1.4
    elif brightness < 110:
        gamma_value = 1.2
    else:
        return image  # No gamma correction for bright images

    inv_gamma = 1.0 / gamma_value
    table = (np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma
    table = (table * 255.0).clip(0, 255).astype(np.uint8)
    result = cv2.LUT(image, table)
    logger.debug(f"Applied adaptive gamma with gamma value: {gamma_value}")
    return result

def apply_clahe_if_dark(image: np.ndarray, brightness: float) -> np.ndarray:
    """
    Enhances local contrast using CLAHE if the image brightness is below ~120.
    """
    if brightness >= 120:
        return image
    clip_limit = float(os.environ.get("CLAHE_CLIP_LIMIT", 1.5))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    logger.debug(f"Applied CLAHE with clipLimit: {clip_limit}")
    return result

def apply_subtle_color_adjustments(image: np.ndarray) -> np.ndarray:
    """
    Applies slight warmth and a mild saturation boost.
    Uses float32 precision to minimize quality loss.
    """
    warm_factor = float(os.environ.get("WARM_FACTOR", 1.05))
    sat_factor = float(os.environ.get("SAT_FACTOR", 1.05))
    
    # Convert to float32 for high-precision processing
    image_float = image.astype(np.float32)
    b, g, r = cv2.split(image_float)
    r = np.clip(r * warm_factor, 0, 255)
    b = np.clip(b * 0.98, 0, 255)
    warmed = cv2.merge((b, g, r)).astype(np.uint8)
    
    # Convert to HSV and apply saturation boost
    hsv = cv2.cvtColor(warmed, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * sat_factor, 0, 255)
    hsv_boosted = cv2.merge([h, s, v]).astype(np.uint8)
    result = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
    logger.debug(f"Applied color adjustments with warm_factor={warm_factor}, sat_factor={sat_factor}")
    return result

def apply_unsharp_mask(image: np.ndarray) -> np.ndarray:
    """
    Applies a mild unsharp mask to enhance clarity.
    Uses a 3x3 Gaussian blur and a slightly reduced strength to avoid artifacts.
    """
    unsharp_strength = float(os.environ.get("UNSHARP_STRENGTH", 1.15))
    gaussian_blur = cv2.GaussianBlur(image, (3, 3), 0)
    result = cv2.addWeighted(image, unsharp_strength, gaussian_blur, -(unsharp_strength - 1.0), 0)
    logger.debug(f"Applied unsharp mask with strength: {unsharp_strength}")
    return result

def final_tweak(image: np.ndarray) -> np.ndarray:
    """
    Applies a final linear brightness/contrast tweak.
    """
    alpha = float(os.environ.get("FINAL_ALPHA", 1.02))
    beta = float(os.environ.get("FINAL_BETA", 5.0))
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    logger.debug(f"Applied final tweak: alpha={alpha}, beta={beta}")
    return result

def production_enhancement_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Full enhancement pipeline:
      1) Adaptive Gamma Correction
      2) Conditional CLAHE
      3) Subtle Color Adjustments
      4) Mild Unsharp Mask
      5) Final Linear Tweak
    """
    brightness = calculate_brightness(image)
    image = apply_adaptive_gamma(image, brightness)
    brightness = calculate_brightness(image)  # Recalculate brightness after gamma correction
    image = apply_clahe_if_dark(image, brightness)
    image = apply_subtle_color_adjustments(image)
    image = apply_unsharp_mask(image)
    image = final_tweak(image)
    return image

def final_brightness_warmth_check(image: np.ndarray) -> np.ndarray:
    """
    If the final image brightness is below a set threshold, applies an extra brightness and warmth boost.
    """
    threshold = float(os.environ.get("FINAL_BRIGHTNESS_THRESHOLD", 130.0))
    brightness = calculate_brightness(image)
    if brightness < threshold:
        logger.info(f"Final brightness ({brightness:.2f}) is below threshold ({threshold}); applying extra boost.")
        image = cv2.convertScaleAbs(image, alpha=1.03, beta=10)
        warm_factor = 1.05
        b, g, r = cv2.split(image)
        r = np.clip(r * warm_factor, 0, 255).astype(np.uint8)
        image = cv2.merge((b, g, r))
    else:
        logger.info(f"Final brightness ({brightness:.2f}) meets threshold; no extra boost applied.")
    return image

################################################
#         OPTIONAL OPENAI DETECTION            #
################################################

async def decide_if_enhancement_needed(brightness: float) -> bool:
    """
    Optionally uses OpenAI's ChatCompletion to decide if enhancement is needed.
    Falls back to a brightness threshold if not enabled.
    """
    use_openai = os.environ.get("USE_OPENAI_DETECTION", "false").lower() == "true"
    if not use_openai:
        return brightness < 140

    system_msg = "Evaluate if an image is too dark and needs enhancement."
    user_msg = (
        f"The image has an average brightness of {brightness:.2f} on a 0-255 scale. "
        "Should we enhance it for better clarity? Answer 'yes' or 'no'."
    )
    try:
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=5,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip().lower()
        logger.info(f"OpenAI says: {answer}")
        return (answer == "yes")
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        return brightness < 140

################################################
#              FASTAPI ENDPOINT                #
################################################

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """
    1) Reads the uploaded image (preserving original resolution).
    2) Determines if enhancement is needed (optionally via OpenAI).
    3) Applies the production enhancement pipeline.
    4) Performs a final brightness/warmth check.
    5) Encodes the final image at maximum JPEG quality (100) and uploads to Cloudinary.
    6) Returns the Cloudinary URL of the enhanced image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # Read and decode the image from bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data.")

        # Log original dimensions to ensure resolution is preserved
        orig_h, orig_w = image.shape[:2]
        logger.info(f"Original image dimensions: {orig_w}x{orig_h}")

        # Determine if enhancement is needed
        brightness = calculate_brightness(image)
        enhance_needed = await decide_if_enhancement_needed(brightness)
        if enhance_needed:
            logger.info("Enhancement needed; applying production enhancement pipeline.")
            image = production_enhancement_pipeline(image)
            image = final_brightness_warmth_check(image)
        else:
            logger.info("Enhancement not needed; using original image.")

        # Encode final image at maximum JPEG quality (100)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        success, buffer = cv2.imencode(".jpg", image, encode_params)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode final image.")

        # Upload to Cloudinary with explicit transformation to preserve quality
        folder = os.environ.get("CLOUDINARY_FOLDER", "enhanced_images")
        upload_result = cloudinary.uploader.upload(
            io.BytesIO(buffer.tobytes()),
            folder=folder,
            resource_type="image",
            transformation=[{"quality": "100", "fetch_format": "auto"}]
        )
        url = upload_result.get("secure_url")
        if not url:
            raise HTTPException(status_code=500, detail="Failed to upload image to Cloudinary.")

        # Log final dimensions
        final_h, final_w = image.shape[:2]
        logger.info(f"Enhanced image dimensions: {final_w}x{final_h}")
        return {"url": url}

    except Exception as e:
        logger.exception("Error processing the image:")
        raise HTTPException(status_code=500, detail=str(e))
