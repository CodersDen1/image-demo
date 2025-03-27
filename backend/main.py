import os
import io
import json
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

# Load environment variables
load_dotenv()

# Configure production-level logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# Configure OpenAI (optional)
openai.api_key = os.environ.get("OPENAI_API_KEY")

app = FastAPI(
    title="Bright & Warm Image Enhancer",
    description="A production-grade FastAPI app that ensures final images are bright and warm, ideal for real estate listings.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Customize for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

################################################
#             ENHANCEMENT PIPELINE             #
################################################

def calculate_brightness(image: np.ndarray) -> float:
    """Compute average brightness (0-255) by converting to grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    logger.info(f"Calculated brightness: {brightness:.2f}")
    return brightness

def apply_adaptive_gamma(image: np.ndarray, brightness: float) -> np.ndarray:
    """
    Apply stronger gamma for very dark images, mild for moderately dark,
    skip for bright images.
    """
    if brightness < 60:
        gamma_value = 1.8
    elif brightness < 90:
        gamma_value = 1.4
    elif brightness < 110:
        gamma_value = 1.2
    else:
        return image  # skip gamma for bright images

    inv_gamma = 1.0 / gamma_value
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    image = cv2.LUT(image, table)
    logger.debug(f"Applied gamma correction: {gamma_value}")
    return image

def apply_clahe_if_dark(image: np.ndarray, brightness: float) -> np.ndarray:
    """
    CLAHE for local contrast if brightness < ~120.
    """
    if brightness >= 120:
        return image
    clip_limit = float(os.environ.get("CLAHE_CLIP_LIMIT", 1.5))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    logger.debug(f"Applied CLAHE with clipLimit={clip_limit}")
    return image

def apply_subtle_color_adjustments(image: np.ndarray) -> np.ndarray:
    """
    Slight warmth + mild saturation to keep a bright, inviting look.
    """
    warm_factor = float(os.environ.get("WARM_FACTOR", 1.05))
    sat_factor = float(os.environ.get("SAT_FACTOR", 1.05))

    b, g, r = cv2.split(image)
    r = np.clip(r * warm_factor, 0, 255).astype(np.uint8)
    b = np.clip(b * 0.98, 0, 255).astype(np.uint8)  # small reduction in blue
    warmed = cv2.merge((b, g, r))

    # mild saturation
    hsv = cv2.cvtColor(warmed, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    s = np.clip(s * sat_factor, 0, 255)
    hsv_boosted = cv2.merge([h, s, v]).astype("uint8")
    image = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
    logger.debug(f"Applied warm_factor={warm_factor}, sat_factor={sat_factor}")
    return image

def apply_unsharp_mask(image: np.ndarray) -> np.ndarray:
    """
    Mild unsharp mask for clarity, avoiding an over-processed look.
    """
    unsharp_strength = float(os.environ.get("UNSHARP_STRENGTH", 1.2))
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = cv2.addWeighted(image, unsharp_strength, gaussian_blur, -(unsharp_strength - 1.0), 0)
    logger.debug(f"Applied unsharp mask with strength={unsharp_strength}")
    return sharpened

def final_tweak(image: np.ndarray) -> np.ndarray:
    """
    Minor linear brightness/contrast tweak to unify the result.
    """
    alpha = float(os.environ.get("FINAL_ALPHA", 1.02))
    beta = float(os.environ.get("FINAL_BETA", 5.0))
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    logger.debug(f"Applied final tweak: alpha={alpha}, beta={beta}")
    return image

def production_enhancement_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Complete pipeline:
      1) Adaptive Gamma
      2) CLAHE if dark
      3) Subtle color adjustments
      4) Unsharp mask
      5) Final tweak
    """
    brightness = calculate_brightness(image)
    image = apply_adaptive_gamma(image, brightness)
    # re-check brightness after gamma
    brightness = calculate_brightness(image)

    image = apply_clahe_if_dark(image, brightness)
    image = apply_subtle_color_adjustments(image)
    image = apply_unsharp_mask(image)
    image = final_tweak(image)

    return image

def final_brightness_warmth_check(image: np.ndarray) -> np.ndarray:
    """
    After the main pipeline, if the result is still below a brightness threshold,
    apply an extra brightness + warmth pass to ensure the photo doesn't look dark.
    """
    threshold = float(os.environ.get("FINAL_BRIGHTNESS_THRESHOLD", 130.0))
    brightness = calculate_brightness(image)
    if brightness < threshold:
        logger.info(f"Final brightness={brightness:.2f} < threshold={threshold}; applying extra warmth & lift.")
        # Slight linear brightness lift
        image = cv2.convertScaleAbs(image, alpha=1.03, beta=10)  # small extra bump

        # Additional mild warmth
        warm_factor = 1.05
        b, g, r = cv2.split(image)
        r = np.clip(r * warm_factor, 0, 255).astype(np.uint8)
        image = cv2.merge((b, g, r))
    else:
        logger.info(f"Final brightness={brightness:.2f} >= threshold; no extra warmth needed.")

    return image

################################################
#         OPTIONAL OPENAI DETECTION            #
################################################

async def decide_if_enhancement_needed(brightness: float) -> bool:
    """
    If USE_OPENAI_DETECTION=true, ask GPT-3.5 whether the image needs enhancement.
    Otherwise, fallback to brightness threshold.
    """
    use_openai = os.environ.get("USE_OPENAI_DETECTION", "false").lower() == "true"
    if not use_openai:
        return brightness < 140

    system_msg = "You evaluate if an image is too dark and needs enhancement."
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
    1) Reads the uploaded image.
    2) Decides if enhancement is needed (optional OpenAI).
    3) Applies pipeline (gamma, CLAHE, color adjustments, unsharp mask, final tweak).
    4) Checks final brightness, adds extra warmth if still under threshold.
    5) Encodes at high JPEG quality & uploads to Cloudinary.
    6) Returns the Cloudinary URL of the final image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # Decode the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data.")

        # Decide if enhancement is needed
        brightness = calculate_brightness(image)
        enhance_needed = await decide_if_enhancement_needed(brightness)
        if enhance_needed:
            logger.info("Enhancement needed; applying pipeline.")
            image = production_enhancement_pipeline(image)
            # final check
            image = final_brightness_warmth_check(image)
        else:
            logger.info("Enhancement not needed; returning original image.")

        # Encode final image at high quality
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        success, buffer = cv2.imencode(".jpg", image, encode_params)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode final image.")

        # Upload to Cloudinary
        folder = os.environ.get("CLOUDINARY_FOLDER", "enhanced_images")
        result = cloudinary.uploader.upload(
            io.BytesIO(buffer.tobytes()),
            folder=folder,
            resource_type="image"
        )
        url = result.get("secure_url")
        if not url:
            raise HTTPException(status_code=500, detail="Failed to upload image to Cloudinary.")

        return {"url": url}

    except Exception as e:
        logger.exception("Error processing the image:")
        raise HTTPException(status_code=500, detail=str(e))
