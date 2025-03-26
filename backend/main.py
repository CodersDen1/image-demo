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

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# OpenAI configuration
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Create FastAPI app
app = FastAPI(
    title="Hybrid Image Enhancer with Warm Filter",
    description="Enhances images using brightness thresholds, optional OpenAI detection, gamma/CLAHE, smoothing, warm filter, and Cloudinary upload.",
    version="4.0.0"
)

# Enable CORS (customize allowed origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_brightness(image: np.ndarray) -> float:
    """
    Calculate average brightness by converting to grayscale and taking the mean intensity.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

async def detect_darkness_with_openai(brightness: float) -> bool:
    """
    Use OpenAI to decide if the image is dark enough to require enhancement.
    Returns True if enhancement is recommended, False otherwise.
    """
    prompt = (
        f"An image has an average brightness of {brightness:.2f} on a scale of 0 to 255. "
        "Should the image be enhanced to improve its lighting? Answer with only 'yes' or 'no'."
    )
    try:
        response = await asyncio.to_thread(
            openai.Completion.create,
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
        )
        answer = response.choices[0].text.strip().lower()
        logger.info(f"OpenAI detection answer: {answer}")
        return (answer == "yes")
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        # Fallback to threshold-based logic if OpenAI fails
        return brightness < 100

def apply_warm_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply a warm filter to the image by enhancing red tones and slightly reducing blue.
    The warm factor can be controlled via the environment variable WARM_FACTOR.
    """
    warm_factor = float(os.environ.get("WARM_FACTOR", 1.1))  # Default warm factor 1.1
    b, g, r = cv2.split(image)
    # Increase red channel intensity and slightly reduce blue
    r = np.clip(r * warm_factor, 0, 255).astype(np.uint8)
    b = np.clip(b * 0.9, 0, 255).astype(np.uint8)
    warm_image = cv2.merge((b, g, r))
    logger.debug(f"Applied warm filter with factor {warm_factor}")
    return warm_image

def hybrid_enhancement(
    image: np.ndarray, 
    brightness: float,
    low_brightness_threshold: float = 100.0,
    high_brightness_threshold: float = 140.0,
) -> np.ndarray:
    """
    Apply different gamma and CLAHE settings based on brightness:
      - Very dark (< low_brightness_threshold): strong gamma & strong CLAHE.
      - Moderate brightness: mild gamma & mild CLAHE.
      - Bright (> high_brightness_threshold): skip gamma/CLAHE.
    Then optionally apply bilateral smoothing, a final brightness/contrast tweak,
    and a warm filter to add warmth to dark images.
    """

    # Load smoothing parameters from environment variables
    apply_smoothing = os.environ.get("APPLY_SMOOTHING", "true").lower() == "true"
    bilateral_d = int(os.environ.get("BILATERAL_D", 9))
    bilateral_sigma_color = float(os.environ.get("BILATERAL_SIGMA_COLOR", 75))
    bilateral_sigma_space = float(os.environ.get("BILATERAL_SIGMA_SPACE", 75))

    # Decide gamma & CLAHE settings based on brightness thresholds
    if brightness < low_brightness_threshold:
        gamma_value = 2.0
        clahe_clip_limit = 2.0
    elif brightness < high_brightness_threshold:
        gamma_value = 1.2
        clahe_clip_limit = 1.0
    else:
        gamma_value = None  # Skip gamma
        clahe_clip_limit = None  # Skip CLAHE

    # --- Step 1: Gamma Correction ---
    if gamma_value:
        inv_gamma = 1.0 / gamma_value
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        image = cv2.LUT(image, table)
        logger.debug(f"Applied gamma correction: {gamma_value}")

    # --- Step 2: CLAHE (if applicable) ---
    if clahe_clip_limit:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        logger.debug(f"Applied CLAHE with clipLimit={clahe_clip_limit}")

    # --- Step 3: Optional Bilateral Smoothing ---
    if apply_smoothing:
        image = cv2.bilateralFilter(image, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)
        logger.debug("Applied bilateral filter for smoothing.")

    # --- Step 4: Final Brightness/Contrast Tweak ---
    alpha = float(os.environ.get("FINAL_ALPHA", 1.05))  # Contrast control
    beta = float(os.environ.get("FINAL_BETA", 10.0))     # Brightness control
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    logger.debug(f"Applied final brightness/contrast tweak: alpha={alpha}, beta={beta}")

    # --- Step 5: Apply Warm Filter ---
    apply_warm = os.environ.get("APPLY_WARM_FILTER", "true").lower() == "true"
    if apply_warm:
        image = apply_warm_filter(image)

    return image

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    1. Reads an uploaded image.
    2. Calculates brightness.
    3. Optionally uses OpenAI to determine if enhancement is needed.
    4. Applies a hybrid enhancement (with gamma/CLAHE/smoothing and warm filter) if necessary.
    5. Uploads the final image to Cloudinary.
    6. Returns the Cloudinary URL.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # Read and decode the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data.")

        # Calculate brightness
        brightness = calculate_brightness(img)
        logger.info(f"Calculated brightness: {brightness:.2f}")

        # Use OpenAI detection if enabled, otherwise fallback to threshold logic
        use_openai_detection = os.environ.get("USE_OPENAI_DETECTION", "false").lower() == "true"
        if use_openai_detection:
            needs_enhancement = await detect_darkness_with_openai(brightness)
        else:
            needs_enhancement = brightness < 140.0

        # Apply hybrid enhancement if needed
        if needs_enhancement:
            logger.info("Enhancement required. Applying hybrid enhancement.")
            processed_img = hybrid_enhancement(
                image=img,
                brightness=brightness,
                low_brightness_threshold=float(os.environ.get("LOW_BRIGHTNESS_THRESHOLD", 100.0)),
                high_brightness_threshold=float(os.environ.get("HIGH_BRIGHTNESS_THRESHOLD", 140.0))
            )
        else:
            logger.info("Enhancement not required. Returning original image.")
            processed_img = img

        # Encode processed image to JPEG
        success, buffer = cv2.imencode(".jpg", processed_img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode processed image.")
        image_bytes = buffer.tobytes()

        # Upload to Cloudinary
        folder = os.environ.get("CLOUDINARY_FOLDER", "enhanced_images")
        upload_result = cloudinary.uploader.upload(
            io.BytesIO(image_bytes),
            folder=folder,
            resource_type="image"
        )
        secure_url = upload_result.get("secure_url")
        if not secure_url:
            raise HTTPException(status_code=500, detail="Failed to upload image to Cloudinary.")

        return {"url": secure_url}

    except Exception as e:
        logger.exception("Error processing image:")
        raise HTTPException(status_code=500, detail=str(e))
