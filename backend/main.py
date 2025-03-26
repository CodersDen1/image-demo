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
import os
from fastapi.middleware.cors import CORSMiddleware

# Get allowed origins from environment variable or use default for development


# Load environment variables from .env
load_dotenv()


# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Cloudinary config
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# OpenAI config
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Create FastAPI app
app = FastAPI(
    title="Hybrid Image Enhancer",
    description="Enhances images using brightness thresholds, optional OpenAI detection, gamma/CLAHE, smoothing, and Cloudinary upload.",
    version="3.0.0"
)

# CORS (customize in production)
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

def hybrid_enhancement(
    image: np.ndarray, 
    brightness: float,
    low_brightness_threshold: float = 100.0,
    high_brightness_threshold: float = 140.0,
) -> np.ndarray:
    """
    Apply different gamma and CLAHE settings based on brightness:
      - Very dark (< low_brightness_threshold): strong gamma & strong CLAHE
      - Moderate brightness: mild gamma & mild CLAHE
      - Bright (> high_brightness_threshold): skip gamma/CLAHE
    Then optionally apply bilateral smoothing and a final brightness/contrast tweak.
    """

    # Load from environment or use defaults
    apply_smoothing = os.environ.get("APPLY_SMOOTHING", "true").lower() == "true"
    bilateral_d = int(os.environ.get("BILATERAL_D", 9))
    bilateral_sigma_color = float(os.environ.get("BILATERAL_SIGMA_COLOR", 75))
    bilateral_sigma_space = float(os.environ.get("BILATERAL_SIGMA_SPACE", 75))

    # Decide gamma & CLAHE clip limit
    if brightness < low_brightness_threshold:
        # Heavily enhance
        gamma_value = 2.0
        clahe_clip_limit = 2.0
    elif brightness < high_brightness_threshold:
        # Mild enhancement
        gamma_value = 1.2
        clahe_clip_limit = 1.0
    else:
        # Already bright, skip gamma/CLAHE
        gamma_value = None
        clahe_clip_limit = None

    # --- Step 1: Gamma Correction ---
    if gamma_value:
        inv_gamma = 1.0 / gamma_value
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")
        image = cv2.LUT(image, table)
        logger.debug(f"Applied gamma correction: {gamma_value}")

    # --- Step 2: CLAHE (if clip limit is set) ---
    if clahe_clip_limit:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, 
            tileGridSize=(8, 8)
        )
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        logger.debug(f"Applied CLAHE with clipLimit={clahe_clip_limit}")

    # --- Step 3: Optional Bilateral Smoothing ---
    if apply_smoothing:
        image = cv2.bilateralFilter(
            image,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space
        )
        logger.debug("Applied bilateral filter for smoothing.")

    # --- Step 4: Final Brightness/Contrast Tweak ---
    # Subtle linear shift to ensure a clean, bright result
    alpha = float(os.environ.get("FINAL_ALPHA", 1.05))  # contrast
    beta = float(os.environ.get("FINAL_BETA", 10.0))    # brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    logger.debug(f"Applied final brightness/contrast tweak: alpha={alpha}, beta={beta}")

    return image

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    1. Read the image from the uploaded file.
    2. Calculate brightness.
    3. (Optional) Use OpenAI to confirm if it's dark enough for enhancement.
    4. Apply a 'hybrid' enhancement approach based on brightness thresholds.
    5. Upload the final image to Cloudinary.
    6. Return the Cloudinary URL.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data.")

        # 1) Calculate brightness
        brightness = calculate_brightness(img)
        logger.info(f"Brightness: {brightness:.2f}")

        # 2) Check if we want to use OpenAI
        use_openai_detection = os.environ.get("USE_OPENAI_DETECTION", "false").lower() == "true"
        needs_enhancement = False
        if use_openai_detection:
            # Ask OpenAI if the image is dark
            needs_enhancement = await detect_darkness_with_openai(brightness)
        else:
            # Simple threshold fallback
            needs_enhancement = (brightness < 140.0)  # You can tweak this

        # 3) Apply hybrid enhancement if needed, else skip
        if needs_enhancement:
            logger.info("Enhancement required. Applying hybrid approach.")
            processed_img = hybrid_enhancement(
                image=img,
                brightness=brightness,
                low_brightness_threshold=float(os.environ.get("LOW_BRIGHTNESS_THRESHOLD", 100.0)),
                high_brightness_threshold=float(os.environ.get("HIGH_BRIGHTNESS_THRESHOLD", 140.0))
            )
        else:
            logger.info("No enhancement required. Returning original image.")
            processed_img = img

        # 4) Encode to JPEG
        success, buffer = cv2.imencode(".jpg", processed_img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode processed image.")
        image_bytes = buffer.tobytes()

        # 5) Upload to Cloudinary
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
