from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance


# ============================================================
# XMP NAMESPACE
# ============================================================

CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"


# ============================================================
# XMP PARSER
# ============================================================

def strip_namespace(tag: str) -> str:
    """Convert {namespace}Exposure2012 -> Exposure2012."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_number(value):
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def parse_xmp(xmp_path: str | Path):
    """
    Read Lightroom / Adobe Camera Raw XMP.

    Returns:
        settings: scalar XMP settings
        curves:   tone-curve point lists
    """

    xmp_path = Path(xmp_path)

    if not xmp_path.exists():
        raise FileNotFoundError(f"Preset not found: {xmp_path}")

    tree = ET.parse(xmp_path)
    root = tree.getroot()

    settings = {}
    curves = {}

    # --------------------------------------------------------
    # Read attributes such as:
    #
    # crs:Exposure2012="+0.50"
    # crs:Contrast2012="+10"
    # --------------------------------------------------------
    for element in root.iter():

        for key, value in element.attrib.items():

            clean_key = strip_namespace(key)

            if key.startswith("{" + CRS_NS + "}"):
                settings[clean_key] = value

        # Most Lightroom controls are attributes, but some XMP producers write
        # a scalar setting as a CRS element instead.  Preserve those too so an
        # uploaded XMP is not silently reduced to its attributes.
        clean_tag = strip_namespace(element.tag)
        if (
            element.tag.startswith("{" + CRS_NS + "}")
            and clean_tag not in settings
            and element.text
            and not list(element)
        ):
            settings[clean_tag] = element.text.strip()

    # --------------------------------------------------------
    # Read RDF sequence/list values such as ToneCurvePV2012
    # --------------------------------------------------------
    for element in root.iter():

        clean_tag = strip_namespace(element.tag)

        if clean_tag.startswith("ToneCurve"):

            points = []

            for child in element.iter():

                child_tag = strip_namespace(child.tag)

                if child_tag == "li" and child.text:

                    text = child.text.strip()

                    match = re.match(
                        r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
                        text,
                    )

                    if match:
                        x = float(match.group(1))
                        y = float(match.group(2))
                        points.append((x, y))

            if points:
                curves[clean_tag] = points

    return settings, curves


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(path):
    image = Image.open(path).convert("RGB")

    return np.asarray(image).astype(np.float32) / 255.0


def save_image(image, path):

    image = np.clip(image, 0.0, 1.0)

    output = Image.fromarray(
        (image * 255.0).round().astype(np.uint8)
    )

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.save(
        path,
        quality=95,
    )


def srgb_to_linear(x):

    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(x):

    x = np.clip(x, 0.0, None)

    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    )


def get_bool(settings, name, default=False):
    value = settings.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes"}


def setting(settings, *names, default=0.0):
    """Return the first numeric XMP setting that is present."""
    for name in names:
        if name in settings:
            return get_number(settings, name, default)
    return default


def apply_white_balance(image, settings):
    """Apply the globally stored Temperature/Tint sliders to an RGB export.

    XMP presets commonly use IncrementalTemperature/IncrementalTint.  Absolute
    temperature is camera-dependent, so this deliberately treats both forms as
    relative corrections—the only meaningful interpretation for a JPEG/TIFF
    arriving from an earlier pipeline.
    """
    temperature = setting(settings, "IncrementalTemperature", "Temperature")
    tint = setting(settings, "IncrementalTint", "Tint")
    if temperature == 0 and tint == 0:
        return image

    linear = srgb_to_linear(image)
    warm = np.clip(temperature, -100.0, 100.0) / 100.0
    magenta = np.clip(tint, -150.0, 150.0) / 150.0
    gains = np.array(
        [1.0 + warm * 0.24 + magenta * 0.10,
         1.0 - magenta * 0.15,
         1.0 - warm * 0.24 + magenta * 0.10],
        dtype=np.float32,
    )
    return np.clip(linear_to_srgb(linear * gains), 0.0, 1.0)


# ============================================================
# BASIC LIGHTROOM-LIKE ADJUSTMENTS
# ============================================================

def apply_exposure(image, value):

    stops = float(value)

    return np.clip(
        image * (2.0 ** stops),
        0.0,
        1.0,
    )


def apply_contrast(image, value):

    value = float(value)

    factor = 1.0 + value / 100.0

    return np.clip(
        (image - 0.5) * factor + 0.5,
        0.0,
        1.0,
    )


def luminance(image):

    return (
        image[..., 0] * 0.2126
        + image[..., 1] * 0.7152
        + image[..., 2] * 0.0722
    )


def apply_highlights(image, value):

    value = float(value) / 100.0

    lum = luminance(image)

    weight = np.clip(
        (lum - 0.5) / 0.5,
        0.0,
        1.0,
    )

    weight = weight[..., None]

    if value >= 0:

        result = image + (
            (1.0 - image)
            * weight
            * value
            * 0.7
        )

    else:

        result = image * (
            1.0
            + weight
            * value
            * 0.7
        )

    return np.clip(result, 0.0, 1.0)


def apply_shadows(image, value):

    value = float(value) / 100.0

    lum = luminance(image)

    weight = np.clip(
        (0.5 - lum) / 0.5,
        0.0,
        1.0,
    )

    weight = weight[..., None]

    if value >= 0:

        result = image + (
            (1.0 - image)
            * weight
            * value
            * 0.7
        )

    else:

        result = image * (
            1.0
            + weight
            * value
            * 0.7
        )

    return np.clip(result, 0.0, 1.0)


def apply_whites(image, value):

    value = float(value) / 100.0

    lum = luminance(image)

    weight = np.clip(
        (lum - 0.65) / 0.35,
        0.0,
        1.0,
    )[..., None]

    result = image + (
        (1.0 - image)
        * weight
        * value
        * 0.6
    )

    return np.clip(result, 0.0, 1.0)


def apply_blacks(image, value):

    value = float(value) / 100.0

    lum = luminance(image)

    weight = np.clip(
        (0.35 - lum) / 0.35,
        0.0,
        1.0,
    )[..., None]

    result = image + (
        image
        * weight
        * value
        * 0.8
    )

    return np.clip(result, 0.0, 1.0)


def apply_saturation(image, value):

    value = float(value)

    gray = luminance(image)[..., None]

    factor = 1.0 + value / 100.0

    return np.clip(
        gray + (image - gray) * factor,
        0.0,
        1.0,
    )


def apply_vibrance(image, value):

    value = float(value) / 100.0

    hsv = cv2.cvtColor(
        image.astype(np.float32),
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[..., 1]

    # Vibrance primarily boosts less-saturated pixels.
    if value >= 0:

        boost = (
            value
            * (1.0 - saturation)
        )

        hsv[..., 1] = np.clip(
            saturation * (1.0 + boost),
            0.0,
            1.0,
        )

    else:

        hsv[..., 1] = np.clip(
            saturation * (1.0 + value),
            0.0,
            1.0,
        )

    return np.clip(
        cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2RGB,
        ),
        0.0,
        1.0,
    )


# ============================================================
# CLARITY / TEXTURE / DEHAZE APPROXIMATIONS
# ============================================================

def apply_clarity(image, value):

    amount = float(value) / 100.0

    if abs(amount) < 1e-6:
        return image

    blur = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=3.0,
    )

    detail = image - blur

    result = image + detail * amount * 1.5

    return np.clip(result, 0.0, 1.0)


def apply_texture(image, value):

    amount = float(value) / 100.0

    if abs(amount) < 1e-6:
        return image

    small_blur = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=1.0,
    )

    detail = image - small_blur

    result = image + detail * amount

    return np.clip(result, 0.0, 1.0)


def apply_dehaze(image, value):

    amount = float(value) / 100.0

    if abs(amount) < 1e-6:
        return image

    if amount > 0:

        result = (
            (image - 0.5)
            * (1.0 + amount * 0.8)
            + 0.5
        )

        result *= (
            1.0 - amount * 0.05
        )

    else:

        haze = -amount

        result = (
            image * (1.0 - haze * 0.25)
            + haze * 0.25
        )

    return np.clip(result, 0.0, 1.0)


def apply_detail(image, settings):
    """Approximate Lightroom Detail panel sharpening and noise reduction."""
    sharpness = setting(settings, "Sharpness")
    radius = np.clip(setting(settings, "SharpenRadius", "SharpnessRadius", default=1.0), 0.5, 3.0)
    detail = np.clip(setting(settings, "SharpenDetail", "SharpnessDetail", default=25.0), 0.0, 100.0)
    masking = np.clip(setting(settings, "SharpenEdgeMasking", "SharpnessMasking"), 0.0, 100.0)
    result = image
    if sharpness:
        blur = cv2.GaussianBlur(result, (0, 0), sigmaX=float(radius))
        high_pass = result - blur
        # Lightroom's Masking slider protects smooth areas from sharpening.
        edges = cv2.Canny(
            (luminance(result) * 255).astype(np.uint8), 30, 90
        ).astype(np.float32) / 255.0
        edge_weight = 1.0 - masking / 100.0 + edges * (masking / 100.0)
        result = result + high_pass * (sharpness / 100.0) * (0.5 + detail / 100.0) * edge_weight[..., None]

    luma_noise = setting(settings, "LuminanceSmoothing", "NoiseReduction", "LuminanceNoiseReduction")
    color_noise = setting(settings, "ColorNoiseReduction")
    if luma_noise:
        sigma = 0.3 + luma_noise / 100.0 * 2.2
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma)
        result = result * (1.0 - luma_noise / 140.0) + blurred * (luma_noise / 140.0)
    if color_noise:
        lab = cv2.cvtColor(np.clip(result, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        sigma = 0.3 + color_noise / 100.0 * 2.0
        lab[..., 1:] = cv2.GaussianBlur(lab[..., 1:], (0, 0), sigmaX=sigma)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(result, 0.0, 1.0)


def apply_grain(image, settings):
    amount = setting(settings, "GrainAmount")
    if amount == 0:
        return image
    size = np.clip(setting(settings, "GrainSize", default=25.0), 0, 100)
    frequency = np.clip(setting(settings, "GrainFrequency", default=50.0), 0, 100)
    rng = np.random.default_rng(20)  # Stable output for the same image/preset.
    noise = rng.normal(0, 1, image.shape[:2]).astype(np.float32)
    sigma = max(0.01, size / 100.0 * 1.4)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma)
    noise /= max(float(np.std(noise)), 1e-6)
    noise *= 0.008 * amount / 100.0 * (0.6 + frequency / 100.0 * 0.4)
    return np.clip(image + noise[..., None], 0.0, 1.0)


def apply_crop(image, settings):
    if not get_bool(settings, "HasCrop") and not any(
        key in settings for key in ("CropTop", "CropLeft", "CropBottom", "CropRight")
    ):
        return image
    height, width = image.shape[:2]
    # Lightroom records these as normalized values in most XMP files.
    left = np.clip(setting(settings, "CropLeft"), 0, 1)
    top = np.clip(setting(settings, "CropTop"), 0, 1)
    right = np.clip(setting(settings, "CropRight", default=1), 0, 1)
    bottom = np.clip(setting(settings, "CropBottom", default=1), 0, 1)
    if right <= left or bottom <= top:
        return image
    return image[round(top * height):round(bottom * height), round(left * width):round(right * width)]


# ============================================================
# TONE CURVES
# ============================================================

def curve_lut(points):

    if not points:
        return np.arange(
            256,
            dtype=np.float32,
        )

    points = sorted(
        points,
        key=lambda p: p[0],
    )

    xs = np.array(
        [p[0] for p in points],
        dtype=np.float32,
    )

    ys = np.array(
        [p[1] for p in points],
        dtype=np.float32,
    )

    # Lightroom curves normally use 0-255 coordinates.
    lut = np.interp(
        np.arange(256),
        xs,
        ys,
    )

    return np.clip(
        lut,
        0,
        255,
    )


def apply_curve_channel(channel, points):

    lut = curve_lut(points)

    indices = np.clip(
        channel * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return lut[indices] / 255.0


def apply_tone_curves(image, curves):

    result = image.copy()

    # --------------------------------------------------------
    # Master RGB curve
    # --------------------------------------------------------

    master_names = [
        "ToneCurvePV2012",
        "ToneCurvePV2012RGB",
    ]

    for name in master_names:

        if name in curves:

            for channel in range(3):

                result[..., channel] = (
                    apply_curve_channel(
                        result[..., channel],
                        curves[name],
                    )
                )

            break

    # --------------------------------------------------------
    # Individual RGB channel curves
    # --------------------------------------------------------

    channel_map = {
        "ToneCurvePV2012Red": 0,
        "ToneCurvePV2012Green": 1,
        "ToneCurvePV2012Blue": 2,
    }

    for name, channel in channel_map.items():

        if name in curves:

            result[..., channel] = (
                apply_curve_channel(
                    result[..., channel],
                    curves[name],
                )
            )

    return np.clip(result, 0.0, 1.0)


def apply_parametric_curve(image, settings):
    """Apply the legacy Parametric Curve sliders when no point curve is used."""
    values = [
        setting(settings, "ParametricShadows"),
        setting(settings, "ParametricDarks"),
        setting(settings, "ParametricLights"),
        setting(settings, "ParametricHighlights"),
    ]
    if not any(values):
        return image
    shadow_split = setting(settings, "ParametricShadowSplit", default=25) / 100.0
    mid_split = setting(settings, "ParametricMidtoneSplit", default=50) / 100.0
    high_split = setting(settings, "ParametricHighlightSplit", default=75) / 100.0
    lum = luminance(image)
    anchors = np.array([0.0, shadow_split, mid_split, high_split, 1.0], dtype=np.float32)
    shifts = np.array([values[0], values[1], values[2], values[3]], dtype=np.float32) / 100.0
    centers = (anchors[:-1] + anchors[1:]) / 2.0
    width = np.maximum((anchors[1:] - anchors[:-1]) / 2.0, 1e-5)
    change = np.zeros_like(lum)
    for center, half_width, amount in zip(centers, width, shifts):
        change += np.clip(1.0 - np.abs(lum - center) / half_width, 0, 1) * amount
    return np.clip(image + change[..., None] * 0.3, 0, 1)


# ============================================================
# HSL / COLOR MIXER
# ============================================================

HSL_COLORS = {
    "Red": 0,
    "Orange": 30,
    "Yellow": 60,
    "Green": 120,
    "Aqua": 180,
    "Blue": 240,
    "Purple": 275,
    "Magenta": 315,
}


def hue_distance(h, center):

    return np.abs(
        ((h - center + 180.0) % 360.0)
        - 180.0
    )


def color_weight(hue, center, width=45):

    distance = hue_distance(
        hue,
        center,
    )

    return np.clip(
        1.0 - distance / width,
        0.0,
        1.0,
    )


def apply_hsl_mixer(image, settings):

    hsv = cv2.cvtColor(
        image.astype(np.float32),
        cv2.COLOR_RGB2HSV,
    )

    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]

    for color, center in HSL_COLORS.items():

        weight = color_weight(
            hue,
            center,
        )

        # Hue
        hue_key = f"HueAdjustment{color}"

        if hue_key in settings:

            amount = (
                float(settings[hue_key])
                / 100.0
                * 30.0
            )

            hue = (
                hue
                + weight * amount
            ) % 360.0

        # Saturation
        sat_key = f"SaturationAdjustment{color}"

        if sat_key in settings:

            amount = (
                float(settings[sat_key])
                / 100.0
            )

            sat = np.clip(
                sat
                * (
                    1.0
                    + weight * amount
                ),
                0.0,
                1.0,
            )

        # Luminance
        lum_key = f"LuminanceAdjustment{color}"

        if lum_key in settings:

            amount = (
                float(settings[lum_key])
                / 100.0
            )

            val = np.clip(
                val
                + weight
                * amount
                * 0.35,
                0.0,
                1.0,
            )

    hsv[..., 0] = hue
    hsv[..., 1] = sat
    hsv[..., 2] = val

    return np.clip(
        cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2RGB,
        ),
        0.0,
        1.0,
    )


# ============================================================
# COLOR GRADING
# ============================================================

def hue_color(hue):

    hsv = np.array(
        [[[float(hue) % 360.0, 1.0, 1.0]]],
        dtype=np.float32,
    )

    rgb = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB,
    )

    return rgb[0, 0]


def grading_layer(
    image,
    hue,
    saturation,
    weight,
):

    if saturation == 0:
        return image

    color = hue_color(hue)

    strength = (
        saturation / 100.0
    )

    mix = (
        weight[..., None]
        * strength
        * 0.35
    )

    return (
        image * (1.0 - mix)
        + color * mix
    )


def apply_color_grading(image, settings):

    lum = luminance(image)

    # Lightroom's actual color grading masks are proprietary.
    # These smooth masks approximate shadow/midtone/highlight
    # tonal ranges.

    shadow_weight = np.clip(
        (0.55 - lum) / 0.55,
        0.0,
        1.0,
    )

    highlight_weight = np.clip(
        (lum - 0.45) / 0.55,
        0.0,
        1.0,
    )

    midtone_weight = np.clip(
        1.0
        - np.abs(lum - 0.5) / 0.5,
        0.0,
        1.0,
    )

    balance = setting(settings, "ColorGradeBalance", "SplitToningBalance") / 100.0
    # Balance moves the split point between the shadow and highlight grades.
    shadow_weight = np.clip(shadow_weight - balance * 0.35, 0.0, 1.0)
    highlight_weight = np.clip(highlight_weight + balance * 0.35, 0.0, 1.0)
    blending = np.clip(setting(settings, "ColorGradeBlending", default=50) / 100.0, 0.0, 1.0)
    midtone_weight *= 0.5 + blending * 0.5

    # Lightroom commonly stores these names.
    shadow_hue = float(
        settings.get(
            "ColorGradeShadowHue",
            settings.get(
                "SplitToningShadowHue",
                0,
            ),
        )
    )

    shadow_sat = float(
        settings.get(
            "ColorGradeShadowSat",
            settings.get(
                "SplitToningShadowSaturation",
                0,
            ),
        )
    )

    mid_hue = float(
        settings.get(
            "ColorGradeMidtoneHue",
            0,
        )
    )

    mid_sat = float(
        settings.get(
            "ColorGradeMidtoneSat",
            0,
        )
    )

    high_hue = float(
        settings.get(
            "ColorGradeHighlightHue",
            settings.get(
                "SplitToningHighlightHue",
                0,
            ),
        )
    )

    high_sat = float(
        settings.get(
            "ColorGradeHighlightSat",
            settings.get(
                "SplitToningHighlightSaturation",
                0,
            ),
        )
    )

    result = grading_layer(
        image,
        shadow_hue,
        shadow_sat,
        shadow_weight,
    )

    result = grading_layer(
        result,
        mid_hue,
        mid_sat,
        midtone_weight,
    )

    result = grading_layer(
        result,
        high_hue,
        high_sat,
        highlight_weight,
    )

    # Luminance sliders in the Color Grading panel brighten/darken only their
    # selected tonal range. ShadowTint is the legacy shadow green↔magenta bias.
    shadow_lum = setting(settings, "ColorGradeShadowLum")
    mid_lum = setting(settings, "ColorGradeMidtoneLum")
    high_lum = setting(settings, "ColorGradeHighlightLum")
    global_lum = setting(settings, "ColorGradeGlobalLum")
    local_luminance = (
        shadow_weight * shadow_lum
        + midtone_weight * mid_lum
        + highlight_weight * high_lum
    ) / 100.0
    result = np.clip(result + local_luminance[..., None] * 0.25, 0.0, 1.0)
    if global_lum:
        result = np.clip(result + global_lum / 100.0 * 0.25, 0.0, 1.0)
    shadow_tint = setting(settings, "ShadowTint")
    if shadow_tint:
        result[..., 0] += shadow_weight * shadow_tint / 100.0 * 0.08
        result[..., 1] -= shadow_weight * shadow_tint / 100.0 * 0.10
        result[..., 2] += shadow_weight * shadow_tint / 100.0 * 0.08

    # Global grading
    global_hue = float(
        settings.get(
            "ColorGradeGlobalHue",
            0,
        )
    )

    global_sat = float(
        settings.get(
            "ColorGradeGlobalSat",
            0,
        )
    )

    if global_sat != 0:

        result = grading_layer(
            result,
            global_hue,
            global_sat,
            np.ones_like(lum),
        )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# VIGNETTE
# ============================================================

def apply_vignette(image, amount):

    amount = float(amount) / 100.0

    if abs(amount) < 1e-6:
        return image

    height, width = image.shape[:2]

    y, x = np.ogrid[
        -1:1:complex(height),
        -1:1:complex(width),
    ]

    distance = np.sqrt(
        x * x + y * y
    )

    distance = np.clip(
        distance,
        0.0,
        1.0,
    )

    mask = distance ** 2

    if amount < 0:

        factor = (
            1.0
            + amount
            * mask
            * 0.75
        )

    else:

        factor = (
            1.0
            + amount
            * mask
            * 0.5
        )

    return np.clip(
        image * factor[..., None],
        0.0,
        1.0,
    )


def apply_vignette_settings(image, settings):
    """Apply the Effects-panel vignette including its shape controls."""
    amount = setting(settings, "PostCropVignetteAmount")
    if amount == 0:
        return image
    height, width = image.shape[:2]
    y, x = np.ogrid[-1:1:complex(height), -1:1:complex(width)]
    roundness = setting(settings, "PostCropVignetteRoundness") / 100.0
    aspect = width / max(height, 1)
    # Positive roundness makes the vignette circular; negative makes it match
    # the frame.  This is an approximation of Lightroom's Style 1 behavior.
    x_scale = (1.0 - roundness * 0.45) / max(aspect, 1e-6)
    y_scale = 1.0 + roundness * 0.45
    distance = np.sqrt((x * x_scale) ** 2 + (y * y_scale) ** 2)
    midpoint = np.clip(setting(settings, "PostCropVignetteMidpoint", default=50) / 100.0, 0.01, 0.99)
    feather = np.clip(setting(settings, "PostCropVignetteFeather", default=50) / 100.0, 0.01, 1.0)
    start = midpoint * 0.9
    edge = np.clip((distance - start) / max(1.0 - start, 1e-6), 0, 1)
    mask = edge ** (1.0 / (0.35 + feather * 1.65))
    factor = 1.0 + amount / 100.0 * mask * 0.75
    highlight_protection = np.clip(
        setting(settings, "PostCropVignetteHighlightContrast") / 100.0,
        -1.0,
        1.0,
    )
    if highlight_protection:
        # Lightroom's Highlight Priority style lets brighter edge details hold
        # more of their original brightness through a dark vignette.
        factor += mask * luminance(image) * highlight_protection * 0.25
    return np.clip(image * factor[..., None], 0.0, 1.0)


def apply_calibration(image, settings):
    """Approximate the Calibration panel's primary hue/saturation controls."""
    result = image.copy()
    for channel, index, center in (("Red", 0, 0), ("Green", 1, 120), ("Blue", 2, 240)):
        hue_value = setting(settings, f"{channel}Hue")
        sat_value = setting(settings, f"{channel}Saturation")
        if hue_value == 0 and sat_value == 0:
            continue
        hsv = cv2.cvtColor(result.astype(np.float32), cv2.COLOR_RGB2HSV)
        weight = color_weight(hsv[..., 0], center, width=90)
        hsv[..., 0] = (hsv[..., 0] + weight * hue_value * 0.30) % 360.0
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + weight * sat_value / 100.0), 0, 1)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # A small primary gain carries the broad color bias of calibration.
        result[..., index] *= 1.0 + sat_value / 500.0
    return np.clip(result, 0.0, 1.0)


# ============================================================
# PRESET PROCESSOR
# ============================================================

SUPPORTED_KEYS = {
    "Exposure2012",
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Whites2012",
    "Blacks2012",
    "Texture",
    "Clarity2012",
    "Dehaze",
    "Vibrance",
    "Saturation",
    "PostCropVignetteAmount",
    "Temperature",
    "Tint",
    "IncrementalTemperature",
    "IncrementalTint",
    "ConvertToGrayscale",
    "ShadowTint",
    "ColorNoiseReduction",
    "LuminanceSmoothing",
    "GrainAmount",
    "GrainSize",
    "GrainFrequency",
    "HasCrop",
}

RENDERED_PREFIXES = (
    "HueAdjustment",
    "SaturationAdjustment",
    "LuminanceAdjustment",
    "ColorGrade",
    "SplitToning",
    "Parametric",
    "PostCropVignette",
    "Crop",
    "Sharpen",
    "Sharpness",
    "NoiseReduction",
    "LuminanceNoise",
    "RedHue", "RedSaturation",
    "GreenHue", "GreenSaturation",
    "BlueHue", "BlueSaturation",
)


def get_number(settings, name, default=0.0):

    value = settings.get(name)

    if value is None:
        return default

    try:
        return float(value)

    except ValueError:
        return default


def apply_preset(image, settings, curves):

    result = apply_crop(image.copy(), settings)

    if get_bool(settings, "ConvertToGrayscale"):
        gray = luminance(result)
        result = np.repeat(gray[..., None], 3, axis=2)

    result = apply_white_balance(result, settings)

    # ========================================================
    # BASIC
    # ========================================================

    if "Exposure2012" in settings:
        result = apply_exposure(
            result,
            get_number(
                settings,
                "Exposure2012",
            ),
        )

    if "Contrast2012" in settings:
        result = apply_contrast(
            result,
            get_number(
                settings,
                "Contrast2012",
            ),
        )

    if "Highlights2012" in settings:
        result = apply_highlights(
            result,
            get_number(
                settings,
                "Highlights2012",
            ),
        )

    if "Shadows2012" in settings:
        result = apply_shadows(
            result,
            get_number(
                settings,
                "Shadows2012",
            ),
        )

    if "Whites2012" in settings:
        result = apply_whites(
            result,
            get_number(
                settings,
                "Whites2012",
            ),
        )

    if "Blacks2012" in settings:
        result = apply_blacks(
            result,
            get_number(
                settings,
                "Blacks2012",
            ),
        )

    # ========================================================
    # TEXTURE / CLARITY / DEHAZE
    # ========================================================

    if "Texture" in settings:
        result = apply_texture(
            result,
            get_number(
                settings,
                "Texture",
            ),
        )

    if "Clarity2012" in settings:
        result = apply_clarity(
            result,
            get_number(
                settings,
                "Clarity2012",
            ),
        )

    if "Dehaze" in settings:
        result = apply_dehaze(
            result,
            get_number(
                settings,
                "Dehaze",
            ),
        )

    result = apply_detail(result, settings)

    # ========================================================
    # TONE CURVES
    # ========================================================

    if curves:
        result = apply_tone_curves(result, curves)
    else:
        result = apply_parametric_curve(result, settings)

    # ========================================================
    # HSL / COLOR MIXER
    # ========================================================

    result = apply_hsl_mixer(
        result,
        settings,
    )

    result = apply_calibration(result, settings)

    # ========================================================
    # VIBRANCE / SATURATION
    # ========================================================

    if "Vibrance" in settings:
        result = apply_vibrance(
            result,
            get_number(
                settings,
                "Vibrance",
            ),
        )

    if "Saturation" in settings:
        result = apply_saturation(
            result,
            get_number(
                settings,
                "Saturation",
            ),
        )

    # ========================================================
    # COLOR GRADING
    # ========================================================

    result = apply_color_grading(
        result,
        settings,
    )

    # ========================================================
    # VIGNETTE
    # ========================================================

    result = apply_grain(result, settings)
    result = apply_vignette_settings(result, settings)

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# REPORT XMP CONTENT
# ============================================================

def report_settings(settings, curves):

    print()
    print("=" * 70)
    print("XMP PRESET")
    print("=" * 70)

    print("\nScalar settings found:")

    for key in sorted(settings):

        print(
            f"  {key}: {settings[key]}"
        )

    print("\nTone curves found:")

    if curves:

        for name, points in curves.items():

            print(
                f"  {name}: "
                f"{len(points)} points"
            )

            print(
                f"      {points}"
            )

    else:

        print("  None")

    # --------------------------------------------------------
    # Detect settings this script does not currently reproduce
    # --------------------------------------------------------

    ignored = []

    metadata_keys = {
        "PresetType",
        "Cluster",
        "UUID",
        "SupportsAmount",
        "SupportsColor",
        "SupportsMonochrome",
        "SupportsHighDynamicRange",
        "SupportsNormalDynamicRange",
        "SupportsSceneReferred",
        "SupportsOutputReferred",
        "CameraModelRestriction",
        "Copyright",
        "ContactInfo",
        "Version",
        "ProcessVersion",
        "Name",
        "Group",
        "SortName",
        "ShortName",
        "Description",
        "ShowInPresets",
        "ShowInQuickActions",
        "SupportsAmount2",
        "RequiresRGBTables",
        "HasSettings",
        "WhiteBalance",
        "ToneCurveName2012",
        "OverrideLookVignette",
        "HDREditMode",
        "CurveRefineSaturation",
    }

    for key in settings:

        if key in SUPPORTED_KEYS:
            continue

        if key in metadata_keys:
            continue

        if key.startswith(RENDERED_PREFIXES):
            continue

        ignored.append(key)

    if ignored:

        print()
        print(
            "Settings present but not currently "
            "reproduced by this processor:"
        )

        for key in sorted(ignored):

            print(
                f"  - {key}: {settings[key]}"
            )

    print("=" * 70)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Test Lightroom XMP presets "
            "on normal RGB images."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input JPG/PNG image",
    )

    parser.add_argument(
        "--preset",
        required=True,
        help="Lightroom/Camera Raw .xmp preset",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    preset_path = Path(args.preset)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not preset_path.exists():

        raise FileNotFoundError(
            f"Preset not found: {preset_path}"
        )

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    if args.output:

        output_path = Path(
            args.output
        )

    else:

        output_path = (
            Path(__file__).parent
            / "outputs"
            / (
                f"{image_path.stem}"
                f"__{preset_path.stem}.jpg"
            )
        )

    # --------------------------------------------------------
    # Parse preset
    # --------------------------------------------------------

    settings, curves = parse_xmp(
        preset_path
    )

    report_settings(
        settings,
        curves,
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    print(
        f"Loading image: {image_path}"
    )

    image = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Apply preset
    # --------------------------------------------------------

    print(
        f"Applying preset: {preset_path.name}"
    )

    result = apply_preset(
        image,
        settings,
        curves,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_image(
        result,
        output_path,
    )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
