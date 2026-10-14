# cSpell:ignore matplotlib seaborn sklearn numpy pyplot figsize imshow cmap fontsize suptitle xticks savefig bbox ravel argmax argsort softmax relu crossentropy adam onednn mnist keras tensorflow conv maxpooling dropout flatten tkinter ellipse lanczos astype ndarray dtype subplots cbar annot fmt xlabel ylabel ylim xlim rcparams tight layout tqdm ndimage scipy thumbnail
# ============================================================
# Helper: turn ANY picture of a handwritten digit into the
# format the CNN was trained on (28x28, white digit on black).
#
#   prepare_digit(img)               -> clean drawings (the drawing window)
#   prepare_digit(img, photo=True)   -> photos of paper (shadows, uneven light)
# ============================================================

import numpy as np
from PIL import Image
from scipy import ndimage   # installed together with scikit-learn


def _strokes_from_drawing(gray):
    """Clean picture (e.g. white pen on a black canvas): stretch the contrast
    and make sure the digit is bright on a dark background."""
    low, high = gray.min(), gray.max()
    if high - low < 1:
        return None
    gray = (gray - low) / (high - low) * 255.0
    if gray.mean() > 127:          # mostly light = dark ink on white paper
        gray = 255.0 - gray        # flip the colours
    return gray


def _strokes_from_photo(gray):
    """Photo of handwriting on paper. Shadows and uneven light are removed by
    keeping only thin lines (the pen strokes), not large dark or bright areas."""
    # 1. Work on a smaller copy so the filters are fast.
    small = Image.fromarray(gray.astype("uint8"))
    small.thumbnail((500, 500), Image.Resampling.LANCZOS)
    g = np.array(small, dtype="float32")

    # 2. A window a bit bigger than the pen stroke. Closing fills thin DARK lines
    #    with the paper colour, opening removes thin BRIGHT lines. Subtracting the
    #    original leaves just the pen strokes and ignores wide shadows.
    window = max(5, int(0.08 * max(g.shape)) | 1)
    dark_ink = ndimage.grey_closing(g, size=(window, window)) - g     # dark pen on light paper
    light_ink = g - ndimage.grey_opening(g, size=(window, window))    # light pen on dark paper
    # Decide which kind of pen it is by comparing the STRONGEST responses. (Comparing
    # the totals is not reliable, because shadow edges add a lot of weak response.)
    def strongest(values):
        return np.sort(values.ravel())[-max(1, values.size // 200):].mean()

    ink = dark_ink if strongest(dark_ink) >= strongest(light_ink) else light_ink

    peak = ink.max()
    if peak < 25:                  # almost nothing is written
        return None

    # 3. Keep the strong pixels and remove tiny specks of dust or paper texture.
    mask = ink > 0.35 * peak
    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    areas = ndimage.sum(mask, labels, index=range(1, count + 1))
    keep = [i + 1 for i, a in enumerate(areas) if a >= 0.15 * areas.max()]
    mask = np.isin(labels, keep)

    # 4. MNIST pen strokes are thick (about 1/8 of the digit height). A thin pen line
    #    is thickened until it is about as bold.
    ys, xs = np.where(mask)
    digit = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    size = max(digit.shape)
    distance = ndimage.distance_transform_edt(digit)
    stroke_width = 2.2 * np.percentile(distance[digit], 90)
    wanted = size / 8.0
    if stroke_width < wanted:
        mask = ndimage.binary_dilation(mask, iterations=int(np.ceil((wanted - stroke_width) / 2)))

    return mask.astype("float32") * 255.0


def _to_mnist_format(strokes, centre_by_mass):
    """Crop to the digit, fit it in a 20x20 box and place it on a 28x28 canvas."""
    ys, xs = np.where(strokes > 60)
    if len(xs) == 0:
        return None
    digit = strokes[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    # MNIST digits fit in a 20x20 box inside the 28x28 picture, so do the same.
    height, width = digit.shape
    scale = 20.0 / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    digit_img = Image.fromarray(digit.astype("uint8")).resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit_img, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    result = np.array(canvas, dtype="float32") / 255.0

    # MNIST digits are centred by their centre of mass, so we do that too (photos only).
    if centre_by_mass and result.sum() > 0:
        row, col = ndimage.center_of_mass(result)
        result = ndimage.shift(result, (13.5 - row, 13.5 - col), order=1, mode="constant")
        result = np.clip(result, 0.0, 1.0)

    return result


def prepare_digit(img, photo=False):
    """Return a (28, 28) float array with values 0-1, or None if no digit is found."""
    gray = np.array(img.convert("L"), dtype="float32")
    strokes = _strokes_from_photo(gray) if photo else _strokes_from_drawing(gray)
    if strokes is None:
        return None
    return _to_mnist_format(strokes, centre_by_mass=photo)