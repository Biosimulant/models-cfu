"""Bounded uploaded-image decoding and inspectable display derivatives."""
import hashlib
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_ANNOTATED_BYTES = 768 * 1024


def identity(raw):
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def decode(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError('Upload must be a bounded regular JPEG or PNG')
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as encoded:
        if encoded.format not in {'JPEG', 'PNG'} or encoded.width * encoded.height > MAX_PIXELS:
            raise ValueError('Upload is outside the JPEG/PNG pixel contract')
        if getattr(encoded, 'n_frames', 1) != 1:
            raise ValueError('Upload must contain one image, not an animation')
        orientation = encoded.getexif().get(274, 1)
        if type(orientation) is not int or orientation not in range(1, 9):
            raise ValueError('Upload has an invalid EXIF orientation')
        normalized = ImageOps.exif_transpose(encoded)
        try:
            normalized.load()
            image = normalized.convert('RGB')
        finally:
            normalized.close()
        metadata = {**identity(raw), 'format': encoded.format,
                    'stored_dimensions': list(encoded.size), 'normalized_dimensions': list(image.size),
                    'exif_orientation': orientation,
                    'coordinate_basis': 'EXIF-normalized original image pixels'}
    return image, metadata


def annotate(image, detections, path):
    # Display resizing never changes the retained original-frame detections.
    width, height = image.size
    for maximum in [1400, 1120, 896, 716]:
        scale = min(1., maximum / max(width, height))
        size = [max(1, round(width * scale)), max(1, round(height * scale))]
        display = image.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (size[0], size[1] + 64), '#101722')
        canvas.paste(display, (0, 0)); display.close()
        drawing = ImageDraw.Draw(canvas)
        for detection in detections:
            box = [v * scale for v in detection['bbox_xyxy']]
            drawing.rectangle(box, outline='#00ffb0', width=2)
        drawing.text((8, size[1] + 6), f"Visible colonies: {len(detections)} | human review required", fill='white')
        drawing.text((8, size[1] + 25), 'Boxes refer to the normalized original; display resized only.', fill='white')
        drawing.text((8, size[1] + 44), 'Research assistance; acquisition conditions may be unvalidated.', fill='white')
        stream = io.BytesIO(); canvas.save(stream, format='PNG', compress_level=9)
        raw = stream.getvalue()
        if len(raw) <= MAX_ANNOTATED_BYTES:
            canvas.close(); path = Path(path); path.write_bytes(raw); path.chmod(0o600)
            return {**identity(raw), 'display_dimensions': size, 'footer_height': 64,
                    'scale': scale, 'format': 'PNG'}
        canvas.close()
    raise RuntimeError('Annotated image exceeds the retained display bound')
