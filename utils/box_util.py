import torch
from PIL import ImageDraw, ImageFont

# Define colors for drawing
colors = (
    (79, 195, 247),
    (236, 64, 122),
    (126, 87, 194),
    (205, 220, 57),
    (103, 58, 183),
    (255, 160, 0)
)

def calc_iou(a, b):
    """
    Calculate Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        a (torch.Tensor): First set of bounding boxes with shape (N, 4).
        b (torch.Tensor): Second set of bounding boxes with shape (M, 4).

    Returns:
        torch.Tensor: IoU values with shape (N, M).
    """
    dims = (a.size(0), b.size(0), 4)
    a = a.unsqueeze(1).expand(*dims)
    b = b.unsqueeze(0).expand(*dims)

    x1 = torch.max(a[..., 0], b[..., 0])
    x2 = torch.min(a[..., 2], b[..., 2])
    y1 = torch.max(a[..., 1], b[..., 1])
    y2 = torch.min(a[..., 3], b[..., 3])

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    intersect = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    union = area_a + area_b - intersect

    return intersect / union

def draw_object_box(img, objects):
    """
    Draw bounding boxes and labels on an image.

    Args:
        img (PIL.Image.Image): The image on which to draw.
        objects (list of tuples): Each tuple contains bounding box coordinates, 
                                  score, and label (x1, y1, x2, y2, score, label).

    Returns:
        PIL.Image.Image: The image with bounding boxes and labels drawn.
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # Use default font

    for i, obj in enumerate(objects):
        if not obj:
            continue

        bg = colors[i % len(colors)]
        x1, y1, x2, y2, score, label = obj

        # Scale bounding box coordinates to image size
        x1 = int(x1 * img.size[0])
        x2 = int(x2 * img.size[0])
        y1 = int(y1 * img.size[1])
        y2 = int(y2 * img.size[1])

        # Create label text
        text = f"{label} ({score:.2f})"
        text_bbox = draw.textbbox((0, 0), text, font=font)  # Calculate text size
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Draw text background
        x0, y0 = x1, y1 - text_height
        draw.rectangle([x0, y0, x0 + text_width, y1], fill=bg)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=bg, width=2)

        # Draw text
        draw.text((x0, y0), text, fill=(0, 0, 0), font=font)

    return img
