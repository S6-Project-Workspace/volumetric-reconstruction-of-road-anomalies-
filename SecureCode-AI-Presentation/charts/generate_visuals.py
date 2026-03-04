from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUND = (8, 2, 18)
AXIS_COLOR = (80, 80, 110)
TEXT_COLOR = (215, 225, 245)
TARGET_COLOR = (123, 47, 247)
OBSERVED_COLOR = (33, 212, 253)
CPU_COLOR = (255, 110, 196)
RAM_COLOR = (120, 115, 245)
STORAGE_COLOR = (33, 212, 253)
IMAGE_SIZE = (1400, 900)
MARGIN = 140


def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_axes(draw: ImageDraw.ImageDraw, origin: tuple[int, int], width: int, height: int) -> None:
    x0, y0 = origin
    draw.line((x0, y0, x0 + width, y0), fill=AXIS_COLOR, width=3)
    draw.line((x0, y0, x0, y0 - height), fill=AXIS_COLOR, width=3)


def measure_multiline(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, line_spacing: int) -> tuple[int, int]:
    lines = text.split("\n")
    widths = []
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_height = sum(heights) + line_spacing * (len(lines) - 1)
    return max(widths), total_height


def draw_multiline(draw: ImageDraw.ImageDraw, position: tuple[float, float], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int], line_spacing: int) -> None:
    x, y = position
    for line in text.split("\n"):
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        line_height = bbox[3] - bbox[1]
        y += line_height + line_spacing


def draw_grouped_bars(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    width: int,
    height: int,
    labels: list[str],
    group_a: list[float],
    group_b: list[float],
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
    max_value: float,
    font: ImageFont.ImageFont,
    label_gap: int = 30,
) -> None:
    x0, y0 = origin
    groups = len(labels)
    spacing = width / groups
    bar_width = spacing * 0.28
    line_spacing = 6

    for i, label in enumerate(labels):
        center = x0 + spacing * (i + 0.5)

        for offset, value, color in ((-bar_width / 2, group_a[i], colors[0]), (bar_width / 2, group_b[i], colors[1])):
            bar_height = (value / max_value) * height
            x_start = center + offset - bar_width / 2
            x_end = center + offset + bar_width / 2
            y_start = y0 - bar_height
            draw.rounded_rectangle((x_start, y_start, x_end, y0), radius=12, fill=color)
            draw.text((center + offset - 20, y_start - 35), f"{value:.1f}", fill=TEXT_COLOR, font=font)

        label_width, label_height = measure_multiline(draw, label, font, line_spacing)
        label_x = center - label_width / 2
        label_y = y0 + label_gap
        draw_multiline(draw, (label_x, label_y), label, font, TEXT_COLOR, line_spacing)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    items: list[tuple[str, tuple[int, int, int]]],
    position: tuple[int, int],
    font: ImageFont.ImageFont,
    box_size: int = 28,
    gap: int = 18,
) -> None:
    x, y = position
    for label, color in items:
        draw.rounded_rectangle((x, y, x + box_size, y + box_size), radius=6, fill=color)
        draw.text((x + box_size + 16, y + 2), label, fill=TEXT_COLOR, font=font)
        y += box_size + gap


def chart_performance() -> Path:
    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(img)

    title_font = load_font(58)
    label_font = load_font(28)
    caption_font = load_font(24)

    title = "SecureCodeAI Performance Benchmarks"
    draw.text((MARGIN, MARGIN // 2), title, fill=TEXT_COLOR, font=title_font)

    origin = (MARGIN, IMAGE_SIZE[1] - MARGIN)
    axes_width = IMAGE_SIZE[0] - 2 * MARGIN
    axes_height = IMAGE_SIZE[1] - 3 * MARGIN

    draw_axes(draw, origin, axes_width, axes_height)

    metrics = [
        "Throughput\nreq/s",
        "Latency p50\nseconds",
        "Latency p95\nseconds",
        "Latency p99\nseconds",
        "Success Rate\npercent",
    ]
    targets = [1.0, 2.0, 5.0, 10.0, 95.0]
    observed = [1.2, 1.8, 4.2, 8.5, 97.0]
    max_value = max(targets + observed) * 1.15

    draw_grouped_bars(draw, origin, axes_width, axes_height, metrics, targets, observed, (TARGET_COLOR, OBSERVED_COLOR), max_value, caption_font)

    legend_items = [("Target", TARGET_COLOR), ("Observed", OBSERVED_COLOR)]
    draw_legend(draw, legend_items, (IMAGE_SIZE[0] - MARGIN - 260, MARGIN + 30), label_font)

    draw.text((MARGIN, IMAGE_SIZE[1] - MARGIN + 70), "Lower is better for latency metrics", fill=AXIS_COLOR, font=caption_font)

    path = OUTPUT_DIR / "performance_benchmarks.png"
    img.save(path)
    return path


def chart_resources() -> Path:
    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(img)

    title_font = load_font(58)
    label_font = load_font(28)
    caption_font = load_font(24)

    title = "SecureCodeAI Resource Requirements"
    draw.text((MARGIN, MARGIN // 2), title, fill=TEXT_COLOR, font=title_font)

    origin = (MARGIN, IMAGE_SIZE[1] - MARGIN)
    axes_width = IMAGE_SIZE[0] - 2 * MARGIN
    axes_height = IMAGE_SIZE[1] - 3 * MARGIN

    draw_axes(draw, origin, axes_width, axes_height)

    environments = ["Development", "Production", "Serverless"]
    cpu = [4.0, 8.0, 6.0]
    ram = [8.0, 16.0, 12.0]
    storage = [20.0, 50.0, 50.0]
    max_value = max(cpu + ram + storage) * 1.25

    draw_grouped_bars(draw, origin, axes_width, axes_height, environments, cpu, ram, (CPU_COLOR, RAM_COLOR), max_value, caption_font)

    # Overlay storage bars as outlines to avoid overcrowding
    spacing = axes_width / len(environments)
    bar_width = spacing * 0.3
    for i, value in enumerate(storage):
        center = origin[0] + spacing * (i + 0.5)
        bar_height = (value / max_value) * axes_height
        x_start = center + bar_width * 0.65
        x_end = x_start + bar_width * 0.7
        y_start = origin[1] - bar_height
        draw.rounded_rectangle((x_start, y_start, x_end, origin[1]), radius=12, outline=STORAGE_COLOR, width=4)
        draw.text((x_start, y_start - 35), f"{value:.0f}", fill=STORAGE_COLOR, font=caption_font)

    legend_items = [
        ("CPU Cores", CPU_COLOR),
        ("RAM (GB)", RAM_COLOR),
        ("Storage (GB)", STORAGE_COLOR),
    ]
    draw_legend(draw, legend_items, (IMAGE_SIZE[0] - MARGIN - 320, MARGIN + 30), label_font)

    draw.text((MARGIN, IMAGE_SIZE[1] - MARGIN + 70), "Serverless resources are auto-managed at runtime", fill=AXIS_COLOR, font=caption_font)

    path = OUTPUT_DIR / "resource_requirements.png"
    img.save(path)
    return path


if __name__ == "__main__":
    perf_path = chart_performance()
    res_path = chart_resources()
    print(f"Saved charts to {perf_path} and {res_path}")
