from pathlib import Path

import fire
from PIL import Image


def main(input: str, scale: int) -> None:
    """Upscale an image of an AprilTag and save it.

    Args:
        input: Path to the input image of the AprilTag.
        scale: Scale factor for upscaling the AprilTag.
    """
    assert scale > 1
    # read and upscale input
    img = Image.open(input)
    upscaled = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    # save to png file
    path = Path(input)
    path = Path(__file__).parent / f"outputs/{path.stem}_{scale}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    upscaled.save(str(path))
    print(f"Saved ChArUco board to {path}")


if __name__ == "__main__":
    fire.Fire(main)
