from pathlib import Path

import cv2
import fire


def main(
    xs: int, ys: int, sq_len: float, mk_len: float, dict: str = "DICT_5X5_50", dpi: int = 300
) -> None:
    """Generate image of ChArUco board with the given specs.

    Args:
        xs: Number of chessboard squares in X direction.
        ys: Number of chessboard squares in Y direction.
        sq_len: Length of square sides of chessboard.
        mk_len: Length of square sides of ArUco markers.
        dict: ArUco dictionary to use.
        dpi: Output DPI for generated image (affects resolution).
    """
    assert xs > 0 and ys > 0
    assert sq_len > 0 and mk_len > 0 and sq_len > mk_len
    assert hasattr(cv2.aruco, dict)
    assert dpi > 0
    # create ChArUco board
    dict_id = getattr(cv2.aruco, dict)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = board = cv2.aruco.CharucoBoard((xs, ys), sq_len, mk_len, aruco_dict)

    # render board
    INCHES_PER_METER = 39.3701
    width_m, height_m = xs * sq_len, ys * sq_len
    width_px = int(width_m * INCHES_PER_METER * dpi)
    height_px = int(height_m * INCHES_PER_METER * dpi)
    img = board.generateImage((width_px, height_px), 10, 100)

    # save to png file
    name = f"charuco_{xs}x{ys}_{sq_len:.2f}_{mk_len:.2f}_{dict}"
    path = Path(__file__).parent / f"outputs/{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)
    print(f"Saved ChArUco board to {path}")


if __name__ == "__main__":
    fire.Fire(main)
