import numpy as np
import cv2


img = np.zeros((512, 512, 3), np.uint8)


def generate_anchors_reference(base_size, aspect_ratios, scales):
    """Generate base anchor to be used as reference of generating all anchors.
    Anchors vary only in width and height. Using the base_size and the
    different ratios we can calculate the wanted widths and heights.
    Scales apply to area of object.
    Args:
        base_size (int): Base size of the base anchor (square).
        aspect_ratios: Ratios to use to generate different anchors. The ratio
            is the value of height / width.
        scales: Scaling ratios applied to area.
    Returns:
        anchors: Numpy array with shape (total_aspect_ratios * total_scales, 4)
            with the corner points of the reference base anchors using the
            convention (x_min, y_min, x_max, y_max).
    """
    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
    base_scales = scales_grid.reshape(-1)
    base_aspect_ratios = aspect_ratios_grid.reshape(-1)

    aspect_ratio_sqrts = np.sqrt(base_aspect_ratios)
    heights = base_scales * aspect_ratio_sqrts * base_size
    widths = base_scales / aspect_ratio_sqrts * base_size

    # i = 5
    for h, w in zip(heights, widths):
        print(h, w)
    #     top_left = (int(256-w/2), int(256-h/2))
    #     bottom_right = (int(256+w/2), int(256+h/2))
    #     img_mod = cv2.rectangle(img, top_left, bottom_right, (255-i, 255-2*i, 255-3*i))
    #     i += 5

    # cv2.imshow("Rectangle", img_mod)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Center point has the same X, Y value.
    center_xy = 0
    # Create anchor reference.
    anchors = np.column_stack([
        center_xy - (widths - 1) / 2,
        center_xy - (heights - 1) / 2,
        center_xy + (widths - 1) / 2,
        center_xy + (heights - 1) / 2,
    ])

    real_heights = (anchors[:, 3] - anchors[:, 1]).astype(np.int)
    real_widths = (anchors[:, 2] - anchors[:, 0]).astype(np.int)

    if (real_widths == 0).any() or (real_heights == 0).any():
        raise ValueError(
            'base_size {} is too small for aspect_ratios and scales.'.format(
                base_size
            )
        )

    # return anchors


def main():
    base_size = 256
    aspect_ratio = [0.5, 1, 1.5]
    scales = [0.5, 1, 1.5, 2.]

    a = generate_anchors_reference(base_size, aspect_ratio, scales)
    print()


if __name__ == '__main__':
    main()
