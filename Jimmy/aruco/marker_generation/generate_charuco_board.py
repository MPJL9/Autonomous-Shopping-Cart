"""
Generate a printable ChArUco calibration board as PNG + PDF.

Usage:
    python generate_charuco_board.py
    python generate_charuco_board.py --squares_x 7 --squares_y 5 --square_len_cm 3.0

Output goes to board_output/ by default.
Print the PDF at 100% scale (disable "Fit to page").
"""

import os
import argparse

import cv2
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas


def cm_to_pt(cm: float) -> float:
    return (cm / 2.54) * 72.0


def save_png_as_pdf_centered(png_path, pdf_path, width_cm, height_cm,
                              pagesize="A4", label=""):
    page_w, page_h = A4 if pagesize.upper() == "A4" else letter
    w_pt = cm_to_pt(width_cm)
    h_pt = cm_to_pt(height_cm)
    x = (page_w - w_pt) / 2
    y = (page_h - h_pt) / 2

    c = canvas.Canvas(pdf_path, pagesize=(page_w, page_h))
    c.drawImage(png_path, x, y, width=w_pt, height=h_pt,
                preserveAspectRatio=True, mask="auto")
    if label:
        c.setFont("Helvetica", 12)
        c.drawCentredString(page_w / 2, y - 18, label)
    c.save()


def main():
    ap = argparse.ArgumentParser(description="Generate a ChArUco board PDF")
    ap.add_argument("--out_dir", default="board_output")
    ap.add_argument("--squares_x", type=int, default=7)
    ap.add_argument("--squares_y", type=int, default=5)
    ap.add_argument("--square_len_cm", type=float, default=3.0)
    ap.add_argument("--marker_len_cm", type=float, default=2.2)
    ap.add_argument("--dict_name", default="DICT_4X4_50")
    ap.add_argument("--pixels_per_square", type=int, default=200)
    ap.add_argument("--pagesize", default="A4", choices=["A4", "LETTER"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dict_name)
    )
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_len_cm / 100.0,
        args.marker_len_cm / 100.0,
        aruco_dict,
    )

    img_w = args.squares_x * args.pixels_per_square
    img_h = args.squares_y * args.pixels_per_square
    board_img = board.generateImage((img_w, img_h))

    tag = f"charuco_{args.squares_x}x{args.squares_y}_{args.square_len_cm:.1f}cm"
    png_path = os.path.join(args.out_dir, f"{tag}.png")
    pdf_path = os.path.join(args.out_dir, f"{tag}.pdf")

    cv2.imwrite(png_path, board_img)

    board_w_cm = args.squares_x * args.square_len_cm
    board_h_cm = args.squares_y * args.square_len_cm
    label = (f"ChArUco {args.squares_x}x{args.squares_y}, "
             f"square={args.square_len_cm:.1f}cm, "
             f"marker={args.marker_len_cm:.1f}cm, "
             f"dict={args.dict_name}")
    save_png_as_pdf_centered(png_path, pdf_path, board_w_cm, board_h_cm,
                              pagesize=args.pagesize, label=label)

    print(f"Saved PNG : {png_path}")
    print(f"Saved PDF : {pdf_path}")
    print("Print the PDF at 100% scale (disable 'Fit to page').")


if __name__ == "__main__":
    main()
