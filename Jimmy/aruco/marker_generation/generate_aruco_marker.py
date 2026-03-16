"""
Generate single ArUco marker PDFs for ground-truth distance collection.
The person wears / holds this marker while being recorded.

Usage:
    python generate_aruco_marker.py
    python generate_aruco_marker.py --ids 0 1 2 --size_cm 15

Output goes to marker_output/ by default.
"""

import os
import argparse

import cv2
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas


def cm_to_pt(cm: float) -> float:
    return (cm / 2.54) * 72.0


def main():
    ap = argparse.ArgumentParser(description="Generate single ArUco marker PDFs")
    ap.add_argument("--out_dir", default="marker_output")
    ap.add_argument("--ids", type=int, nargs="+", default=[0],
                    help="Marker IDs to generate")
    ap.add_argument("--size_cm", type=float, default=15.0,
                    help="Printed marker side length in cm")
    ap.add_argument("--dict_name", default="DICT_4X4_50")
    ap.add_argument("--pixels", type=int, default=1200,
                    help="Marker image resolution (pixels)")
    ap.add_argument("--pagesize", default="A4", choices=["A4", "LETTER"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dict_name)
    )

    page_w, page_h = A4 if args.pagesize == "A4" else letter
    size_pt = cm_to_pt(args.size_cm)

    for marker_id in args.ids:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id,
                                                    args.pixels)
        img_path = os.path.join(args.out_dir,
                                f"aruco_id{marker_id}_{args.size_cm:.0f}cm.png")
        pdf_path = os.path.join(args.out_dir,
                                f"aruco_id{marker_id}_{args.size_cm:.0f}cm.pdf")

        cv2.imwrite(img_path, marker_img)

        c = canvas.Canvas(pdf_path, pagesize=(page_w, page_h))
        x = (page_w - size_pt) / 2
        y = (page_h - size_pt) / 2
        c.drawImage(img_path, x, y, width=size_pt, height=size_pt,
                    preserveAspectRatio=True, mask="auto")
        c.setFont("Helvetica", 12)
        c.drawCentredString(page_w / 2, y - 20,
                            f"ArUco ID {marker_id} — {args.size_cm:.0f} cm")
        c.save()

        print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
