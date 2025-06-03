import cv2
import numpy as np


def main():
    np.set_printoptions(suppress=True, precision=15)
    # Independent margin fractions [0.0 = no change, >0 expand, <0 shrink]
    margins = {
        "left": 0.015,
        "right": 0.015,
        "top": 0.025,
        "bottom": 0.025,
    }

    # init ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera")
        return

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None and len(ids) >= 4:
                ids_list = ids.flatten().tolist()
                required = [0, 1, 2, 3]
                if all(r in ids_list for r in required):
                    # build src array (TL, TR, BR, BL)
                    src = np.zeros((4, 2), dtype=np.float32)
                    # marker id → corner index in corners[][][0]: 0→TL, 1→TR, 2→BR, 3→BL
                    mapping = {0: 0, 1: 1, 2: 2, 3: 3}
                    for marker_id, corner_idx in mapping.items():
                        i = ids_list.index(marker_id)
                        src[marker_id] = corners[i][0][corner_idx]

                    # draw original quad (green)
                    cv2.polylines(frame,
                                  [src.reshape(-1, 1, 2).astype(int)],
                                  isClosed=True, color=(0, 255, 0), thickness=2)

                    # compute pixel margins
                    mpx = {
                        "left": margins["left"] * w,
                        "right": margins["right"] * w,
                        "top": margins["top"] * h,
                        "bottom": margins["bottom"] * h,
                    }

                    # build dst array (blue) by expanding/shrinking each side
                    dst = np.zeros_like(src)
                    # TL
                    dst[0] = (src[0, 0] - mpx["left"], src[0, 1] - mpx["top"])
                    # TR
                    dst[1] = (src[1, 0] + mpx["right"], src[1, 1] - mpx["top"])
                    # BR
                    dst[2] = (src[2, 0] + mpx["right"], src[2, 1] + mpx["bottom"])
                    # BL
                    dst[3] = (src[3, 0] - mpx["left"], src[3, 1] + mpx["bottom"])

                    # draw expanded quad (blue)
                    cv2.polylines(frame,
                                  [dst.reshape(-1, 1, 2).astype(int)],
                                  isClosed=True, color=(255, 0, 0), thickness=2)

                    # compute homography mapping src → dst (ensure using the blue corners)
                    src_pts = src.astype(np.float32)
                    dst_pts = dst.astype(np.float32)
                    H, status = cv2.findHomography(src_pts, dst_pts)
                    if H is not None:
                        print("Homography matrix H:")
                        print(H)
                    else:
                        print("Failed to compute homography")

            # overlay detected markers
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.imshow("Aruco w/ Individual Margins", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()