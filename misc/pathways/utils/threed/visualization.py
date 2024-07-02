import copy
import numpy as np
import open3d as o3d
import cv2


def WritePoints3DToPly(pts3d, colors, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)


def WritePosesToPly(P, output_path):
    # Save the camera coordinate frames as meshes for visualization
    m_cam = None
    for j in range(P.shape[0]):
        # R_d = P[j, :, :3]
        # C_d = -R_d.T @ P[j, :, 3]
        T = np.eye(4)
        T[:3, :3] = P[j, :, :3].transpose()
        T[:3, 3] = -P[j, :, :3].transpose() @ P[j, :, 3]
        m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        m.transform(T)
        if m_cam is None:
            m_cam = m
        else:
            m_cam += m
    o3d.io.write_triangle_mesh(output_path, m_cam)


def skewsymm(x):
    S = np.zeros((x.shape[0], 3, 3))
    S[:, 0, 1] = -x[:, 2]
    S[:, 1, 0] = x[:, 2]
    S[:, 0, 2] = x[:, 1]
    S[:, 2, 0] = -x[:, 1]
    S[:, 1, 2] = -x[:, 0]
    S[:, 2, 1] = x[:, 0]

    return S


def VisualizeTriangulationMultiPoses(X, uv1, P, Im, K):
    out_img = []
    for i in range(P.shape[0]):
        kp_img = np.zeros_like(Im[i])
        # kpt = [cv2.KeyPoint(uv1[i, 0], uv1[i, 1], 20)]
        # cv2.drawKeypoints(Im[i], kpt, kp_img, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kp_img = cv2.drawMarker(Im[i], (int(uv1[i, 0]), int(uv1[i, 1])),
                                color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=5)

        reproj = K @ P[i] @ np.hstack((X, np.array([1.0])))
        reproj = reproj / reproj[2]
        # reproj = [cv2.KeyPoint(reproj[0], reproj[1], 10)]
        # cv2.drawKeypoints(kp_img, reproj, kp_img, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kp_img = cv2.drawMarker(kp_img, (int(reproj[0]), int(reproj[1])),
                                color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=5)
        out_img.append(kp_img)

    out_img = np.concatenate(out_img, axis=1)
    small_out_image = cv2.resize(out_img, (out_img.shape[1] // 2, out_img.shape[0] // 2))
    cv2.imshow('kpts', small_out_image)
    cv2.waitKey(0)


def VisualizeMatches(im1_file, kp1, im2_file, kp2):
    im1 = cv2.imread(im1_file)
    im2 = cv2.imread(im2_file)

    cv_kp1 = [cv2.KeyPoint(x=kp1[i][0], y=kp1[i][1], _size=5) for i in range(len(kp1))]
    cv_kp2 = [cv2.KeyPoint(x=kp2[i][0], y=kp2[i][1], _size=5) for i in range(len(kp1))]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

    out_img = np.zeros_like(im1, dtype=np.uint8)
    out_img = cv2.drawMatches(im1, cv_kp1, im2, cv_kp2, matches, None)
    cv2.imshow('matches', out_img)
    cv2.waitKey(0)


def VisualizeTrack(track, Im_input, K):
    Im = copy.deepcopy(Im_input)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    out_img = []
    trackx = track[:, 0] * fx + cx
    tracky = track[:, 1] * fy + cy

    for i in range(Im.shape[0]):
        if track[i, 0] == -1 and track[i, 1] == -1:
            out_img.append(Im[i])
        else:
            kp_img = cv2.drawMarker(Im[i], (int(trackx[i]), int(tracky[i])),
                                    color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=5)

            # kp_img = cv2.drawMarker(kp_img, (int(reproj[0]), int(reproj[1])),
            #                         color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=5)
            out_img.append(kp_img)

    out_img = np.concatenate(out_img, axis=1)
    small_out_image = cv2.resize(out_img, (out_img.shape[1] // 4, out_img.shape[0] // 4))
    cv2.imshow('kpts', small_out_image)
    cv2.waitKey(0)


def VisualizeReprojectionError(P, X, track, K, Im):
    uv = track @ K.T
    kp_img = copy.deepcopy(Im)
    for i in range(uv.shape[0]):
        kp_img = cv2.drawMarker(kp_img, (int(uv[i, 0]), int(uv[i, 1])),
                                color=(0, 255, 0), markerType=cv2.MARKER_SQUARE, thickness=5)

    reproj = K @ P @ np.concatenate((X.T, np.ones((1, uv.shape[0]))), axis=0)
    reproj = reproj[:, reproj[2] > 0.3]
    reproj = reproj / reproj[2]
    for i in range(reproj.shape[1]):
        kp_img = cv2.drawMarker(kp_img, (int(reproj[0, i]), int(reproj[1, i])),
                                color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=5)
    cv2.imshow('kpts', kp_img)
    cv2.waitKey(0)


def VisualizeBadPoseImage(Im):
    out_img = []
    num_images, h, w, _ = Im.shape
    num_image_per_row = 8
    rows = num_images // num_image_per_row
    left_over = num_images - num_image_per_row * rows
    for i in range(num_images // num_image_per_row):
        image_row = np.transpose(Im[num_image_per_row * i:num_image_per_row * (i + 1)], (1, 0, 2, 3))
        image_row = image_row.reshape((h, num_image_per_row * w, 3))
        out_img.append(image_row)

    out_img = np.concatenate(out_img, axis=0)
    small_out_image = cv2.resize(out_img, (out_img.shape[1] // 8, out_img.shape[0] // 8))
    cv2.imshow('bad pose images', small_out_image)
    cv2.waitKey(0)


def draw_camera(K, R, t, w, h,
                scale=1.0, color=[0.8, 0.2, 0.8]):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8 * scale)
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_in_world),
        lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]
    # return [plane, line_set]