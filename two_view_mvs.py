import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from feature_macthing import Feature_matching


class Two_view_mvs:
    def __init__(self, img1, img2, intrinsic_matrix):
        self.img1 = img1
        self.img2 = img2
        self.intrinsic_matrix = intrinsic_matrix
        # get the matching points coordinates in the two images
        self.feature_matching = Feature_matching(img1, img2)
        self.src_pts, self.dst_pts = self.feature_matching.get_matches()
        # get the extrinsic parameters
        self.R, self.T = self.compute_view_angles()
        # R2->P2, (x, y)->(x, y, 1), K^-1 * x = x'
        self.src_pts = np.vstack(
            (self.src_pts.T, np.ones(self.src_pts.shape[0])))
        self.dst_pts = np.vstack(
            (self.dst_pts.T, np.ones(self.dst_pts.shape[0]))
        )
        self.src_pts = np.dot(np.linalg.inv(
            self.intrinsic_matrix), self.src_pts)
        self.dst_pts = np.dot(np.linalg.inv(
            self.intrinsic_matrix), self.dst_pts)
        # get epipolar points
        # ! epipolar_points_src and epipolar_points_dst are in homogeneous coordinates and K^-1 * x
        self.epipolar_points_src, self.epipolar_points_dst = self.get_epipolar_points(
            self.R, self.T)

    def draw_3d_matches(self, windows_size=5):
        # Draw the matches
        point3D_list = []
        for pix_x in range(200, 600, 40):
            for pix_y in range(200, 600, 40):
                src_pix = [pix_x, pix_y]
                dst_pix = self.get_corresponding_pixel(
                    src_pix, self.R, self.T, self.intrinsic_matrix, self.epipolar_points_dst, windows_size)
                point3D_list.append(self.get_3d_points(
                    np.array([[src_pix[0]], [src_pix[1]], [1]]), np.array([[dst_pix[0]], [dst_pix[1]], [1]]), self.R, self.T))
        # Draw the 3D points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(point3D_list)):
            ax.scatter(point3D_list[i][0],
                       point3D_list[i][1], point3D_list[i][2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def compute_view_angles(self):
        # Find the essential matrix
        E, mask = cv2.findEssentialMat(
            self.src_pts, self.dst_pts, self.intrinsic_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Decompose the essential matrix to get rotation and translation
        _, R, T, mask = cv2.recoverPose(
            E, self.src_pts, self.dst_pts, self.intrinsic_matrix)

        return R, T

    def get_epipolar_points(self, R, T):
        # get the epipolar points: dst origin projected to the src image
        epipolar_points_src = T
        epipolar_points_src = epipolar_points_src / epipolar_points_src[2][0]
        # get the epipolar points: src origin projected to the dst image
        epipolar_points_dst = np.dot(R.T, -T)
        epipolar_points_dst = epipolar_points_dst / epipolar_points_dst[2][0]

        return epipolar_points_src, epipolar_points_dst

    def get_corresponding_pixel(self, pixel_src, R, T, instrinsic_matrix, epipolar_points_dst, windows_size=5):
        # ! pixel_src should be [pixel_x, pixel_y]
        # get pixel_src in homogeneous coordinates and K^-1 * x and column vector
        x0 = np.array([[pixel_src[0]], [pixel_src[1]], [1]])  # in image field
        pixel_src = np.array([[pixel_src[0]], [pixel_src[1]], [1]])
        pixel_src = np.dot(np.linalg.inv(intrinsic_matrix), pixel_src)
        # project the pixel_src to a line in the second image
        line_dst = np.dot(R.T, pixel_src - T)
        line_dst = line_dst / line_dst[2][0]
        x1 = np.dot(instrinsic_matrix, line_dst)  # in image field
        x2 = np.dot(instrinsic_matrix, epipolar_points_dst)  # in image field
        # line_dst and epipolar_points_dst compose a line (x1, x2)
        # extract the window around the pixel_src
        window_src = self.img1[x0[1][0]-windows_size:x0[1][0]+windows_size+1,
                               x0[0][0]-windows_size:x0[0][0]+windows_size+1]
        best_score = 0
        best_match = (0, 0)
        # search the corresponding pixel on the line in the dst image
        for x in range(windows_size, 800-windows_size):
            y = int((x - x1[0][0]) * (x2[1][0] - x1[1][0]) /
                    (x2[0][0] - x1[0][0]) + x1[1][0])
            if y <= windows_size and y > 800-windows_size:
                continue
            # calculate the matching degree
            # Extract windows around the destination pixels
            window_dst = self.img2[y-windows_size:y+windows_size+1,
                                   x-windows_size:x+windows_size+1]
            if window_src.shape == window_dst.shape:
                # Calculate the normalized cross-correlation
                result = cv2.matchTemplate(
                    window_dst, window_src, cv2.TM_CCOEFF_NORMED)
                score = result[0][0]
                if score > best_score:
                    best_score = score
                    best_match = (x, y)

        return best_match

    def get_3d_points(self, src_pos, dst_pos, R, T):
        # ! src_pos and dst_pos should be K^-1 * x and in P2 and column vector
        # calculate lambda2 * dst = lambda1 * R * src + T
        x1 = np.dot(R, src_pos)
        x2 = dst_pos
        # linear equation: X * [lambda1, lambda2]^T = t
        t = np.array([[T[0][0]], [T[1][0]]])
        X_matrix = np.array([[x2[0][0], -x1[0][0]], [x2[1][0], -x1[1][0]]])
        lambda2, lambda1 = np.dot(np.linalg.inv(X_matrix), t)
        # get the 3D points from two views
        P_through_src = lambda1 * src_pos
        P_through_dst = np.dot(self.R.T, lambda2 * dst_pos - self.T)
        P = (P_through_src + P_through_dst) / 2

        return P


if __name__ == "__main__":
    img1 = cv2.imread('image/r_85.png')
    img2 = cv2.imread('image/r_99.png')
    intrinsic_matrix = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]])
    two_view_mvs = Two_view_mvs(img1, img2, intrinsic_matrix)
    two_view_mvs.draw_3d_matches()
