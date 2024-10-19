import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from feature_macthing import Feature_matching
# * have to know the intrinsic matrix of the camera


class Two_view_sfm:
    def __init__(self, img1, img2, intrinsic_matrix):
        self.img1 = img1
        self.img2 = img2
        # K^-1 is used to normalize the coordinates
        self.K_inv = np.linalg.inv(intrinsic_matrix)
        self.feature_matching = Feature_matching(img1, img2)
        # get the matching points coordinates in the two images
        self.src_pts, self.dst_pts = self.feature_matching.get_matches()
        # R2->P2, (x, y)->(x, y, 1)
        self.src_pts = np.vstack(  # (3, n)
            (self.src_pts.T, np.ones(self.src_pts.shape[0])))
        self.dst_pts = np.vstack(  # (3, n)
            (self.dst_pts.T, np.ones(self.dst_pts.shape[0])))

        # * K^-1 * x = x'
        self.src_pts = np.dot(self.K_inv, self.src_pts)
        self.dst_pts = np.dot(self.K_inv, self.dst_pts)

        # * calculate the essential matrix
        self.essential_matrix = self.get_essential_matrix(
            self.src_pts, self.dst_pts)

        # * decompose the essential matrix to get the rotation and translation
        self.R, self.T = self.decompose_essential_matrix(
            self.essential_matrix, self.src_pts, self.dst_pts)

        # * get the 3D points
        self.points_3d = self.get_3d_points()

    def get_3d_points(self):
        points_3d = []
        src_pts = self.src_pts
        dst_pts = self.dst_pts
        # calculate the 3D points
        for idx in range(src_pts.shape[1]):
            src = src_pts[:, idx]
            dst = dst_pts[:, idx]
            # calculate lambda2 * dst = lambda1 * R * src + T
            x1 = np.dot(self.R, src).squeeze()
            x2 = dst.squeeze()
            # linear equation: X * [lambda1, lambda2]^T = t
            t = np.array([[self.T[0]], [self.T[1]]])
            X_matrix = np.array([[x2[0], -x1[0]], [x2[1], -x1[1]]])
            lambda2, lambda1 = np.dot(np.linalg.inv(X_matrix), t)
            # get the 3D points from two views
            P_through_src = lambda1 * src
            P_through_dst = np.dot(self.R.T, lambda2 * dst - self.T)
            P = (P_through_src + P_through_dst) / 2

            points_3d.append(P)

        return np.array(points_3d).T

    def plot_3d_points(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points_3d[0], self.points_3d[1], self.points_3d[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # plot camera positions
        ax.scatter(0, 0, 0, c='r', marker='o')  # camera 1
        ax.scatter(self.T[0], self.T[1], self.T[2],
                   c='b', marker='o')  # camera 2

        plt.show()

    def get_essential_matrix(self, origin_src_pts, origin_dst_pts, random_times=1000):
        # use 8-point algorithm to calculate the essential matrix
        # randomly choose 8 points
        error = float('inf')
        E_final = None
        while random_times != 0:
            idx = np.random.choice(origin_src_pts.shape[1], 8, replace=False)
            src_pts = origin_src_pts[:, idx]  # (3, 8)
            dst_pts = origin_dst_pts[:, idx]  # (3, 8)
            # src_pts^T * E * dst_pts = 0 => A * E_vec = 0
            # E_vec = [e11, e12, e13, e21, e22, e23, e31, e32, e33]
            # a = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]
            # A^T (8, 9)
            # E_vec (9, 1)
            A = np.vstack((src_pts[0] * dst_pts[0], src_pts[1] * dst_pts[0],
                           dst_pts[0], src_pts[0] *
                           dst_pts[1], src_pts[1] * dst_pts[1],
                           dst_pts[1], src_pts[0], src_pts[1], np.ones(src_pts.shape[1]))).T
            # calculate min |AE|^2/|E|^2 => min |A*E_vec|^2/|E_vec|^2
            # E_vec is the minimum eigenvector of A^T * A
            _, _, V = np.linalg.svd(A)
            E_vec = V[-1]
            E = E_vec.reshape(3, 3)

            # test src_pts^T * E * dst_pts = 0
            # choose the E with the smallest error
            tmp_error = np.sum(
                np.abs(np.sum(src_pts * np.dot(E, dst_pts), axis=0)))
            if tmp_error < error:
                error = tmp_error
                E_final = E

            # decrease the random times
            random_times -= 1
        print('error:', error)

        return E_final

    def decompose_essential_matrix(self, E, src_pts, dst_pts):
        # SVD of the essential matrix
        U, S, V = np.linalg.svd(E)
        # the two possible rotation matrices
        # R1 = U*Rz(+90)*V^T, R2 = U*Rz(-90)*V^T
        Rz_p90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        Rz_n90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        R1 = np.dot(np.dot(U, Rz_p90), V)
        R2 = np.dot(np.dot(U, Rz_n90), V)
        # the two possible translation vectors
        # T1^ = U*Rz(+90)*S*U^T, T2^ = U*Rz(-90)*S*U^T
        T1_ = np.dot(np.dot(U, Rz_p90), np.dot(np.diag(S), U.T))
        T2_ = np.dot(np.dot(U, Rz_n90), np.dot(np.diag(S), U.T))
        T1 = np.array([[T1_[2][1]], [T1_[0][2]], [T1_[1][0]]]).squeeze()
        T2 = np.array([[T2_[2][1]], [T2_[0][2]], [T2_[1][0]]]).squeeze()
        # choose the correct rotation and translation
        # choose the one with positive depth
        # randomly choose a point to check the depth
        random_times = 100
        side = 0
        while random_times != 0:
            random_times -= 1

            idx = np.random.choice(src_pts.shape[1], 1, replace=False)
            src = src_pts[:, idx]
            dst = dst_pts[:, idx]
            # calculate lambda2 * dst = lambda1 * R * src + T
            x1 = np.dot(R1, src).squeeze()
            x2 = dst.squeeze()
            # linear equation: X * [lambda1, lambda2]^T = t
            t = np.array([[T1[0]], [T1[1]]])
            X_matrix = np.array([[x2[0], -x1[0]], [x2[1], -x1[1]]])
            lambda2, lambda1 = np.dot(np.linalg.inv(X_matrix), t)

            if lambda1 * lambda2 > 0:
                side += 1

        if side > 50:
            R = R1
            T = T1
        else:
            R = R2
            T = T2

        print('R:', R)
        print('T:', T)

        return R, T


if __name__ == "__main__":
    img1 = cv2.imread('image/r_85.png')
    img2 = cv2.imread('image/r_99.png')
    intrinsic_matrix = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]])
    two_view_sfm = Two_view_sfm(img1, img2, intrinsic_matrix)
    two_view_sfm.plot_3d_points()
