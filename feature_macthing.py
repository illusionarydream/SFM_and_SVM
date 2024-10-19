import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Feature_matching:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.kp1, self.des1 = self.get_keypoints(self.img1)
        self.kp2, self.des2 = self.get_keypoints(self.img2)
        self.matches = self.get_coarse_matches(self.des1, self.des2)
        self.matches = self.get_fine_matches(self.kp1, self.kp2, self.matches)

    def draw_matches(self):
        self._draw_matches(self.img1, self.img2, self.kp1,
                           self.kp2, self.matches)

    def get_matches(self):
        # return the matching points coordinates in the two images
        src_pts = np.float32(
            [self.kp1[m[0].queryIdx].pt for m in self.matches]).reshape(-1, 1, 2).squeeze()
        dst_pts = np.float32(
            [self.kp2[m[0].trainIdx].pt for m in self.matches]).reshape(-1, 1, 2).squeeze()
        return src_pts, dst_pts

    def get_keypoints(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        # kp: keypoint, des: descriptor
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des

    def get_coarse_matches(self, des1, des2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for first_match, second_match in matches:
            # m is the closest match, n is the second closest match
            if first_match.distance < 0.5 * second_match.distance:
                good.append([first_match])
        return good

    def get_fine_matches(self, kp1, kp2, matches):
        # use RANSAC to remove outliers
        src_pts = np.float32(
            [kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        matches = [m for m, msk in zip(matches, matches_mask) if msk == 1]

        return matches

    def _draw_matches(self, img1, img2, kp1, kp2, matches):
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
        # show the image
        plt.imshow(img3)
        plt.show()


if __name__ == '__main__':
    img1 = cv2.imread('image/r_85.png')
    img2 = cv2.imread('image/r_99.png')
    fm = Feature_matching(img1, img2)
    fm.draw_matches()
    # print(fm.get_matches())
