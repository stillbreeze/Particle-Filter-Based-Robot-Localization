import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
import pdb

from MapReader import MapReader

class SensorModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        Initialize Sensor Model parameters here
        """
        self.map = occupancy_map
        self.Z_MAX = 8183
        self.P_HIT_SIGMA = 250
        self.P_SHORT_LAMBDA = 0.01
        self.Z_PHIT = 1000
        self.Z_PSHORT = 0.01
        self.Z_PMAX = 0.03
        self.Z_PRAND = 100000

    def bresenham(self, x0, y0, x1, y1):
        """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
        Input coordinates should be integers.
        The result will contain both the start and the end point.
        """
        dx = x1 - x0
        dy = y1 - y0

        xsign = 1 if dx > 0 else -1
        ysign = 1 if dy > 0 else -1

        dx = abs(dx)
        dy = abs(dy)

        if dx > dy:
            xx, xy, yx, yy = xsign, 0, 0, ysign
        else:
            dx, dy = dy, dx
            xx, xy, yx, yy = 0, ysign, xsign, 0

        D = 2*dy - dx
        y = 0

        for x in range(dx + 1):
            yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
            if D >= 0:
                y += 1
                D -= 2*dx
            D += 2*dy

    def ray_trace(self,x_t):
        def bresenham_collide_pt(x1,y1,x2,y2): 
            x1 = min(int(x1), 799)
            y1 = int(y1) 
            x2 = min(int(x2), 799)
            y2 = int(y2)
            if x1<x2:
                final_pts = [x2,y2]
            else:
                final_pts = [x1,y1]
            
            collide_pts = self.bresenham(x1,y1,x2,y2)
            
            for pt in collide_pts:
                if pt[1] >= 800:
                    return final_pts
                if self.occupancy_map[pt[1]][pt[0]] > self.map_threshold:
                    final_pts = [pt[0],pt[1]]
                    return final_pts      
            return final_pts
        def check_intersection(ray_origin, ray_direction, point1, point2):
            point1 = np.array(point1)
            point2 = np.array(point2)
            v1 = ray_origin - point1
            v2 = point2 - point1
            v3 = np.array([-ray_direction[1], ray_direction[0]])
            t1 = np.cross(v2, v1) / np.dot(v2, v3)
            t2 = np.dot(v1, v3) / np.dot(v2, v3)
            if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
                return [ray_origin + t1 * ray_direction]
            return []

        map_size = self.occupancy_map.shape
        pos_x, pos_y, pos_theta = x_t
        pos_x = pos_x/10.
        pos_y = pos_y/10.
        
        ray_direction_x = math.cos(pos_theta)
        ray_direction_y = math.sin(pos_theta)
        ray_direction = np.array([ray_direction_x, ray_direction_y])
        ray_origin = np.array([pos_x, pos_y])
        pts1, pts2, pts3, pts4 = [0,0], [map_size[0], 0], [map_size[0],map_size[1]], [0,map_size[1]]
        for p1, p2 in zip([pts1, pts2, pts3, pts4], [pts2, pts3, pts4, pts1]):
            flag = check_intersection(ray_origin, ray_direction, p1, p2)
            if len(flag) > 0:
                final_x, final_y = flag[0]
                break
        collide_pts = bresenham_collide_pt(pos_x,pos_y,final_x,final_y)
        z_tk_star = np.linalg.norm(collide_pts - ray_origin)
        return z_tk_star + 2.5

    def p_hit(self, z_tk, x_t, z_tk_star):
        if 0 <= z_tk <= self.Z_MAX:
            gaussian = (math.exp(-(z_tk - z_tk_star)**2 / (2 * self.P_HIT_SIGMA**2)))/ math.sqrt(2 * math.pi * self.P_HIT_SIGMA**2)
            return gaussian
        else:
            return 0.0

    def p_short(self, z_tk, x_t, z_tk_star):
        if 0 <= z_tk <= z_tk_star:
            eta = 1 / (1 - math.exp(-self.P_SHORT_LAMBDA * z_tk_star))
            return eta * self.P_SHORT_LAMBDA * math.exp(-self.P_SHORT_LAMBDA * z_tk)
        else:
            return 0.0

    def p_max(self, z_tk, x_t):
        if z_tk == self.Z_MAX:
            return 1.0
        else:
            return 0.0

    def p_rand(self, z_tk, x_t):
        if 0 <= z_tk < self.Z_MAX:
            return 1.0 / self.Z_MAX
        else:
            return 0.0

 
    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        """
        TODO : Add your code here
        """
        pos_x, pos_y, pos_theta = x_t1
        temp = self.map[min(int(pos_y/10.), 799)][min(int(pos_x/10.), 799)]
        if temp > 0.4 or temp == -1:
            return 1e-100
        q = 0.0

        laser_x = 25.0 * np.cos(pos_theta)
        laser_y = 25.0 * np.sin(pos_theta)
        coord_x = int(round((pos_x + laser_x) / 10.0))
        coord_y = int(round((pos_y + laser_y) / 10.0))

        for deg in range (-90,90, 10):
            z_t1_true = self.rayCast(deg, pos_theta, coord_x, coord_y)
            z_t1_k = z_t1_arr[deg+90]
            p1 = self.Z_PHIT * self.p_hit(z_t1_k, x_t1, z_t1_true)
            p2 = self.Z_PSHORT * self.p_short(z_t1_k, x_t1, z_t1_true)
            p3 = self.Z_PMAX * self.p_max(z_t1_k, x_t1)
            p4 = self.Z_PRAND * self.p_rand(z_t1_k, x_t1)
            p = p1 + p2 + p3 + p4
            if p > 0:
                q = q + np.log(p)
        return math.exp(q)


    def rayCast(self, deg, ang, coord_x, coord_y):
        final_angle= ang + math.radians(deg)
        start_x = coord_x
        start_y = coord_y
        final_x = coord_x
        final_y = coord_y
        while 0 < final_x < self.map.shape[1] and 0 < final_y < self.map.shape[0] and abs(self.map[final_y, final_x]) < 0.0000001:
            start_x += 2 * np.cos(final_angle)
            start_y += 2 * np.sin(final_angle)
            final_x = int(round(start_x))
            final_y = int(round(start_y))
        end_p = np.array([final_x,final_y])
        start_p = np.array([coord_x,coord_y])
        dist = np.linalg.norm(end_p-start_p) * 10
        return dist
if __name__=='__main__':
    pass
