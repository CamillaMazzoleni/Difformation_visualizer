from utils.math_utils import fexp
import numpy as np

class SuperQuadrics:
    def __init__(self, size, shape, resolution=64):
        self.a1, self.a2, self.a3 = size
        self.e1, self.e2 = shape
        self.N = resolution
        self.x, self.y, self.z, self.eta, self.omega = self.sample_equal_distance_on_sq()

    def sq_surface(self, eta, omega):
        x = self.a1 * fexp(np.cos(eta), self.e1) * fexp(np.cos(omega), self.e2)
        y = self.a2 * fexp(np.cos(eta), self.e1) * fexp(np.sin(omega), self.e2)
        z = self.a3 * fexp(np.sin(eta), self.e1)
        return x, y, z

    def sample_equal_distance_on_sq(self):
        eta = np.linspace(-np.pi / 2, np.pi / 2, self.N)
        omega = np.linspace(-np.pi, np.pi, self.N)
        eta, omega = np.meshgrid(eta, omega)
        x, y, z = self.sq_surface(eta, omega)
        return x, y, z, eta, omega

    def apply_global_linear_tapering(self, ty, tz):
        Y = (((ty / self.a1) * self.x) + 1) * self.y
        Z = (((tz / self.a1) * self.x) + 1) * self.z
        return self.x, Y, Z

    def apply_tapering(self, ty, tz, method="linear"):
        if method == "linear":
            self.x, self.y, self.z = self.apply_global_linear_tapering(ty, tz)
            return self.x, self.y, self.z