class PositionTracker:
    def __init__(self, start_x=0.0, start_y=0.0, start_phi=0.0):
        self.x = start_x
        self.y = start_y
        self.phi = start_phi
        self.mode = "moving"
        self.odo_x = None
        self.odo_y = None
        self.world_x = start_x
        self.world_y = start_y

    def update(self, x, y, phi):
        if self.odo_x is None:
            self.odo_x = x
            self.odo_y = y
            self.world_x = self.x
            self.world_y = self.y
        self.x = self.world_x + (x - self.odo_x)
        self.y = self.world_y + (y - self.odo_y)
        self.phi = phi
        return self.x, self.y, self.phi

    def set_position(self, x, y, phi=None):
        self.x = x
        self.y = y
        if phi is not None:
            self.phi = phi
        self.odo_x = None
        self.odo_y = None
        self.world_x = x
        self.world_y = y

    def pose_dictionary(self):
        return {"x": self.x, "y": self.y, "phi": self.phi}
