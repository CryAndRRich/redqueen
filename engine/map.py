import numpy as np

dx = [0, 0, -1, 1]
dy = [1, -1, 0, 0]


class Map:
    GRASS = 0
    WALL = 1
    BOX = 2
    ITEM_RADIUS = 3
    ITEM_CAPACITY = 4

    def __init__(self, width=13, height=13, seed=None):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.box_spawn_probability = 0.3
        self.wall_spawn_probability = 0.05
        self.par = np.zeros(self.width * self.height, dtype=np.int32)
        self.grid = np.zeros((height, width), dtype=int)
        self._setup_walls()
        self._generate_map()

    def _setup_walls(self):
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

    def _is_wall(self, row, col):
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return True
        return self.grid[row, col] == self.WALL

    def _convert(self, row, col):
        return (row - 1) * (self.width - 2) + col

    def _ensure_connectivity(self):
        walls_coordinates = []
        self.par = np.zeros(self.width * self.height, dtype=np.int32)

        def find(u):
            if self.par[u] == 0:
                self.par[u] = -1
            if self.par[u] < 0:
                return u
            self.par[u] = find(self.par[u])
            return self.par[u]

        def merge(s, t):
            if self.par[s] < self.par[t]:
                s, t = t, s
            self.par[t] += self.par[s]
            self.par[s] = t

        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if not self._is_wall(row, col):
                    for d in range(4):
                        nrow, ncol = row + dx[d], col + dy[d]
                        if not self._is_wall(nrow, ncol):
                            u, v = self._convert(row, col), self._convert(nrow, ncol)
                            s = find(u)
                            t = find(v)
                            if s != t:
                                merge(s, t)
                else:
                    walls_coordinates.append((row, col))

        self.rng.shuffle(walls_coordinates)

        for row, col in walls_coordinates:
            components = set()
            for d in range(4):
                nrow, ncol = row + dx[d], col + dy[d]
                if not self._is_wall(nrow, ncol):
                    components.add(find(self._convert(nrow, ncol)))
            if len(components) > 1:
                self.grid[row, col] = self.GRASS
                new_cell = self._convert(row, col)
                for u in components:
                    s, t = u, find(new_cell)
                    if s != t:
                        merge(s, t)

        while True:
            components = set()
            for row in range(1, self.height - 1):
                for col in range(1, self.width - 1):
                    if self.grid[row, col] != self.WALL:
                        components.add(find(self._convert(row, col)))
            if len(components) == 1:
                break
            for row in range(1, self.height - 1):
                for col in range(1, self.width - 1):
                    if self.grid[row, col] == self.WALL:
                        if self.rng.random() < 0.2:
                            self.grid[row, col] = self.GRASS
                            for d in range(4):
                                nrow, ncol = row + dx[d], col + dy[d]
                                if not self._is_wall(nrow, ncol):
                                    u, v = self._convert(row, col), self._convert(nrow, ncol)
                                    s = find(u)
                                    t = find(v)
                                    if s != t:
                                        merge(s, t)

    def _generate_map(self):
        self.grid[2:self.height - 1:2, 2:self.width - 1:2] = self.WALL
        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if self.grid[row, col] == 0 and self.rng.random() < self.wall_spawn_probability:
                    self.grid[row, col] = self.WALL

        for row in range(1, self.height - 2):
            for col in range(1, self.width - 2):
                if self.grid[row:row + 2, col:col + 2].tolist() == [[self.WALL, self.WALL], [self.WALL, self.WALL]]:
                    self.grid[row + self.rng.integers(0, 2), col + self.rng.integers(0, 2)] = self.GRASS

        self.grid[1:3, 1:3] = self.GRASS
        self.grid[1:3, self.width - 3:self.width - 1] = self.GRASS
        self.grid[self.height - 3:self.height - 1, 1:3] = self.GRASS
        self.grid[self.height - 3:self.height - 1, self.width - 3:self.width - 1] = self.GRASS

        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if col < self.width - 4 and self.grid[row, col:col + 4].tolist() == [self.WALL, self.WALL, self.WALL, self.WALL]:
                    self.grid[row, col + self.rng.integers(0, 4)] = self.GRASS
                if row < self.height - 4 and self.grid[row:row + 4, col].flatten().tolist() == [self.WALL, self.WALL, self.WALL, self.WALL]:
                    self.grid[row + self.rng.integers(0, 4), col] = self.GRASS

        self._ensure_connectivity()

        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if self.grid[row, col] == self.GRASS and self.rng.random() < self.box_spawn_probability:
                    self.grid[row, col] = self.BOX

        self.grid[1:3, 1:3] = self.GRASS
        self.grid[1:3, self.width - 3:self.width - 1] = self.GRASS
        self.grid[self.height - 3:self.height - 1, 1:3] = self.GRASS
        self.grid[self.height - 3:self.height - 1, self.width - 3:self.width - 1] = self.GRASS
