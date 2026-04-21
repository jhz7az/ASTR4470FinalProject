import numpy as np


class QuadTreeNode:
    """
    One node of a 2D quadtree for the Barnes-Hut algorithm.

    Each node represents a square region of space.
    A node may either:
    - contain one particle directly, or
    - have four child nodes that subdivide the region.
    """

    def __init__(self, x_center, y_center, half_size):
        """
        Create a quadtree node.

        Parameters
        ----------
        x_center : float
            x-coordinate of the center of this square region.
        y_center : float
            y-coordinate of the center of this square region.
        half_size : float
            Half the width of the square region.
        """

        # Geometric information for this square region
        self.x_center = x_center
        self.y_center = y_center
        self.half_size = half_size

        # Child nodes (NW, NE, SW, SE)
        self.children = []

        # Particle information for a leaf node
        self.particle_index = None

        # Mass and center-of-mass information for this node
        self.mass = 0.0
        self.com_x = 0.0
        self.com_y = 0.0

    def is_leaf(self):
        """
        Return True if this node has no children.
        """
        return len(self.children) == 0

    def contains_point(self, x, y):
        """
        Check whether a point lies inside this node's square region.
        """

        x_min = self.x_center - self.half_size
        x_max = self.x_center + self.half_size
        y_min = self.y_center - self.half_size
        y_max = self.y_center + self.half_size

        return (x_min <= x < x_max) and (y_min <= y < y_max)

    def subdivide(self):
        """
        Split this node into four equal child squares.
        """

        quarter = self.half_size / 2.0

        # Northwest child
        nw = QuadTreeNode(
            self.x_center - quarter,
            self.y_center + quarter,
            quarter
        )

        # Northeast child
        ne = QuadTreeNode(
            self.x_center + quarter,
            self.y_center + quarter,
            quarter
        )

        # Southwest child
        sw = QuadTreeNode(
            self.x_center - quarter,
            self.y_center - quarter,
            quarter
        )

        # Southeast child
        se = QuadTreeNode(
            self.x_center + quarter,
            self.y_center - quarter,
            quarter
        )

        self.children = [nw, ne, sw, se]

    def insert(self, particle_index, positions, masses):
        """
        Insert one particle into the tree.

        Parameters
        ----------
        particle_index : int
            Index of the particle to insert.
        positions : ndarray, shape (N, 2)
            Particle positions.
        masses : ndarray, shape (N,)
            Particle masses.
        """

        x = positions[particle_index, 0]
        y = positions[particle_index, 1]
        m = masses[particle_index]

        # If the point is not inside this node, do nothing
        if not self.contains_point(x, y):
            return False

        # ------------------------------------------
        # Case 1: this node is an empty leaf
        # ------------------------------------------
        if self.is_leaf() and self.particle_index is None:
            self.particle_index = particle_index
            self.mass = m
            self.com_x = x
            self.com_y = y
            return True

        # ------------------------------------------
        # Case 2: this node is a leaf but already
        # contains a particle, so subdivide it
        # ------------------------------------------
        if self.is_leaf():
            old_particle = self.particle_index
            self.particle_index = None

            self.subdivide()

            # Reinsert the old particle into one child
            for child in self.children:
                if child.insert(old_particle, positions, masses):
                    break

        # ------------------------------------------
        # Insert the new particle into one child
        # ------------------------------------------
        inserted = False
        for child in self.children:
            if child.insert(particle_index, positions, masses):
                inserted = True
                break

        # ------------------------------------------
        # Update this node's total mass and center of mass
        # after the insertion
        # ------------------------------------------
        if inserted:
            total_mass = 0.0
            weighted_x = 0.0
            weighted_y = 0.0

            for child in self.children:
                total_mass += child.mass
                weighted_x += child.mass * child.com_x
                weighted_y += child.mass * child.com_y

            self.mass = total_mass

            if total_mass > 0.0:
                self.com_x = weighted_x / total_mass
                self.com_y = weighted_y / total_mass

        return inserted


def build_quadtree(positions, masses, box_size):
    """
    Build a full quadtree containing all particles.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions.
    masses : ndarray, shape (N,)
        Particle masses.
    box_size : float
        Half-size of the root simulation box.

    Returns
    -------
    root : QuadTreeNode
        Root node of the completed quadtree.
    """

    # Root node covers the full simulation square
    root = QuadTreeNode(0.0, 0.0, box_size)

    # Insert every particle into the tree
    for i in range(len(masses)):
        root.insert(i, positions, masses)

    return root