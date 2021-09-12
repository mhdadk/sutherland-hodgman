"""
Given a subject polygon defined by the vertices in clockwise order

subject_polygon = [(x_1,y_1),(x_2,y_2),...,(x_N,y_N)]

and a clipping polygon, which will be used to clip the subject polygon,
defined by the vertices in clockwise order

clipping_polygon = [(x_1,y_1),(x_2,y_2),...,(x_K,y_K)]

and assuming that the subject polygon and clipping polygon overlap,
the Sutherland-Hodgman algorithm works as follows:

for i = 1 to K:
    
    # this will  store the vertices of the final clipped polygon
    final_polygon = []
    
    # these two vertices define a line segment (edge) in the clipping
    # polygon. It is assumed that indices wrap around, such that if
    # i = 1, then i - 1 = K.
    c_vertex1 = clipping_polygon[i]
    c_vertex2 = clipping_polygon[i - 1]
    
    for j = 1 to N:
        
        # these two vertices define a line segment (edge) in the subject
        # polygon. It is assumed that indices wrap around, such that if
        # j = 1, then j - 1 = N.
        s_vertex1 = subject_polygon[j]
        s_vertex2 = subject_polygon[j - 1]
        
        # next, we want to check if the points s_vertex1 and s_vertex2 are
        # inside the clipping polygon. Since the points that define the
        # edges of the clipping polygon are listed in clockwise order in
        # clipping_polygon, then we can do this by checking if s_vertex1
        # and s_vertex2 are to the right of the line segment defined by
        # the points (c_vertex1,c_vertex2).
        #
        # if both s_vertex1 and s_vertex2 are inside the clipping polygon,
        # then s_vertex2 is added to the final_polygon list.
        # 
        # if s_vertex1 is outside the clipping polygon and s_vertex2 is
        # inside the clipping polygon, then we first add the point of
        # intersection between the edge defined by (s_vertex1,s_vertex2)
        # and the edge defined by (c_vertex1,c_vertex2) to final_polygon,
        # and then we add s_vertex2 to final_polygon.
        # 
        # if s_vertex1 is inside the clipping polygon and s_vertex2 is
        # outside the clipping polygon, then we add the point of
        # intersection between the edge defined by (s_vertex1,s_vertex2)
        # and the edge defined by (c_vertex1,c_vertex2) to final_polygon.
        #
        # if both s_vertex1 and s_vertex2 are outside the clipping polygon,
        # then neither are added to final_polygon.
        #
        # note that since we only compute the point of intersection if
        # we know that the edge of the clipping polygon and the edge of
        # the subject polygon intersect, then we can treat them as infinite
        # lines and use the formula given here:
        #
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
        #
        # to compute the point of intersection.

"""

import numpy as np
import warnings

# POINTS NEED TO BE PRESENTED CLOCKWISE OR ELSE THIS WONT WORK

class PolygonClipper:
    
    def __init__(self,warn_if_empty=True):
        self.warn_if_empty = warn_if_empty
    
    def is_inside(self,p1,p2,q):
        R = (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])
        if R <= 0:
            return True
        else:
            return False

    def compute_intersection(self,p1,p2,p3,p4):
        
        """
        given points p1 and p2 on line L1, compute the equation of L1 in the
        format of y = m1 * x + b1. Also, given points p3 and p4 on line L2,
        compute the equation of L2 in the format of y = m2 * x + b2.
        
        To compute the point of intersection of the two lines, equate
        the two line equations together
        
        m1 * x + b1 = m2 * x + b2
        
        and solve for x. Once x is obtained, substitute it into one of the
        equations to obtain the value of y.
        
        if one of the lines is vertical, then the x-coordinate of the point of
        intersection will be the x-coordinate of the vertical line. Note that
        there is no need to check if both lines are vertical (parallel), since
        this function is only called if we know that the lines intersect.
        """
        
        # if first line is vertical
        if p2[0] - p1[0] == 0:
            x = p1[0]
            
            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            b2 = p3[1] - m2 * p3[0]
            
            # y-coordinate of intersection
            y = m2 * x + b2
        
        # if second line is vertical
        elif p4[0] - p3[0] == 0:
            x = p3[0]
            
            # slope and intercept of first line
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b1 = p1[1] - m1 * p1[0]
            
            # y-coordinate of intersection
            y = m1 * x + b1
        
        # if neither line is vertical
        else:
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b1 = p1[1] - m1 * p1[0]
            
            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            b2 = p3[1] - m2 * p3[0]
        
            # x-coordinate of intersection
            x = (b2 - b1) / (m1 - m2)
        
            # y-coordinate of intersection
            y = m1 * x + b1
        
        intersection = (x,y)
        
        return intersection
    
    def clip(self,subject_polygon,clipping_polygon):
        
        final_polygon = subject_polygon.copy()
        
        for i in range(len(clipping_polygon)):
            
            # stores the vertices of the next iteration of the clipping procedure
            next_polygon = final_polygon.copy()
            
            # stores the vertices of the final clipped polygon
            final_polygon = []
            
            # these two vertices define a line segment (edge) in the clipping
            # polygon. It is assumed that indices wrap around, such that if
            # i = 1, then i - 1 = K.
            c_edge_start = clipping_polygon[i - 1]
            c_edge_end = clipping_polygon[i]
            
            for j in range(len(next_polygon)):
                
                # these two vertices define a line segment (edge) in the subject
                # polygon
                s_edge_start = next_polygon[j - 1]
                s_edge_end = next_polygon[j]
                
                if self.is_inside(c_edge_start,c_edge_end,s_edge_end):
                    if not self.is_inside(c_edge_start,c_edge_end,s_edge_start):
                        intersection = self.compute_intersection(s_edge_start,s_edge_end,c_edge_start,c_edge_end)
                        final_polygon.append(intersection)
                    final_polygon.append(tuple(s_edge_end))
                elif self.is_inside(c_edge_start,c_edge_end,s_edge_start):
                    intersection = self.compute_intersection(s_edge_start,s_edge_end,c_edge_start,c_edge_end)
                    final_polygon.append(intersection)
        
        return np.asarray(final_polygon)
    
    def __call__(self,A,B):
        clipped_polygon = self.clip(A,B)
        if len(clipped_polygon) == 0 and self.warn_if_empty:
            warnings.warn("No intersections found. Are you sure your \
                          polygon coordinates are in clockwise order?")
        
        return clipped_polygon

if __name__ == '__main__':
    
    # some test polygons
    
    clip = PolygonClipper()
    
    # squares
    # subject_polygon = [(-1,1),(1,1),(1,-1),(-1,-1)]
    # clipping_polygon = [(0,0),(0,2),(2,2),(2,0)]
    
    # squares: different order of points
    # subject_polygon = [(-1,-1),(-1,1),(1,1),(1,-1)]
    # clipping_polygon = [(2,0),(0,0),(0,2),(2,2)]
    
    # triangles
    # subject_polygon = [(0,0),(2,1),(2,0)]
    # clipping_polygon = [(1,0.5),(3,1.5),(3,0.5)]
    
    # star and square
    subject_polygon = [(0,3),(0.5,0.5),(3,0),(0.5,-0.5),(0,-3),(-0.5,-0.5),(-3,0),(-0.5,0.5)]
    clipping_polygon = [(-2,-2),(-2,2),(2,2),(2,-2)]
    
    # star and triangle
    # subject_polygon = [(0,3),(0.5,0.5),(3,0),(0.5,-0.5),(0,-3),(-0.5,-0.5),(-3,0),(-0.5,0.5)]
    # clipping_polygon = [(0,2),(2,-2),(-2,-2)]
    
    subject_polygon = np.array(subject_polygon)
    clipping_polygon = np.array(clipping_polygon)
    clipped_polygon = clip(subject_polygon,clipping_polygon)
    