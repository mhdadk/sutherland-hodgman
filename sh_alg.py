import numpy as np

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

def is_inside(p1,p2,q):
    R = (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])
    if R < 0:
        return True
    else:
        return False

def compute_intersection(p1,p2,p3,p4):
    
    # if this is really small, then the line segments are almost coincident
    den = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    
    num_x = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0])
           - (p1[0] * p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0]))
    
    num_y = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1])
           - (p1[1] * p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0]))
    
    intersection = (num_x/den,num_y/den)
    
    return intersection

def clip(subject_polygon,clipping_polygon):
    
    final_polygon = subject_polygon.copy()
    
    for i in range(len(clipping_polygon)):
        
        # stores the vertices of the next iteration of the clipping procedure
        next_polygon = final_polygon.copy()
        
        # stores the vertices of the final clipped polygon
        final_polygon = []
        
        # these two vertices define a line segment (edge) in the clipping
        # polygon. It is assumed that indices wrap around, such that if
        # i = 1, then i - 1 = K.
        c_vertex1 = clipping_polygon[i]
        c_vertex2 = clipping_polygon[i - 1]
        
        for j in range(len(subject_polygon)):
            
            # these two vertices define a line segment (edge) in the subject
            # polygon
            s_vertex1 = next_polygon[j]
            s_vertex2 = next_polygon[j - 1]
            
            if is_inside(c_vertex1,c_vertex2,s_vertex2):
                if not is_inside(c_vertex1,c_vertex2,s_vertex1):
                    intersection = compute_intersection(s_vertex1,s_vertex2,c_vertex1,c_vertex2)
                    final_polygon.append(intersection)
                final_polygon.append(s_vertex2)
            elif is_inside(c_vertex1,c_vertex2,s_vertex1):
                intersection = compute_intersection(s_vertex1,s_vertex2,c_vertex1,c_vertex2)
                final_polygon.append(intersection)
    
    return final_polygon
