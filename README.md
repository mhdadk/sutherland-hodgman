# Sutherland–Hodgman
 A NumPy and PyTorch implementation of the Sutherland–Hodgman algorithm for clipping polygons in 2D. The difference between the two implementations is that the PyTorch implementation is [differentiable](https://en.wikipedia.org/wiki/Automatic_differentiation). For example, this allows using the Sutherland-Hodgman algorithm to implement the [Generalized Intersection over Union](https://openaccess.thecvf.com/content_CVPR_2019/html/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.html) metric for the case of more complex bounding polygons. Additionally, either implementation of the Sutherland-Hodgman algorithm, together with the [shoelace algorithm](https://en.wikipedia.org/wiki/Shoelace_formula), can be used to compute the area of intersection of two or more polygons in 2D.

## Usage



## Explanation

The following explanation of the Sutherland-Hodgman algorithm applies to both the NumPy and PyTorch implementations. Given a `N x 2` array containing the vertices of a subject polygon that are [arranged in clockwise order](https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order/1180256):

```python
S = np.array([[x_1,y_1],
              [x_2,y_2],
              ...,
              [x_N,y_N]])
```

and given a `M x 2` array containing the vertices of a clipping polygon that are arranged in clockwise order:

```python
C = np.array([[x_1,y_1],
              [x_2,y_2],
              ...,
              [x_M,y_M]])
```

For example, suppose that the subject polygon `S` and the clipping polygon `C` are:

<img src="figures/step1.svg" style="zoom:125%;" />

<div style="text-align:center;font-weight:bold"">Figure 1</div>

The first step of the Sutherland-Hodgman algorithm is to pick the first two vertices defined in the `C` array. These are the blue points marked as `C_1` and `C_2` respectively  in figure 2.

<img src="figures/step2.svg" style="zoom:125%;" />

<div style="text-align:center;font-weight:bold"">Figure 2</div>

The next step is to pick the first two vertices defined in the `S` array. These are the red points marked as `S_1` and `S_2` respectively in figure 3:

<img src="figures/step3.svg" style="zoom:125%;" />

<div style="text-align:center;font-weight:bold"">Figure 3</div>

We then want to check if the red points are inside the clipping polygon. Since the vertices that define the clipping polygon in the `C` array are arranged in clockwise order, then we can check if the red points are inside the clipping polygon by checking that the red points are "to the right" of the line connecting the blue points. Given any two points `A` and `B`, which are defined by the coordinates `(A_x,A_y)` and `(B_x,B_y)` respectively, and a third point `P` defined by the coordinates `(P_x,P_y)`, which are shown in figure 4,

<img src="figures/check_inside.svg" style="zoom:200%;" />

<div style="text-align:center;font-weight:bold"">Figure 4</div>



we can check if the point `P` is to the right of the line connecting points `A` and `B` by computing:

```
R = (P_x - A_x) * (B_y - A_y) - (P_y - A_y) * (B_x - A_x)
```

If `R < 0`, then the point `P` is to the right of the line connecting points `A` and `B`, and if `R > 0`, then the point `P` is to the left. If `R = 0`, then the point `P` is on the line. For more information about this method, see [this answer](https://math.stackexchange.com/a/274728/652310).

In figure 3, the point `S_1` is to the left of the line connecting points `C_1` and `C_2`, while the point `S_2` is to the right of this line. Since we are performing polygon clipping, we want to discard point `S_1`, save the point of intersection between the line connecting points `S_1` and `S_2` and the line connecting points `C_1` and `C_2`, and save point `S_2`. 
