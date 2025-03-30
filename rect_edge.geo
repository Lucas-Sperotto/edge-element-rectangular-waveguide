
    lc = 0.00025;
    Point(1) = {0, 0, 0, lc};
    Point(2) = {0.01, 0, 0, lc};
    Point(3) = {0.01, 0.005, 0, lc};
    Point(4) = {0, 0.005, 0, lc};
    Line(1) = {1, 2};
    Line(2) = {2, 3};
    Line(3) = {3, 4};
    Line(4) = {4, 1};
    Line Loop(5) = {1, 2, 3, 4};
    Plane Surface(6) = {5};
    Physical Surface(200) = {6};
    Mesh 2;
    Save "rect_edge.msh";
    