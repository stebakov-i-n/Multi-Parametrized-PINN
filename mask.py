import numpy as np
import torch
from stl import mesh
import cv2
from PIL import Image, ImageOps
from geomdl import BSpline, utilities
from matplotlib import path


def get_point_from_segment(points, segment, n, x3=None):
    delta = segment[1] - segment[0]
    if torch.rand(1) > 0.5:
        n = int(n) + 1
    else:
        n = int(n)
    if n > 0:
        if x3 is not None:
            points.append(torch.cat((segment[0].unsqueeze(0).repeat(n, 1) + delta.unsqueeze(0).repeat(n, 1) * torch.rand(n).unsqueeze(1).to(delta.device),
                                         x3.reshape(1).unsqueeze(0).repeat(n, 1).to(delta.device))))
        else:
            points.append(segment[0].unsqueeze(0).repeat(n, 1) + delta.unsqueeze(0).repeat(n, 1) * torch.rand(n).unsqueeze(1).to(delta.device))


def sample_boundary_points(segments, m_all, x3=None):
    dist_all = 0
    for i in segments:
        tmp = torch.stack(i, axis=0)
        dist_all += torch.sum(torch.sum((tmp[:, 0] - tmp[:, 1]) ** 2, axis=1) ** 0.5)

    walls_points = []

    for i in range(len(segments)):
        tmp = torch.stack(segments[i], axis=0)
        dist = torch.sum((tmp[:, 0] - tmp[:, 1]) ** 2, axis=1) ** 0.5
        for j in range(len(segments[i])):
            m = dist[j] / (dist_all / m_all)
            get_point_from_segment(walls_points, segments[i][j], m, x3[i].reshape(1) if x3 is not None else x3)

    x = torch.cat(walls_points)
    return x


def is_inside(triangles, X, buffer=False):
    """Copyright 2018 Alexandre Devert

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software."""
    
    # Вычисление определителя 3x3 вдоль оси 1
    def adet(X, Y, Z):
        ret  = (X[:,0] * Y[:,1] * Z[:,2] + Y[:,0] * Z[:,1] * X[:,2] + Z[:,0] * X[:,1] * Y[:,2] - 
                Z[:,0] * Y[:,1] * X[:,2] - Y[:,0] * X[:,1] * Z[:,2] - X[:,0] * Z[:,1] * Y[:,2])
        return ret

    # Инициализация обобщенного порядка точки
    ret = torch.zeros(X.shape[0], dtype=X.dtype).to(X.device)
    
    # Накопление обобщенного порядок точки для каждого треугольника
    for U, V, W in triangles:
        A, B, C = U - X, V - X, W - X
        omega = adet(A, B, C)

        a, b, c = torch.norm(A, dim=1), torch.norm(B, dim=1), torch.norm(C, dim=1)
        k  = a * b * c + c * torch.sum(A * B, dim=1) + a * torch.sum(B * C, dim=1) + b * torch.sum(C * A,dim=1)
        
        ret += torch.arctan2(omega, k)

    return ret >= 2 * np.pi - (buffer if buffer else 0.)


def points_on_triangle(triangle, m):
    p = m % 1
    m = int(np.floor(m)) + (1 if np.random.random() < p else 0)
    x, y = torch.rand(m), torch.rand(m)
    q = abs(x - y)
    s, t, u = q, 0.5 * (x + y - q), 1 - 0.5 * (q + x + y)
    return torch.stack((s * triangle[0] + t * triangle[3] + u * triangle[6],
                        s * triangle[1] + t * triangle[4] + u * triangle[7],
                        s * triangle[2] + t * triangle[5] + u * triangle[8]), 1)


def sample_boundary_points_from_stl(path, centering, m_all):
    mesh_ = mesh.Mesh.from_file(path)

    points = torch.tensor(np.array(mesh_.points)[~np.isclose(mesh_.normals[:, 0], 0., rtol=1e-07, atol=1e-10) & (np.isclose(mesh_.normals[:, 1], 0., rtol=1e-07, atol=1e-10))])

    points[:, :3] -= centering.cpu().numpy()
    points[:, 3:6] -= centering.cpu().numpy()
    points[:, 6:9] -= centering.cpu().numpy()

    areas = torch.tensor(np.array(mesh_.areas)[~np.isclose(mesh_.normals[:, 0], 0., rtol=1e-07, atol=1e-10) & (np.isclose(mesh_.normals[:, 1], 0., rtol=1e-07, atol=1e-10))])

    areas_all = areas.sum()

    boundary_points = torch.zeros(0, 3)

    for i in range(len(points)):
        m = areas[i] / (areas_all / m_all)

        boundary_points = torch.concatenate((boundary_points,
                                             points_on_triangle(points[i], m)))

    x = boundary_points
    return x


def load_stl(path, n, n_interior, n_walls, n_inlet, n_outlet, n_q_fix=0, length=[1., 1., 1.], device='cpu', use_3d=False, inside_buffer=False):
    print(f'Mask generation with path: {path}')
    closed_mesh = mesh.Mesh.from_file(path)
    
    centering = torch.zeros(3).to(device)

    closed_points = torch.tensor(np.array(closed_mesh.points)).to(device)

    centering[0] = closed_points[:, ::3].min() + (closed_points[:, ::3].max() - closed_points[:, ::3].min()) / 2
    centering[1] = closed_points[:, 1::3].min() + (closed_points[:, 1::3].max() - closed_points[:, 1::3].min()) / 2
    centering[2] = closed_points[:, 2::3].min() + (closed_points[:, 2::3].max() - closed_points[:, 2::3].min()) / 2

    closed_points[:, :3] -= centering
    closed_points[:, 3:6] -= centering
    closed_points[:, 6:9] -= centering

    x1 = torch.linspace(-length[0] / 2, length[0] / 2, n)
    x2 = torch.linspace(-length[1] / 2, length[1] / 2, n) if use_3d else torch.tensor(0.001 * length[1])
    x3 = torch.linspace(-length[2] / 2, length[2] / 2, n)

    x1, x2, x3 = torch.meshgrid(x1, x2, x3, indexing='ij')

    dx = torch.tensor([closed_points[:, ::3].max() - closed_points[:, ::3].min(),
                       closed_points[:, 1::3].max() - closed_points[:, 1::3].min(),
                       closed_points[:, 2::3].max() - closed_points[:, 2::3].min()]).to(device)
    
    x = torch.stack([x1, x2, x3])
    x = x.reshape(3, -1).T.to(device)

    mask = is_inside(zip(closed_points[:, :3], 
                         closed_points[:, 3:6],
                         closed_points[:, 6:9]), x, inside_buffer)
    mask = mask.reshape(n, n, n).cpu() if use_3d else mask.reshape(n, n).cpu()
    
    print('done\n\nInterior points generation')
    x = (torch.rand(int(0.2 * n_interior), 3) * dx.cpu() * 1.1 - (dx.cpu() * 1.1 / 2)).to(device)
    mask_ = is_inside(zip(closed_points[:, :3], 
                         closed_points[:, 3:6],
                         closed_points[:, 6:9]), x, inside_buffer)
    
    x = x[mask_]
    x = x.repeat(int(n_interior * 1.3 / len(x)), 1)
    x = x + ((torch.rand(*x.shape).to(device) - 0.5) * dx * 0.05)

    mask_ = is_inside(zip(closed_points[:, :3], 
                         closed_points[:, 3:6],
                         closed_points[:, 6:9]), x, inside_buffer)
    
    x_interior = x[mask_].cpu()
    x_interior = x_interior[torch.randperm(len(x_interior))[:n_interior]]

    closed_mesh = None
    closed_points = None
    x1 = None
    x2 = None
    x3 = None
    dx = None
    x = None
    mask_ = None

    mask = {'num': mask.float(), 'bool': mask}

    print('done\n\nWalls points generation')
    x_walls = sample_boundary_points_from_stl(path, centering, int(n_walls * 1.1))
    print('done\n\nInlet points generation')
    x_inlet = sample_boundary_points([[torch.tensor([[0.0005, -0.005], [-0.0005, -0.005]])]], n_inlet)
    # x_inlet = sample_boundary_points_from_stl(path + '/inlet.stl', centering, int(n_inlet * 1.1))
    print('done\n\nOutlet points generation')
    x_outlet = sample_boundary_points([[torch.tensor([[0.0005, 0.005], [-0.0005, 0.005]])]], n_outlet)
    # x_outlet = sample_boundary_points_from_stl(path + '/outlet.stl', centering, int(n_outlet * 1.1))
    print('done\n\n')
    # print('done\n\nQ_fix points generation')
    # x_q_fix = sample_boundary_points_from_stl(path + '/q_fix.stl', centering, int(n_q_fix * 1.1))
    # print('done')
    if not use_3d:
        x_interior = x_interior[:, ::2]
        x_walls = x_walls[:, ::2]
        # x_inlet = x_inlet[:, ::2]
        # x_outlet = x_outlet[:, ::2]
        # x_q_fix = x_q_fix[:, ::2]

    x_walls = x_walls[torch.randperm(len(x_walls))[:n_walls]]
    # x_inlet = x_inlet[torch.randperm(len(x_inlet))[:n_inlet]]
    # x_outlet = x_outlet[torch.randperm(len(x_outlet))[:n_outlet]]
    # x_q_fix = x_q_fix[torch.randperm(len(x_q_fix))[:n_q_fix]]

    return mask, x_interior, x_walls, x_inlet, x_outlet, 0, 0.0005


# 
def img_to_mask(path, n, n_interior, n_walls, n_inlet, n_outlet, n_q_fix=0, length=[1., 1., 1.], device='cpu', use_3d=False, inside_buffer=False, smooth_degree=1, smooth_sample_size=2):
    print(f'Mask generation with path: {path}')
    mask = Image.open(path)
    mask = ImageOps.grayscale(mask)
    mask = np.array(mask).astype(np.float32) / 255

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    segments, contours, orig_segments, s, input_segments, output_segments = smooth_segments(contours, mask.shape[0], length[0], length[-1],
                                                           device, use_3d=False, degree=smooth_degree,
                                                           sample_size=smooth_sample_size)

    s *= length[1]

    print('done\n\nInterior points generation')

    mask = find_mask(contours, n, length[2], length[0])[0].to(device)

    x_interior = sample_inside_points(contours, n_interior * 5, length[0], length[2], s=None)[0].to(device)

    mask = {'num': mask.to(device), 'bool': (mask == 1).to(device)}
    print('done\n\nWalls points generation')

    x_walls = sample_boundary_points([orig_segments], n_walls * 1.3).to(device)
    print('done\n\nInlet points generation')

    x_inlet = sample_boundary_points([input_segments], n_inlet * 1.3).to(device)
    print('done\n\nOutlet points generation')

    x_outlet = sample_boundary_points([output_segments], n_outlet * 1.3).to(device)
    print('done\n\n')

    tmp = []
    for seg in input_segments:
        tmp.append(seg[0][0])
        tmp.append(seg[1][0])

    center = (max(tmp) + min(tmp)) / 2
    h = (max(tmp) - min(tmp)) / 2

    return mask, x_interior, x_walls, x_inlet, x_outlet, center.item(), h.item()


def smooth_segments(contours, n, l1, l2, device, use_3d=False, calc_s=False, degree=3, sample_size=2):
    new_segments = []
    new_contours = []
    orig_segments = []
    input_segments = []
    output_segments = []
    orig_contours = []
    for i in contours:
        i = np.concatenate((i[i[:, 0, 0].argmin() + 1:, 0], i[:i[:, 0, 0].argmin() + 2, 0]))
        start = 0
        if not use_3d:
            for k in range(len(i)):
                if i[k, 0] == i[0 if (k == len(i) - 1) else k + 1, 0] and (i[k, 0] == 0 or i[k, 0] == (n - 1)):
                    contour = i.copy()[start:k + 1]
                    contour = (contour / (n - 1))
                    contour[:, 0] = contour[:, 0] * l2 - l2 / 2
                    contour[:, 1] = contour[:, 1] * l1 - l1 / 2
                    contour = contour.tolist()
                    start = k + 1
                    if len(contour) < 2:
                        continue
                    if degree == 1:
                        evalpts = contour
                    else:
                        try:
                            evalpts = spline_curve(contour, degree, sample_size)
                        except BSpline.GeomdlException:
                            evalpts = contour

                    tmp_segments = ([torch.tensor([evalpts[j], evalpts[j + 1]]).to(device).flip(1)
                                     for j in range(len(evalpts) - 1)])
                    orig_segments += tmp_segments
                    if k == (len(i) - 2):
                        orig_contours[-1] += [i[:] for i in evalpts]
                    else:
                        orig_contours.append([i[:] for i in evalpts])

                    tmp_segments = ([torch.tensor([evalpts[j], evalpts[j + 1]]).to(device).flip(1)
                                     for j in range(len(evalpts) - 1)])
                    new_segments += tmp_segments
                    if k == (len(i) - 2):
                        new_contours[-1] += evalpts
                    else:
                        new_contours.append(evalpts)
                #  or i[k, 0] == (n - 1)):
                    if i[k, 0] == 0:
                        contour = i.copy()[k:k + 2]
                        contour = (contour / (n - 1))
                        contour[:, 0] = contour[:, 0] * l2 - l2 / 2
                        contour[:, 1] = contour[:, 1] * l1 - l1 / 2
                        contour = contour.tolist()

                        tmp_segments = ([torch.tensor([contour[j], contour[j + 1]]).to(device).flip(1)
                                        for j in range(len(contour) - 1)])
                        input_segments += tmp_segments
                    elif i[k, 0] == (n - 1):
                        contour = i.copy()[k:k + 2]
                        contour = (contour / (n - 1))
                        contour[:, 0] = contour[:, 0] * l2 - l2 / 2
                        contour[:, 1] = contour[:, 1] * l1 - l1 / 2
                        contour = contour.tolist()

                        tmp_segments = ([torch.tensor([contour[j], contour[j + 1]]).to(device).flip(1)
                                        for j in range(len(contour) - 1)])
                        output_segments += tmp_segments

        if not start:
            contour = i.copy()
            contour = (contour / (n - 1))
            contour[:, 0] = contour[:, 0] * l2 - l2 / 2
            contour[:, 1] = contour[:, 1] * l1 - l1 / 2
            contour = contour.tolist()
            contour.append(contour[0])
            if degree == 1:
                    evalpts = contour
            else:
                try:
                    evalpts = spline_curve(contour, degree, sample_size)
                except BSpline.GeomdlException:
                    evalpts = contour
            new_segments += ([torch.tensor([evalpts[j], evalpts[j + 1]]).to(device).flip(1)
                              for j in range(len(evalpts) - 1)])
            orig_segments += ([torch.tensor([evalpts[j], evalpts[j + 1]]).to(device).flip(1)
                               for j in range(len(evalpts) - 1)])
            orig_contours.append(evalpts)
            new_contours.append(evalpts)

    if not use_3d:
        s = 0.
        for contour in orig_contours:
            contour = np.array(contour)
            tmp = contour[contour[:, 0].argsort()][-2:]
            if tmp[0, 0] > ((l2 / 2) - (l2 / (2 * n))) and tmp[1, 0] > ((l2 / 2) - (l2 / (2 * n))):
                s += abs(tmp[0, 1] - tmp[1, 1])
    elif calc_s:
        s = []
        for contour in orig_contours:
            contour = np.array(contour)
            s.append(polygon_area(contour[:, 0], contour[:, 1]))
    else:
        s = None
    return new_segments, new_contours, orig_segments, s, input_segments, output_segments

def find_mask(contours, n, l1, l2, s=None):
    mask = np.zeros((n, n)).astype(np.bool_)
    x1 = np.linspace(-l1 / 2, l1 / 2, n)
    x2 = np.linspace(-l2 / 2, l2 / 2, n)
    x1, x2 = np.meshgrid(x1, x2)
    if s is not None:
        s.append(0.)
    for i in range(len(contours)):
        p = path.Path(contours[i])
        if s is not None:
            mask_sum = mask.sum()
        mask = np.logical_xor(p.contains_points(
            np.hstack((x1.flatten()[:, np.newaxis],
                       x2.flatten()[:, np.newaxis]))).reshape(mask.shape), mask)
        if s is not None:
            mask_sum_after = mask.sum()
            if mask_sum_after > mask_sum:
                s[-1] += s[i]
            else:
                s[-1] -= s[i]
    if s is not None:
        s = s[-1]
    return torch.tensor(mask).float(), s

def polygon_area(x, y):
    # Initialize area
    area = 0.0
    n = len(x)

    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0, n):
        area += (x[j] + x[i]) * (y[j] - y[i])
        j = i  # j is previous vertex to i

    # Return absolute value
    return abs(area / 2.0)


def sample_inside_points(contours, n, l1, l2, s=None):
    mask = np.zeros((n)).astype(np.bool_)

    points = np.random.rand(n, 2).astype(np.float32) * 2 - 1
    points[:, 0] *= (l1 * 0.5)
    points[:, 1] *= (l2 * 0.5)

    x1 = points[:, 0]
    x2 = points[:, 1]
   
    if s is not None:
        s.append(0.)
    for i in range(len(contours)):
        p = path.Path(contours[i])
        if s is not None:
            mask_sum = mask.sum()
        mask = np.logical_xor(p.contains_points(
            np.hstack((x2.flatten()[:, np.newaxis],
                       x1.flatten()[:, np.newaxis]))).reshape(mask.shape), mask)
        if s is not None:
            mask_sum_after = mask.sum()
            if mask_sum_after > mask_sum:
                s[-1] += s[i]
            else:
                s[-1] -= s[i]
    if s is not None:
        s = s[-1]
    return torch.tensor(points[mask]), s


def spline_curve(ctrlpts, degree=1, sample_size=1):
    # Create a curve instance
    curve = BSpline.Curve()
    # Set degree
    curve.degree = degree
    # Set control points
    curve.ctrlpts = ctrlpts
    # Auto−generate knot vectors
    curve.knotvector = utilities.generate_knot_vector(curve.degree, num_ctrlpts=len(curve.ctrlpts))
    # Set sample size
    curve.sample_size = len(curve.ctrlpts) * sample_size
    # Evaluate curve
    curve.evaluate()
    return curve.evalpts
