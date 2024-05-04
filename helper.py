import time
from IPython.display import Image, display
import PIL.Image
import matplotlib.image as mpimg
import scipy.ndimage
import cv2 # For Sobel etc
import glob
import matplotlib.pylab as plt
import numpy as np
np.set_printoptions(suppress=True, linewidth=200) 
plt.rcParams['image.cmap'] = 'jet'

import numpy as np
from matplotlib.pyplot import imshow, show
import matplotlib.patches as patches

def compute_gradients(images):
  
    grad_mags = []
    grad_phases = []

    for img in images:
        
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

       
        grad_mag = np.sqrt(gx**2 + gy**2).astype(np.float32)  
        grad_phase = np.arctan2(gy, gx)  

        
        gradient_mask_threshold = 2 * np.mean(grad_mag.flatten())
        grad_phase_masked = np.where(grad_mag > gradient_mask_threshold, grad_phase, np.nan)

      
        grad_mags.append(grad_mag)
        grad_phases.append(grad_phase_masked)

    return grad_mags, grad_phases

def getSaddle(gray_img, visualize=False):

    if not isinstance(gray_img, np.ndarray) or len(gray_img.shape) != 2:
        raise ValueError("Input must be a single-channel grayscale image.")

    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)

    S = gxx * gyy - gxy**2

    if visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(S, cmap='jet')
        plt.colorbar()
        plt.title("Saddle Points")
        plt.show()

    return S

def nonmax_sup(img, win=10):
    
    w, h = img.shape
    img_sup = np.zeros_like(img, dtype=np.float64)  

    
    for i, j in np.argwhere(img):
        
        ta = max(0, i - win)
        tb = min(w, i + win + 1)
        tc = max(0, j - win)
        td = min(h, j + win + 1)
        cell = img[ta:tb, tc:td]  
        val = img[i, j] 

        
        if (cell.max() == val and np.sum(cell.max() == cell) == 1):
            img_sup[i, j] = val 

    return img_sup

def pruneSaddle(s, max_features=10000, initial_thresh=128):
    
    thresh = initial_thresh
    score = (s > 0).sum()  

    while score > max_features:
        thresh *= 2 
        s[s < thresh] = 0  
        score = (s > 0).sum()  



def process_saddles(images):
    
    saddles = []
    for img in images:
        saddle = getSaddle(img)
        saddle = np.maximum(-saddle, 0)  

        pruneSaddle(saddle)  

        saddles.append(saddle)

        
        plt.figure(figsize=(6, 6))
        plt.imshow(saddle, cmap='gray')
        plt.title("Pruned Saddle Points")
        plt.colorbar()
        plt.show()

    return saddles

def simplifyContours(contours):
   
    for i in range(len(contours)):
        epsilon = 0.04 * cv2.arcLength(contours[i], True)
        contours[i] = cv2.approxPolyDP(contours[i], epsilon, True)

def getAngle(a, b, c):
   

    k = (a*a + b*b - c*c) / (2 * a * b)
    k = max(min(k, 1), -1)
    return np.arccos(k) * (180.0 / np.pi)

def is_square(cnt, eps=3.0, xratio_thresh=0.5):
    

    dd = np.sqrt(np.sum(np.square(np.diff(cnt[:, 0, :], axis=0, append=cnt[:1, 0, :])), axis=1))
    xa = np.linalg.norm(cnt[0, 0, :] - cnt[2, 0, :])
    xb = np.linalg.norm(cnt[1, 0, :] - cnt[3, 0, :])
    xratio = min(xa, xb) / max(xa, xb)


    angles = np.array([getAngle(dd[i], dd[(i+1) % 4], xb if i % 2 == 0 else xa) for i in range(4)])
    good_angles = np.all((angles > 40) & (angles < 140))

    
    side_ratios = np.array([max(dd[i] / dd[(i+1) % 4], dd[(i+1) % 4] / dd[i]) for i in range(4)])
    good_side_ratios = np.all(side_ratios < eps)

    return good_side_ratios and good_angles and xratio > xratio_thresh

def getContourVals(cnt, img):
    
    cimg = np.zeros_like(img)
    cv2.drawContours(cimg, [cnt], 0, color=255, thickness=-1)
    return img[cimg == 255]

def pruneContours(contours, hierarchy, saddle):
    
    new_contours, new_hierarchies = [], []
    for i, cnt in enumerate(contours):
        h = hierarchy[i]
        if h[2] != -1 or len(cnt) != 4 or cv2.contourArea(cnt) < 64 or not is_square(cnt):
            continue
        cnt = updateCorners(cnt, saddle)
        if len(cnt) != 4:
            continue
        new_contours.append(cnt)
        new_hierarchies.append(h)

    if not new_contours:
        return np.array([]), np.array([])

    areas = [cv2.contourArea(c) for c in new_contours]
    median_area = np.median(areas)
    filtered_contours = [c for c, a in zip(new_contours, areas) if median_area * 0.25 <= a <= median_area * 2.0]
    return np.array(filtered_contours), np.array(new_hierarchies)

def getContours(img, edges, iters=10):
    

    if not isinstance(edges, np.ndarray):
        raise ValueError("Edges input must be a numpy array.")

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

   
    contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    
    simplifyContours(contours)

   
    if hierarchy is None or len(hierarchy) == 0:
        raise ValueError("No hierarchy data found, check the input edges for adequate feature details.")

    return np.array(contours), hierarchy[0]

def getContours(img, edges):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    
    contours = list(contours)
    simplifyContours(contours)

    return contours, hierarchy[0]

def updateCorners(contour, saddle):
    ws = 4  
    new_contour = contour.copy()
    for i in range(len(contour)):
        cc, rr = contour[i, 0, :]
        
        rl, cl = max(0, rr - ws), max(0, cc - ws)
        rh, ch = min(saddle.shape[0], rr + ws + 1), min(saddle.shape[1], cc + ws + 1)
        window = saddle[rl:rh, cl:ch]
       
        br, bc = np.unravel_index(window.argmax(), window.shape)
        if window[br, bc] > 0:
            new_contour[i, 0, :] = (cc + bc - min(ws, cl), rr + br - min(ws, rl))
        else:
            
            return contour
    return new_contour

def getIdentityGrid(N):
    a = np.arange(N)
    b = a.copy()
    aa,bb = np.meshgrid(a,b)
    return np.vstack([aa.flatten(), bb.flatten()]).T

def getChessGrid(quad):
    quadA = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    quadB = getIdentityGrid(4)-1
    quadB_pad = np.pad(quadB, ((0,0),(0,1)), 'constant', constant_values=1)
    C_thing = (np.matrix(M)*quadB_pad.T).T

    C_thing[:,:2] /= C_thing[:,2]
    return C_thing

def getMinSaddleDist(saddle_pts, pt):
    best_dist = None
    best_pt = pt
    for saddle_pt in saddle_pts:
        saddle_pt = saddle_pt[::-1]
        dist = np.sum((saddle_pt - pt)**2)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = saddle_pt
    return best_pt, np.sqrt(best_dist)


def findGoodPoints(grid, spts, max_px_dist=5):
    
    new_grid = grid.copy()
    chosen_spts = set()
    N = len(new_grid)
    grid_good = np.zeros(N,dtype=np.bool_)
    hash_pt = lambda pt: "%d_%d" % (pt[0], pt[1])

    for pt_i in range(N):
        pt2, d = getMinSaddleDist(spts, grid[pt_i,:2].A.flatten())
        if hash_pt(pt2) in chosen_spts:
            d = max_px_dist
        else:
            chosen_spts.add(hash_pt(pt2))
        if (d < max_px_dist): 
            new_grid[pt_i,:2] = pt2
            grid_good[pt_i] = True
    return new_grid, grid_good

def getInitChessGrid(quad):
    quadA = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    return makeChessGrid(M,1)

def makeChessGrid(M, N=1):
    ideal_grid = getIdentityGrid(2+2*N)-N
    ideal_grid_pad = np.pad(ideal_grid, ((0,0),(0,1)), 'constant', constant_values=1) 
   
    grid = (np.matrix(M)*ideal_grid_pad.T).T
    grid[:,:2] /= grid[:,2] 
    grid = grid[:,:2] 
    return grid, ideal_grid, M

def generateNewBestFit(grid_ideal, grid, grid_good):
    a = np.float32(grid_ideal[grid_good])
    b = np.float32(grid[grid_good])
    M = cv2.findHomography(a, b, cv2.RANSAC)
    return M

def getGrads(img):
    img = cv2.blur(img,(5,5))
    gx = cv2.Sobel(img,cv2.CV_64F,1,0)
    gy = cv2.Sobel(img,cv2.CV_64F,0,1)

    grad_mag = gx*gx+gy*gy
    grad_phase = np.arctan2(gy, gx) 
    grad_phase_masked = grad_phase.copy()
    gradient_mask_threshold = 2*np.mean(grad_mag.flatten())
    grad_phase_masked[grad_mag < gradient_mask_threshold] = np.nan
    return grad_mag, grad_phase_masked, grad_phase, gx, gy



def getBestLines(img_warped):
    grad_mag, grad_phase_masked, grad_phase, gx, gy = getGrads(img_warped)

    # X
    gx_pos = gx.copy()
    gx_pos[gx_pos < 0] = 0
    gx_neg = -gx.copy()
    gx_neg[gx_neg < 0] = 0
    score_x = np.sum(gx_pos, axis=0) * np.sum(gx_neg, axis=0)
    # Y
    gy_pos = gy.copy()
    gy_pos[gy_pos < 0] = 0
    gy_neg = -gy.copy()
    gy_neg[gy_neg < 0] = 0
    score_y = np.sum(gy_pos, axis=1) * np.sum(gy_neg, axis=1)

 
    a = np.array([(offset + np.arange(7) + 1)*32 for offset in np.arange(1,11-2)])
    scores_x = np.array([np.sum(score_x[pts]) for pts in a])
    scores_y = np.array([np.sum(score_y[pts]) for pts in a])


    best_lines_x = a[scores_x.argmax()]
    best_lines_y = a[scores_y.argmax()]
    return (best_lines_x, best_lines_y)

def getUnwarpedPoints(best_lines_x, best_lines_y, M):
    x,y = np.meshgrid(best_lines_x, best_lines_y)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = np.expand_dims(xy,0)

    xy_unwarp = cv2.perspectiveTransform(xy, M)
    return xy_unwarp[0,:,:]

def loadImage(image_path, isRotated):
    img_orig = PIL.Image.open(image_path)
    img_width, img_height = img_orig.size
    print(f'In load class, value {isRotated}')
    if not isRotated:
        img_orig = img_orig.rotate(90)
        

    aspect_ratio = min(500.0/img_width, 500.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
    img = img.convert('L') 
    img = np.array(img)

    return img

def findChessboard(img, min_pts_needed=15, max_pts_needed=25):
    blur_img = cv2.blur(img, (3,3)) 
    saddle = getSaddle(blur_img)
    saddle = -saddle
    saddle[saddle<0] = 0
    pruneSaddle(saddle)
    s2 = nonmax_sup(saddle)
    s2[s2<100000]=0
    spts = np.argwhere(s2)

    edges = cv2.Canny(img, 20, 250)
    contours_all, hierarchy = getContours(img, edges)
    contours, hierarchy = pruneContours(contours_all, hierarchy, saddle)

    curr_num_good = 0
    curr_grid_next = None
    curr_grid_good = None
    curr_M = None

    for cnt_i in range(len(contours)):
       
        cnt = contours[cnt_i].squeeze()
        grid_curr, ideal_grid, M = getInitChessGrid(cnt)

        for grid_i in range(7):
            grid_curr, ideal_grid, _ = makeChessGrid(M, N=(grid_i+1))
            grid_next, grid_good = findGoodPoints(grid_curr, spts)
            num_good = np.sum(grid_good)
            
            if num_good < 4:
                M = None
                
                break
            M, _ = generateNewBestFit(ideal_grid, grid_next, grid_good)
            
            if M is None or np.abs(M[0,0] / M[1,1]) > 15 or np.abs(M[1,1] / M[0,0]) > 15:

                M = None
                #print ("Failed to converge on this one")
                break
        if M is None:
            continue
        elif num_good > curr_num_good:
            curr_num_good = num_good
            curr_grid_next = grid_next
            curr_grid_good = grid_good
            curr_M = M

        
        if num_good > max_pts_needed:
            break

  
    if curr_num_good > min_pts_needed:
        final_ideal_grid = getIdentityGrid(2+2*7)-7
        return curr_M, final_ideal_grid, curr_grid_next, curr_grid_good, spts
    else:
        return None, None, None, None, None
#     return M, ideal_grid, grid_next, grid_good, spts

def getBoardOutline(best_lines_x, best_lines_y, M):
    d = best_lines_x[1] - best_lines_x[0]
    ax = [best_lines_x[0]-d, best_lines_x[-1]+d]
    ay = [best_lines_y[0]-d, best_lines_y[-1]+d]
    x,y = np.meshgrid(ax, ay)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = xy[[0,1,3,2,0],:]
    xy = np.expand_dims(xy,0)

    xy_unwarp = cv2.perspectiveTransform(xy, M)
    return xy_unwarp[0,:,:]

def extrapolate_lines(lines):
    spacings = np.diff(lines)
    avg_spacing = np.mean(spacings)
    extended_lines = np.zeros(len(lines) + 2)
    extended_lines[1:-1] = lines
    extended_lines[0] = lines[0] - avg_spacing
    extended_lines[-1] = lines[-1] + avg_spacing
    return extended_lines

def compute_grid_points(size, grid_count=9):
    
    step = size / (grid_count - 1)
    grid_points = np.array([(x * step, y * step) for y in range(grid_count) for x in range(grid_count)])
    return grid_points.reshape((grid_count, grid_count, 2))

def get_squares_from_points(points):
    squares = []
    for i in range(points.shape[0] - 1):
        for j in range(points.shape[1] - 1):
            top_left = points[i, j]
            top_right = points[i, j+1]
            bottom_right = points[i+1, j+1]
            bottom_left = points[i+1, j]
            square = [top_left, top_right, bottom_right, bottom_left]
            squares.append(square)
    return squares

def extract_templates(image, squares):
   
    templates = []
    for square in squares:
        
        pts = np.array(square, np.int32)
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        template = image[y:y+h, x:x+w]
        templates.append(template)
    return templates

def calculate_centroid(points):
        
        points = np.array(points)
        x = np.mean(points[:, 0])
        y = np.mean(points[:, 1])
        return (x, y)

def get_square_centroids(squares):
    
    square_centroids = [calculate_centroid(square) for square in squares]
    return square_centroids

def process_chessboard_image(image_path, isRotated):
    image = loadImage(image_path, isRotated)
    matrix, ideal_grid, grid_next, grid_good, spts = findChessboard(image)

    if matrix is not None:
        matrix, _ = generateNewBestFit((ideal_grid+8)*32, grid_next, grid_good)
        img_warp = cv2.warpPerspective(image, matrix, (17*32, 17*32), flags=cv2.WARP_INVERSE_MAP)
        
        lines_x, lines_y = getBestLines(img_warp)
        xy_unwarp = getUnwarpedPoints(lines_x, lines_y, matrix)

    else:
        print("Matrix is None. Fail")
        return None

    lines_x_extended = extrapolate_lines(lines_x)
    lines_y_extended = extrapolate_lines(lines_y)
    xy_unwarp_extended = getUnwarpedPoints(lines_x_extended, lines_y_extended, matrix)
    points_grid = xy_unwarp_extended.reshape((9, 9, 2))

    squares = []
    for i in range(8):
        for j in range(8):
            top_left = points_grid[i, j]
            top_right = points_grid[i, j + 1]
            bottom_right = points_grid[i + 1, j + 1]
            bottom_left = points_grid[i + 1, j]
            square = [top_left, top_right, bottom_right, bottom_left]
            squares.append(square)

    
    top_left, bottom_left, bottom_right, top_right = points_grid[0, 0], points_grid[0, -1], points_grid[-1, -1], points_grid[-1, 0]
    src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    size = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - top_right),
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - bottom_left)
    )
    dest_points = np.array([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dest_points)
    warped_image = cv2.warpPerspective(image, matrix, (int(size), int(size)))
    grad_mag_image, _ = compute_gradients([warped_image])
    grad_mag_image = grad_mag_image[0]

    new_grid_points = compute_grid_points(int(size))
    new_squares = get_squares_from_points(new_grid_points)
    square_centroids = get_square_centroids(new_squares)

    return {
        'warped_image': warped_image,
        'new_squares': new_squares,
        'new_grid_points': new_grid_points,
        'transformation_matrix': matrix,
        'square_centroids': square_centroids,
        'lines_x': lines_x,
        'lines_y': lines_y,
        'xy_unwarp': xy_unwarp,
        'lines_x_extended': lines_x_extended,
        'lines_y_extended': lines_y_extended,
        'xy_unwarp_extended': xy_unwarp_extended,
        'squares': squares,
        'src_points': src_points,
        'size': size,
        'dest_points': dest_points,
        'grad_mag_image': grad_mag_image
    }

def zero_out_lines(image, grid_points):
    
    for i in range(len(grid_points)):
        for j in range(len(grid_points[0])):
            if j < len(grid_points[0]) - 1:  
                cv2.line(image, tuple(grid_points[i][j].astype(int)), tuple(grid_points[i][j+1].astype(int)), 0, 1)
            if i < len(grid_points) - 1:  
                cv2.line(image, tuple(grid_points[i][j].astype(int)), tuple(grid_points[i+1][j].astype(int)), 0, 1)

def apply_multiscale_morphological_operations(image, kernel_sizes):
    combined_result = None
    for size in kernel_sizes:
        kernel = np.ones((size, size), np.uint8)

        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        if combined_result is None:
            combined_result = closed
        else:
            combined_result = cv2.bitwise_or(combined_result, closed)

    return combined_result

def advanced_morphological_operations(image):

    kernel = np.ones((3,3), np.uint8)

    dilated = cv2.dilate(image, kernel, iterations=1)  

    
    combined = cv2.bitwise_or(image, dilated)

   
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return closed

def subtract_images(artifacts, image2):
    image1 = artifacts['grad_mag_image']
    grid_points = artifacts['new_grid_points']
    
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2

 
    difference = cv2.absdiff(img1, img2)
    

    _, thresholded = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)


    
    if grid_points is not None:
        zero_out_lines(thresholded, grid_points)
        
    
    thresholded = thresholded.astype(np.uint8)
    

    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    

    return difference, thresholded, cleaned

def refine_homography_with_internal_points(frame1_points, frame2_points):

    H, status = cv2.findHomography(frame2_points, frame1_points, cv2.RANSAC)
    return H

def apply_homography(H, points):

    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  
    transformed_points = np.dot(H, points_homogeneous.T).T
    transformed_points /= transformed_points[:, 2][:, np.newaxis]  
    return transformed_points[:, :2]

def warp_image(image, H, size):
    warped_image = cv2.warpPerspective(image, H, (size[1], size[0])) 
    return warped_image

def align_and_compare_images(artifacts_1, artifacts_2):
    
    grid_points_frame1 = np.array(artifacts_1['new_grid_points'].reshape(81, 2))
    grid_points_frame2 = np.array(artifacts_2['new_grid_points'].reshape(81, 2))

    
    refined_homography = refine_homography_with_internal_points(grid_points_frame1, grid_points_frame2)

    
    aligned_points_frame2 = apply_homography(refined_homography, grid_points_frame2)

    
    displacements = grid_points_frame1 - aligned_points_frame2

    
    dimensions = (artifacts_1['grad_mag_image'].shape[1], artifacts_1['grad_mag_image'].shape[0])
    aligned_grad_mag_image_frame2 = warp_image(artifacts_2['grad_mag_image'], refined_homography, dimensions)

    return displacements, aligned_grad_mag_image_frame2

def compute_square_intensity(square, image):
        mask = np.zeros(image.shape, dtype=np.uint8)
        points = np.array([square], dtype=np.int32)
        cv2.fillPoly(mask, points, 255)
        return np.sum(image[mask == 255])

def get_top_squares(artifacts, thresholded_image):
    
    square_intensities = [compute_square_intensity(square, thresholded_image) for square in artifacts['new_squares']]

    
    top_two_indices = np.argsort(square_intensities)[-2:]  

    top_two_squares = [artifacts['new_squares'][i] for i in top_two_indices]
    top_two_intensities = [square_intensities[i] for i in top_two_indices]

    return top_two_indices, top_two_squares, top_two_intensities

def index_to_chess_notation(index):
    file = chr((index % 8) + ord('a'))  
    rank = 8 - (index // 8)  
    return f"{file}{rank}"

def determine_move_direction(artifacts_1, index1, index2):
    grad_image = artifacts_1['grad_mag_image']
    square1 = artifacts_1['new_squares'][index1]
    square2 = artifacts_1['new_squares'][index2]

    avg_intensity1 = np.mean(square1)
    avg_intensity2 = np.mean(square2)

    if avg_intensity1 > avg_intensity2:
        move_start_idx, move_end_idx = index1, index2
    else:
        move_start_idx, move_end_idx = index2, index1

    return move_start_idx, move_end_idx

def nextMove(image_path_1, image_path_2, isRotated=False):
    start_time = time.time()
    
    artifacts_1 = process_chessboard_image(image_path_1, isRotated)
    artifacts_2 = process_chessboard_image(image_path_2, isRotated)

    displacements, aligned_grad_mag_image_frame2 = align_and_compare_images(artifacts_1, artifacts_2)
    difference_image, thresholded_image, cleaned_image = subtract_images(artifacts_1, aligned_grad_mag_image_frame2)

    top_two_indices, top_two_squares, top_two_intensities = get_top_squares(artifacts_2, cleaned_image)
    move_start_idx, move_end_idx = determine_move_direction(artifacts_1, top_two_indices[0], top_two_indices[1])

    
    start_notation = index_to_chess_notation(move_start_idx)
    end_notation = index_to_chess_notation(move_end_idx)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")
    return [start_notation, end_notation]