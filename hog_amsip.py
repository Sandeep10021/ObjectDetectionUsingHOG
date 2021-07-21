import numpy as np

def cell(magnitude,orientation,orientation_start,orientation_end,cell_columns,cell_rows,
         column_index,row_index,size_columns,size_rows,range_rows_start,range_rows_stop,
         range_columns_start,range_columns_stop):
    
    total=0
    
    for cell_row in range(int(range_rows_start),int(range_rows_stop)):
        cell_row_index = row_index + cell_row
        if (cell_row_index < 0 or cell_row_index >= size_rows):
            continue

        for cell_column in range(int(range_columns_start), int(range_columns_stop)):
            cell_column_index = column_index + cell_column
            if (cell_column_index < 0 or cell_column_index >= size_columns
                    or orientation[int(cell_row_index), int(cell_column_index)]
                    >= orientation_start
                    or orientation[int(cell_row_index), int(cell_column_index)]
                    < orientation_end):
                continue

            total += magnitude[int(cell_row_index), int(cell_column_index)]

    return total / (cell_rows * cell_columns)

def hog_histograms(gradient_columns,gradient_rows,
                   cell_columns,cell_rows,
                   size_columns,size_rows,
                   number_of_cells_columns,number_of_cells_rows,
                   number_of_orientations,
                   orientation_histogram):

    magnitude = np.hypot(gradient_columns,gradient_rows)
    orientation = np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % 180


    r_0 = cell_rows / 2
    c_0 = cell_columns / 2
    cc = cell_rows * number_of_cells_rows
    cr = cell_columns * number_of_cells_columns
    range_rows_stop = cell_rows / 2
    range_rows_start = -range_rows_stop
    range_columns_stop = cell_columns / 2
    range_columns_start = -range_columns_stop
    number_of_orientations_per_180 = 180 / number_of_orientations


    # compute orientations integral images
    for i in range(number_of_orientations):
        # isolate orientations in this range
        orientation_start = number_of_orientations_per_180 * (i + 1)
        orientation_end = number_of_orientations_per_180 * i
        c = c_0
        r = r_0
        r_i = 0
        c_i = 0

        while r < cc:
            c_i = 0
            c = c_0

            while c < cr:
                orientation_histogram[r_i, c_i, i] = \
                    cell(magnitude, orientation,
                             orientation_start, orientation_end,
                             cell_columns, cell_rows, c, r,
                             size_columns, size_rows,
                             range_rows_start, range_rows_stop,
                             range_columns_start, range_columns_stop)
                c_i += 1
                c += cell_columns

            r_i += 1
            r += cell_rows
            
def normalize_block(block, method, eps=1e-5):
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    else:
        raise ValueError('Please select a valid normalization method')
    return out

def gradient_hog(channel):
    
    g_row = np.empty(channel.shape, dtype=np.double)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
    g_col = np.empty(channel.shape, dtype=np.double)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]

    return g_row, g_col

def hog_nd(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2),
        block_norm='L2', visualize=False, transform_sqrt=False,
        feature_vector=True, multichannel=None):

    image = np.atleast_2d(image)

    if multichannel is None:
        multichannel = (image.ndim == 3)

    ndim_spatial = image.ndim - 1 if multichannel else image.ndim
    if ndim_spatial != 2:
        raise ValueError('Only images with 2 spatial dimensions are '
                         'supported. If using with color/multichannel '
                         'images, specify `multichannel=True`.')


    if transform_sqrt:
        image = np.sqrt(image)


    if image.dtype.kind == 'u':
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers
        image = image.astype('float')

    if multichannel:
        g_row_by_ch = np.empty_like(image, dtype=np.double)
        g_col_by_ch = np.empty_like(image, dtype=np.double)
        g_magn = np.empty_like(image, dtype=np.double)

        for idx_ch in range(image.shape[2]):
            g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch] = \
                gradient_hog(image[:, :, idx_ch])
            g_magn[:, :, idx_ch] = np.hypot(g_row_by_ch[:, :, idx_ch],
                                            g_col_by_ch[:, :, idx_ch])

        # For each pixel select the channel with the highest gradient magnitude
        idcs_max = g_magn.argmax(axis=2)
        rr, cc = np.meshgrid(np.arange(image.shape[0]),
                             np.arange(image.shape[1]),
                             indexing='ij',
                             sparse=True)
        g_row = g_row_by_ch[rr, cc, idcs_max]
        g_col = g_col_by_ch[rr, cc, idcs_max]
    else:
        g_row, g_col = gradient_hog(image)


    s_row, s_col = image.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations))

    hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row,
                                 n_cells_col, n_cells_row,
                                 orientations, orientation_histogram)

    # now compute the histogram for each cell
    hog_image = None

    if visualize:
        from skimage.draw import draw

        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(orientations)
        # set dr_arr, dc_arr to correspond to midpoints of orientation bins
        orientation_bin_midpoints = (
            np.pi * (orientations_arr + .5) / orientations)
        dr_arr = radius * np.sin(orientation_bin_midpoints)
        dc_arr = radius * np.cos(orientation_bin_midpoints)
        hog_image = np.zeros((s_row, s_col), dtype=float)
        for r in range(n_cells_row):
            for c in range(n_cells_col):
                for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                    centre = tuple([r * c_row + c_row // 2,
                                    c * c_col + c_col // 2])
                    rr, cc = draw.line(int(centre[0] - dc),
                                       int(centre[1] + dr),
                                       int(centre[0] + dc),
                                       int(centre[1] - dr))
                    hog_image[rr, cc] += orientation_histogram[r, c, o]


    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros((n_blocks_row, n_blocks_col,
                                  b_row, b_col, orientations))

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r:r + b_row, c:c + b_col, :]
            normalized_blocks[r, c, :] = \
                normalize_block(block, method=block_norm)


    if feature_vector:
        normalized_blocks = normalized_blocks.ravel()

    if visualize:
        return normalized_blocks, hog_image
    else:
        return normalized_blocks