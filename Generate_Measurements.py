import numpy as np
import cv2
from scipy.stats import linregress
import os
from google.colab.patches import cv2_imshow

def image_to_world_coordinates(rgb_image, depth_image, x_y_points, aspect_ratio, near_plane, far_plane, fov_horizontal_deg=67.73):

    world_coordinates = []
    image_height, image_width = rgb_image.shape[:2]
    fov_horizontal_rad = np.deg2rad(fov_horizontal_deg)

    #focal length
    fu = image_width / (2 * np.tan(fov_horizontal_rad / 2))

    u0 = image_width / 2
    v0 = image_height / 2

    for x, y in x_y_points:
        #get depth
        depth_value = depth_image[int(y), int(x)]
        scaled_depth_value = (depth_value / 255) * (far_plane - near_plane) + near_plane

        #pixel to world
        X = (x - u0) * scaled_depth_value / fu
        Y = (y - v0) * scaled_depth_value / fu
        Z = scaled_depth_value
        world_coordinates.append((X, Y, Z))

    return world_coordinates

def find_points_of_interest(x_y_points):

    sums = [sum(point) for point in x_y_points]

    #index of the point with the smallest and largest sum
    min_sum_index = sums.index(min(sums))
    max_sum_index = sums.index(max(sums))

    #within 6 units of the x value of the first point
    first_x = x_y_points[min_sum_index][0]
    candidate_points = [(x, y) for x, y in x_y_points if abs(x - first_x) < 6]
    third_point = max(candidate_points, key=lambda p: p[1])

    #smallest y value within 6 units of the x value of the second point
    second_x = x_y_points[max_sum_index][0]
    candidate_points = [(x, y) for x, y in x_y_points if abs(x - second_x) < 6]
    fourth_point = min(candidate_points, key=lambda p: p[1])

    return [third_point, x_y_points[max_sum_index],fourth_point , x_y_points[min_sum_index]]

def draw_points_and_lines(rgb_image, points_of_interest, real_world_coordinates):

    image_with_points = np.copy(rgb_image)

    #points
    for point in points_of_interest:
        cv2.circle(image_with_points, point, 40, (0, 255, 255), -1)

    #lines and text
    for i in range(len(points_of_interest)):

        next_index = (i + 1) % len(points_of_interest)
        cv2.line(image_with_points, points_of_interest[i], points_of_interest[next_index], (255, 255, 255), 10)

        diff_x = abs(real_world_coordinates[next_index][0] - real_world_coordinates[i][0])
        diff_y = abs(real_world_coordinates[next_index][1] - real_world_coordinates[i][1])
        distance = np.sqrt(diff_x**2 + diff_y**2)

        text_x = int((points_of_interest[i][0] + points_of_interest[next_index][0]) / 2)
        text_y = int((points_of_interest[i][1] + points_of_interest[next_index][1]) / 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2
        font_thickness = 2
        color = (208, 245, 42)
        cv2.putText(image_with_points, f'Side Distance: {distance:.4f}', (text_x, text_y), font, font_scale, color, font_thickness)

    return image_with_points

if __name__ == "__main__":
    #RGB and depth images
    # rgb_image = cv2.imread("/content/MyImages/short.png")
    # depth_image = cv2.imread("/content/MyImages/short_depth.png", cv2.IMREAD_GRAYSCALE)
    # rgb_image = cv2.imread("/content/MyImages/long.png")
    # depth_image = cv2.imread("/content/MyImages/long_depth.png", cv2.IMREAD_GRAYSCALE)
    rgb_image = cv2.imread("/content/MyImages/door.png")
    depth_image = cv2.imread("/content/MyImages/door_depth.png", cv2.IMREAD_GRAYSCALE)
    resized_depth_image = cv2.resize(depth_image, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # for inversion of depth image if needed for depth image type
    resized_depth_image = 255 - resized_depth_image

    #camera parameters
    aspect_ratio = 4/3
    near_plane = 26
    # far_plane = 360
    # far_plane = 252
    far_plane = 1200
    fov_horizontal = 67.73
    im_width=550
    im_height=412
# this is for short
#     x_y_points = [
#     (167.578, 66.95), (166.719, 67.594), (150.391, 67.594), (149.531, 68.238), (138.359, 68.238),
#     (137.5, 67.594), (127.188, 67.594), (126.328, 68.238), (121.172, 68.238), (120.313, 68.881),
#     (116.016, 68.881), (115.156, 69.525), (112.578, 69.525), (111.719, 70.169), (110.859, 70.169),
#     (110.859, 371.444), (422.813, 371.444), (422.813, 240.763), (421.953, 240.119), (421.953, 213.725),
#     (421.094, 213.081), (421.094, 178.319), (420.234, 177.675), (420.234, 173.813), (419.375, 173.169),
#     (419.375, 166.731), (418.516, 166.088), (418.516, 155.788), (417.656, 155.144), (417.656, 147.419),
#     (416.797, 146.775), (416.797, 124.244), (417.656, 123.6), (417.656, 110.081), (418.516, 109.438),
#     (418.516, 97.85), (419.375, 97.206), (419.375, 75.963), (418.516, 75.319), (418.516, 70.169),
#     (417.656, 69.525), (417.656, 68.881), (416.797, 68.238), (415.078, 68.238), (414.219, 67.594),
#     (392.734, 67.594), (391.875, 66.95)
# ]
# this is for long
#     x_y_points = [
#     (25.781, 93.988), (25.781, 379.169), (347.188, 379.169), (348.047, 378.525),
#     (393.594, 378.525), (394.453, 377.881), (409.063, 377.881), (409.922, 377.238),
#     (413.359, 377.238), (414.219, 376.594), (414.219, 375.95), (415.078, 375.306),
#     (415.078, 366.294), (414.219, 365.65), (414.219, 290.331), (415.078, 289.688),
#     (415.078, 246.556), (414.219, 245.913), (414.219, 197.631), (413.359, 196.988),
#     (413.359, 185.4), (412.5, 184.756), (412.5, 165.444), (413.359, 164.8),
#     (413.359, 153.856), (412.5, 153.213), (412.5, 134.544), (413.359, 133.9),
#     (413.359, 128.75), (414.219, 128.106), (414.219, 124.244), (415.078, 123.6),
#     (415.078, 113.944), (415.938, 113.3), (415.938, 96.563), (415.078, 95.919),
#     (395.313, 95.919), (394.453, 95.275), (382.422, 95.275), (381.563, 94.631),
#     (363.516, 94.631), (362.656, 93.988)
# ]

# this is for door
    x_y_points = [
    (61.016, 133.256), (61.016, 141.625), (61.875, 142.269), (61.875, 146.131), (62.734, 146.775),
    (62.734, 157.719), (63.594, 158.363), (63.594, 161.581), (62.734, 162.225), (62.734, 169.306),
    (61.875, 169.95), (61.875, 186.044), (61.016, 186.688), (61.016, 368.225), (85.938, 368.225),
    (86.797, 368.869), (97.969, 368.869), (98.828, 369.513), (117.734, 369.513), (118.594, 370.156),
    (224.297, 370.156), (225.156, 370.8), (256.953, 370.8), (257.813, 371.444), (283.594, 371.444),
    (284.453, 372.088), (298.203, 372.088), (299.063, 372.731), (389.297, 372.731), (390.156, 373.375),
    (434.844, 373.375), (435.703, 374.019), (440, 374.019), (440.859, 374.663), (446.016, 374.663),
    (446.016, 138.406), (432.266, 138.406), (431.406, 139.05), (430.547, 138.406), (416.797, 138.406),
    (415.938, 137.763), (409.063, 137.763), (408.203, 137.119), (395.313, 137.119), (394.453, 136.475),
    (283.594, 136.475), (282.734, 135.831), (275, 135.831), (274.141, 136.475), (243.203, 136.475),
    (242.344, 135.831), (194.219, 135.831), (193.359, 135.188), (190.781, 135.188), (189.922, 134.544),
    (188.203, 134.544), (187.344, 133.9), (183.906, 133.9), (183.047, 133.256)
]
    points_of_interest = find_points_of_interest(x_y_points)

    print("Points of Interest:")
    for i, point in enumerate(points_of_interest):
        print(f"Point {i+1}: {point}")

    #image dimensions
    image_height, image_width = rgb_image.shape[:2]
    scaled_points = []
    for point in points_of_interest:
        x_scaled = int(point[0]/im_width * image_width)
        y_scaled = int(point[1]/im_height * image_height)
        scaled_points.append((x_scaled, y_scaled))

    print("Scaled points:", scaled_points)

    real_world_coordinates = image_to_world_coordinates(rgb_image, resized_depth_image, scaled_points, aspect_ratio, near_plane, far_plane, fov_horizontal)

    print("Real-world coordinates:")
    for i, point in enumerate(real_world_coordinates):
        print(f"Point {i+1}: {point} units")

    result_image = draw_points_and_lines(rgb_image.copy(), scaled_points, real_world_coordinates)

    cv2_imshow(result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
