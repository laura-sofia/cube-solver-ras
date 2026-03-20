
# import cube_solver_folder.photo as photo
# import cube_solver_folder.edge_detector as edge_detector
# import cube_solver_folder.color_analysis as color_analysis
# import cube_solver_folder.cube_solver as cube_solver
# import cube_solver_folder.execute_movements as execute_movements

import photo
import edge_detector
import color_analysis
import cube_solver
import execute_movements
import os

PHOTO_FILEPATHS = []


def take_photos():
    """vertical side than up than down"""

    movements = ['', 'Y', 'Y', 'Y', 'YX', 'XX']
    for i in range(6):
        execute_movements.move(movements[i])
        photo.take_photo(PHOTO_FILEPATHS[i])

    # voltar o cubo a configuracao inicial
    execute_movements.move('X')


def detect_edges():
    colors = []
    for i in range(6):
        colors.append(edge_detector.main(PHOTO_FILEPATHS[i]))
    return colors


def run_kmean(colors):
    labels_2d = color_analysis.group_hsv_kmeans(colors)
    if not color_analysis.check_color(labels_2d):
        return []
    return labels_2d


def clear_all_photos():

    for i in range(6):
        file_path = PHOTO_FILEPATHS[i]

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} deleted successfully")
        else:
            print(f"{file_path} does not exist")


def create_filepaths():
    for i in range(6):
        PHOTO_FILEPATHS.append(f"cubo{i}.jpg")


def main():
    create_filepaths()
    take_photos()
    colors = detect_edges()
    clear_all_photos()
    if len(colors) != 54:
        return

    labels_2d = run_kmean(colors)
    if not labels_2d:
        return

    with open("kmean.txt", "w") as file:
        file.write(str(labels_2d))

    movements_string = cube_solver.solve(labels_2d)

    execute_movements.move(movements_string)


if __name__ == "__main__":
    main()
