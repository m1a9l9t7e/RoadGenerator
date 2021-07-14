import pickle
from interpolation import interpolate_track_points_piece_wise
from ip.iteration import get_custom_solution, convert_solution_to_graph
from util import generate_track_points


def write():
    width, height = (4, 4)
    square_size = 2
    track_width = 0.2
    solution = get_custom_solution(width, height)
    graph = convert_solution_to_graph(solution, scale=square_size)
    _, _, points = generate_track_points(graph, track_width=track_width)
    r_polynomials, l_polynomials, c_polynomials = interpolate_track_points_piece_wise(points)
    data = {
        'right': r_polynomials,
        'left': l_polynomials,
        'center': c_polynomials
    }
    output = open('data.pkl', 'wb')
    pickle.dump(data, output)
    output.close()


def read():
    pkl_file = open('data.pkl', 'rb')
    data = pickle.load(pkl_file)
    print(data)
    pkl_file.close()


if __name__ == '__main__':
    write()
    # read()

# roslaunch loewen sim_with_drive_control.launch static_map:=import_test.py
