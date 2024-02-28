import numpy as np


def create_interpolated_matrix_by_divisions(arr, divisions, dims):
    """
    Function to create a matrix by interpolating values from the given array to 1,
    dividing the range into a specified number of divisions, but only for specific dimensions.

    :param arr: Input array with values in the range [0, 1].
    :param divisions: Number of divisions to interpolate between the array value and 1.
    :param dims: Dimensions (indices of the array) to apply the interpolation.
    :return: A matrix with interpolated values for specific dimensions.
    """
    # Calculate the step size based on the number of divisions
    step = 1.0 / divisions

    # Calculate the number of steps required for each element
    num_steps = np.ceil((1 - arr) / step).astype(int)
    max_steps = num_steps.max()

    # Initialize the matrix
    matrix = np.zeros((divisions, len(arr)))

    # Fill the matrix by interpolating values only for specified dimensions
    for i in range(len(arr)):
        if i in dims:
            matrix[:num_steps[i], i]   = np.linspace(arr[i], 1, num_steps[i], endpoint=False)
            matrix[num_steps[i]-1:, i] = 1
        else:
            matrix[:, i] = arr[i]

    return matrix



if __name__ == '__main__':
    # Example array

    example_arr   = np.array([0. , 0.6, 0.5, 0. , 0.5, 0. ])
    specific_dims = [1, 2, 4]

    # Example with divisions
    divisions = 10  # Number of divisions
    interpolated_matrix_divisions = create_interpolated_matrix_by_divisions(example_arr, divisions, specific_dims)
    print(interpolated_matrix_divisions)
