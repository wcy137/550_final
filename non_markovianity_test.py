import numpy as np
from scipy.linalg import sqrtm
# from scympy import Matrix

def trace_norm(matrix):
    """
    Calculate the trace norm of a matrix.

    Args:
        matrix (np.ndarray): The matrix to calculate the trace norm of.

    Returns:
        float: The trace norm of the matrix.
    """
    temp = np.matmul(matrix,matrix.conj().T)
    # hermitian and unitary matrices are always diagonalizable
    return np.trace(sqrtm(temp))
    # if Matrix(temp).is_diagonalizable():
    #     return np.trace(sqrtm(temp))
    # else:
    #     raise Exception("input matrix A such that A.A^{dagger} is not diagonalizable")

def get_maximally_entagled_density_matrix(n):
    """
    Generate a maximally entangled density matrix of n qubits.

    Args:
        n (int): The number of qubits.

    Returns:
        np.ndarray: The maximally entangled density matrix.
    """
    dim = 2**n
    max_entangled_state = np.array([1/np.sqrt(dim)]*dim)
    return np.outer(max_entangled_state.conj(), max_entangled_state)

def apply_kraus_map(operators, rho):
    result = np.zeros(rho.shape, dtype=complex)
    for op in operators:
        op_dag = np.conjugate(np.transpose(op))
        result += np.matmul(np.matmul(op, rho),op_dag)
    return result

def get_extended_choi_matrix_operators(kraus_map):
    """
    Calculate the extended Choi matrix of a map given by its Kraus operators.

    Args:
        kraus_map (list): The Kraus operators of the map.

    Returns:
        np.ndarray: The extended Choi matrix of the map.
    """
    identity = np.eye(2)
    return np.array([np.kron(identity, op) for op in kraus_map])


# First non-markovianity measure
def first_non_markovianity_measure(kraus_map_list, operating_qubits, delta_t):
    """
    Equations 15.3.1 from lecture notes
    We test for non-markovianity based on divisibility of the map.

    D_NM = G / (G + 1)
    G = Intagral_0^inf dt g(t)
    g(t) = (||I tensor Nu_{t+dt,t}|| - 1)/dt
    Nu_{t+dt,t} = T_+ Exp[Integral_s^t L_{\tau} d\tau]

    G (approx) = sum_i g(t_i)*dt (from i=0 to some large time)

    Problem: time is taken from 0 to infinity, which is not practical.

    Args:
        nu_map (dict): The map to be tested.
        operating_qubits (int): The number of qubits the computation was done on.

    Returns:
        float: The measure of non-markovianity.
        0 if markovian, else is non-markovian and quantifies how non-markovian the map is.
    """
    # using the choi matrix of the map (pg 261 of book)
    entagled_rho = get_maximally_entagled_density_matrix(operating_qubits+1)
    G = 0
    for kraus_map in kraus_map_list:
        new_kraus_operators = get_extended_choi_matrix_operators(kraus_map)
        new_rho = apply_kraus_map(new_kraus_operators, entagled_rho)
        g = trace_norm(new_rho) - 1
        G += g # integral sum
    return G / (G + 1)


#############################################

def trace_distance(rho1, rho2):
    rho1 = np.array(rho1)
    rho2 = np.array(rho2)
    return 0.5 * np.linalg.norm(rho1 - rho2, ord=1)

def second_non_markovian_helper_function(states):
    """
    States must be a list of density matrices of the system at different times.
    basis_states[i] = [[p_{1,i} (t1), p_{2,i} (t1)], [p_{1,i} (t2), p_{2,i} (t2)], ...]
    basis_states[i] are in some basis {X,Y,Z}

    states = [basis_states[X], basis_states[Y], basis_states[Z]]
    """
    basis_values = [0,0,0] # basis values for [X,Y,Z]

    for basis, basis_states in enumerate(states):
        n = len(basis_states)
        for i in range(0,n-1,1):
            before = trace_distance(basis_states[i][0],basis_states[i][1])
            after = trace_distance(basis_states[i+1][0],basis_states[i+1][1])
            sigma = after - before # derivative of trace distance
            basis_values[basis] += max(sigma,0)
    return max(basis_values)
