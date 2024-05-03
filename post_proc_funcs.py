from qiskit.quantum_info import Choi, SuperOp
import numpy as np
# from qiskit.circuit import QuantumCircuit


def extract_channel(experiment_data, vectorization):

    ''' extracts choi matrix from a parallel experiment data
    inputs:
    experiment_data: the experiment data output from a parallel experiment
    vectorization: QPT in qiskit ouputs choi matrix representations. If
    vectorization is true, output superoperator form.

    output:
    proc_ls: list of process operators either in SuperOp or Choi representation
    '''

    proc_ls = []
    data_ls = experiment_data.child_data()
    for i in range(len(data_ls)):
        curr_choi = (data_ls[i].analysis_results(dataframe=True)['value']
                     .iloc[0])
        if vectorization:
            curr_superop = SuperOp(curr_choi)
            proc_ls.append(curr_superop)
        else:
            proc_ls.append(curr_choi)

    return proc_ls


def compute_intmdt_maps(superop_ls):

    ''' computes list of intermediate maps from superoperator list
    input: superop_ls: list of vectorized superoperators

    We create a separate map that are products of maps up to the
    current index. Then multiply the two maps together to get a list
    of intermediate maps.

    output: intmdt_map_ls: list of intermediate maps

    '''
    dim = 2 ** (superop_ls[0].num_qubits * 2)
    intmdt_map_ls = [0 for i in range(len(superop_ls))]
    intmdt_map_ls[0] = superop_ls[0]
    for i in range(1, len(superop_ls)):
        curr_applied_inv = np.identity(dim)
        for j in range(i):
            curr_applied_inv = curr_applied_inv @ np.linalg.pinv(superop_ls[j])
        intmdt_map_ls[i] = superop_ls[i] @ curr_applied_inv
    return intmdt_map_ls


def compute_Drhp(intmdt_map_ls, base_circ_time=90):
    ''' compute normalized non-markovianity
    inputs:
    intmdt_map_ls: list of intermediate maps
    base_circ_time: duration of the shortest circuit

    output: non-markovianity of the map built out of the list of intermediate
    maps
    '''
    gt = []
    for map in intmdt_map_ls:
        gt.append((np.linalg.norm(Choi(map), ord=1) - 1) / base_circ_time)
    Nrhp = sum(gt)
    Drhp = Nrhp / (1 + Nrhp)
    return Drhp
