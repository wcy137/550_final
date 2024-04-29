from qiskit_experiments.library.tomography import ProcessTomography
from qiskit_experiments.framework import ParallelExperiment, BatchExperiment
from numpy import arange
from qiskit import QuantumCircuit


def gen_circ_ls(base_qc: QuantumCircuit, num_partitions: int):

    ''' generates a list of circuits
    inputs:
        base_qc: the base quantum circuit to be repeated
        num_partitions: the number of Phi-maps to be used for QPT

    returns:
    qc_ls: list of quantum circuits that are repititions of base circuits
    '''

    qc_ls = []
    for i in range(num_partitions):
        curr_qc = base_qc.repeat(i)
        curr_qc = curr_qc.decompose(reps=1)
        qc_ls.append(curr_qc)
    return qc_ls


def gen_qubit_ls(total_qubits: int, num_qubit: int, repeat: bool):

    ''' generates a list of qubits ready for parallel computation
    inputs:
    total_qubits: the number of qubits to be used on the quantum computer
    num_qubits: the number of qubits that corresponds the dimension of the
    system
    repeat: whether to repeat the qubit list. Used for repeated 2-qubit batch
    experiment

    returns:
    ls: a list of list of indicies for the indicies of the qubits used on the
    devices for each experiment.
    '''

    qubit_arr = arange(0, total_qubits)
    qubit_arr = [qubit_arr[i:i+num_qubit] for i in range(0, len(qubit_arr),
                                                         num_qubit)]
    qubit_ls = [ls.tolist() for ls in qubit_arr]
    if repeat:
        qubit_ls.extend(qubit_ls)
    return qubit_ls


def exp_2q(qc_ls: list, backend, qubit_ls: list, analysis='default'):

    ''' generates a BatchExperiment object that contains two parallel
    experiments
    inputs:
    qc_ls: list of quantum circuits that are repeated sequences
    backend
    qubit_ls: list of qubits generated from gen_qubit_ls with repeat=True
    analysis: analysis method provided by qiskit_experiment
    '''

    exp_ls = []
    for i in range(len(qc_ls)):
        curr_qc = qc_ls[i]
        curr_qubits_used = qubit_ls[i]
        curr_exp = ProcessTomography(curr_qc, backend,
                                     physical_qubits=curr_qubits_used,
                                     analysis=analysis)
        exp_ls.append(curr_exp)
    parallel_exp_ls = [ParallelExperiment(exp_ls[0: 63],
                                          flatten_results=False),
                       ParallelExperiment(exp_ls[63: len(exp_ls)],
                                          flatten_results=False)]
    batch_exp = BatchExperiment(parallel_exp_ls, flatten_results=False)
    return batch_exp
