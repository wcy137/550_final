from qiskit_experiments.library.tomography import ProcessTomography, MitigatedProcessTomography, StateTomography, MitigatedStateTomography
from qiskit_experiments.framework import ParallelExperiment, BatchExperiment
from numpy import arange
from qiskit import QuantumCircuit


def gen_circ_ls(base_qc: QuantumCircuit, num_partitions: int):

    ''' Define a base circuit and create a list of circuits that are
    repititions of the base circuits on increasing times
    inputs:
        base_qc: the base quantum circuit to be repeated
        num_partitions: the number of Phi-maps to be used for QPT

    returns:
    qc_ls: list of quantum circuits that are repititions of base circuits
    '''

    qc_ls = []
    for i in range(1, num_partitions+1):
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
    devices for each experiment. For example, we want to ouput
    [[0, 1], [2, 3], [4, 5], ...]
    '''

    qubit_arr = arange(0, total_qubits)
    qubit_arr = [qubit_arr[i:i+num_qubit] for i in range(0, len(qubit_arr),
                                                         num_qubit)]
    qubit_ls = [ls.tolist() for ls in qubit_arr]
    if repeat:
        qubit_ls.extend(qubit_ls)
    return qubit_ls


def batch_2_parallel_exp_2q(qc_ls: list, backend, qubit_ls: list,
                            analysis='default'):

    ''' generates a BatchExperiment object that contains two parallel
    experiments. This generation of experiments can only be used by
    2-qubit systems on 127 qubit devices.

    inputs:
    qc_ls: list of quantum circuits that are repeated sequences
    backend
    qubit_ls: list of qubits generated from gen_qubit_ls with repeat=True
    analysis: analysis method provided by qiskit_experiment

    We create a parallel circuit of half of the QPTs, those experiments
    run for 144 circuits. The other 144 circuits performs QPT for the
    second half of maps. We then return those experiment ready to be
    run.
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


def parallel_exp_1q2q(qc_ls: list, backend, qubit_ls: list,
                      mitigation=False, analysis='default', state_tom=False):
    ''' generates a ParallelExperiment object that contains floor(127/n) QPT
    experiments where n is the number of qubits of the system. The generated
    experiments can only be implemented on 1 or 2-qubit systems

    inputs
    qc_ls: list of quantum circuits that are repeated sequences
    backend
    qubit_ls: list of qubits generated from gen_qubit_ls with repeat=True
    analysis: analysis method provided by qiskit_experiment
    '''
    # TODO: add a max_circuits options to expand the implementation of
    # experiments onto more qubits by submitting multiple runs to ibm.

    exp_ls = []
    exp_ls = []
    for i in range(len(qc_ls)):
        curr_qc = qc_ls[i]
        curr_qubits_used = qubit_ls[i]
        curr_exp = None
        if mitigation and not state_tom:
            curr_exp = MitigatedProcessTomography(curr_qc, backend,
                                     physical_qubits=curr_qubits_used,
                                     analysis=analysis)
        elif not mitigation and not state_tom:
            curr_exp = ProcessTomography(curr_qc, backend,
                                     physical_qubits=curr_qubits_used,
                                     analysis=analysis)
        elif not mitigation and state_tom: 
            curr_exp = StateTomography(curr_qc, backend,
                                     physical_qubits=curr_qubits_used,
                                     analysis=analysis)
        elif mitigation and state_tom:
            curr_exp = MitigatedStateTomography(curr_qc, backend,
                                     physical_qubits=curr_qubits_used,
                                     analysis=analysis)
        exp_ls.append(curr_exp)
    parallel_exp = ParallelExperiment(exp_ls,
                                      flatten_results=False)
    return parallel_exp




###################################################

def gen_delay_circ_ls(num_qubits:int, num_maps: int, total_us_time:int, start_state='0'):
    """
    Generates delay circuits with increasing delays simulating the identity gate
    
    Args:
        num_qubits: number of qubits to execute the circuit on
        num_maps: number of delay circuits to generate
        total_us_time: total time in microseconds
    """
    qc_ls = []
    dt = round(total_us_time/num_maps,1)
    t = dt
    while t <= total_us_time:
        base_qc = QuantumCircuit(num_qubits)
        base_qc = prepare_state(base_qc, start_state)
        base_qc.delay(t, unit='us')
        qc_ls.append(base_qc)
        t += dt
    return qc_ls


def prepare_state(qc, num_qubits, start_state='0'):
    """
    Prepares the initial state of the circuit
    
    Args:
        qc: QuantumCircuit object
        start_state: desired initial state of the circuit
    """
    if start_state == '0':
        pass
    elif start_state == '1':
        for i in range(num_qubits):
            qc.x(i)
    elif start_state == '+':
        for i in range(num_qubits):
            qc.h(i)
    elif start_state == '-':
        for i in range(num_qubits):
            qc.x(i)
            qc.h(i)
    elif start_state == '+i':
        for i in range(num_qubits):
            qc.h(i)
            qc.s(i)
    elif start_state == '-i':
        for i in range(num_qubits):
            qc.x(i)
            qc.h(i)
            qc.s(i)
    else:
        raise ValueError("Invalid start state. Must be either '0', '1', '+', '-', '+i', or '-i'")
    return qc