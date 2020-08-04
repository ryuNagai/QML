
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
from cirq.contrib.svg import SVGCircuit

p = 3
K = 20
command_param = 0.1

N = 6

control_params = []
for i in range(p):
    gamma = 'gamma' + str(i)
    beta = 'beta' + str(i)
    control_params.append(sympy.symbols(gamma))
    control_params.append(sympy.symbols(beta))

def CostUnitary(circuit, control_params, i, K):
    circuit.append(cirq.ZZ(qubits[0], qubits[1])**((K / 2 + 1) * control_params[i]))
    circuit.append(cirq.ZZ(qubits[0], qubits[2])**(K / 2 * control_params[i]))
    circuit.append(cirq.Z(qubits[0])**((-K / 2 - 6) * control_params[i]))
    circuit.append(cirq.ZZ(qubits[1], qubits[2])**((K + 1) / 2 * control_params[i]))
    circuit.append(cirq.Z(qubits[1])**((- K / 2 - 7) * control_params[i]))
    circuit.append(cirq.Z(qubits[2])**((- K / 2 - 5) * control_params[i]))
    circuit.append(cirq.ZZ(qubits[3], qubits[4])**((K / 2 + 1) * control_params[i]))
    circuit.append(cirq.ZZ(qubits[3], qubits[5])**(K / 2 * control_params[i]))
    circuit.append(cirq.Z(qubits[3])**((- K / 2 - 6) * control_params[i]))
    circuit.append(cirq.ZZ(qubits[4], qubits[5])**((K + 1) / 2 * control_params[i]))
    circuit.append(cirq.Z(qubits[4])**((- K / 2 - 7) * control_params[i]))
    circuit.append(cirq.Z(qubits[5])**((- K / 2 - 5) * control_params[i]))
    circuit.append(cirq.ZZ(qubits[0], qubits[3])**(2 * control_params[i]))
    circuit.append(cirq.ZZ(qubits[0], qubits[4])**(control_params[i]))
    circuit.append(cirq.ZZ(qubits[1], qubits[3])**(control_params[i]))
    circuit.append(cirq.ZZ(qubits[1], qubits[4])**(2 * control_params[i]))
    circuit.append(cirq.ZZ(qubits[1], qubits[5])**(1 / 2 * control_params[i]))
    circuit.append(cirq.ZZ(qubits[2], qubits[4])**(1 / 2 * control_params[i]))
    circuit.append(cirq.ZZ(qubits[2], qubits[5])**(2 * control_params[i]))
    return circuit
    
def Mixer(circuit, i, N):
    for j in range(N):
        circuit.append(cirq.XPowGate(exponent = control_params[i]).on(qubits[j]))
    return circuit

def init_circuit(N):
    initCircuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, N)
    for j in range(N):
        initCircuit.append(cirq.H(qubits[j]))
    return initCircuit


qubits = [cirq.GridQubit(i, 0) for i in range(N)]
model_circuit = cirq.Circuit()
for i in range(p):
    model_circuit = CostUnitary(model_circuit, control_params, 2 * i, K)
    model_circuit = Mixer(model_circuit, 2 * i + 1, N)

def ZZoperator(i, j):
    return cirq.Z(qubits[i]) * cirq.Z(qubits[j])

operators = [[-1 * ((K / 2 + 1) * ZZoperator(0, 1) + (K / 2) * ZZoperator(0, 2) - \
            (K / 2 + 6) * cirq.Z(qubits[0]) + (K / 2 + 1 / 2) * ZZoperator(1, 2) - \
            (K / 2 + 7) * cirq.Z(qubits[1]) - (K / 2 + 5) * cirq.Z(qubits[2]) + \
            (K / 2 + 1) * ZZoperator(3, 4) + (K / 2) * ZZoperator(3, 5) - \
            (K / 2 + 6) * cirq.Z(qubits[3]) + (K / 2 + 1 / 2) * ZZoperator(4, 5) - \
            (K / 2 + 7) * cirq.Z(qubits[4]) - (K / 2 + 5) * cirq.Z(qubits[5]) + \
            2 * ZZoperator(0, 3) + ZZoperator(0, 4) + ZZoperator(1, 3) + 2 * ZZoperator(1, 4) +\
            1 / 2 * ZZoperator(1, 5) + 1 / 2 * ZZoperator(2, 4) + 2 * ZZoperator(2, 5) + 2 * K + 24)]]

controller = tf.keras.Sequential([
    tf.keras.layers.Dense(20 * p, activation='elu'),
    tf.keras.layers.Dense(len(control_params))
])

init_circuits = tfq.convert_to_tensor([init_circuit(N)])

commands_input = tf.keras.layers.Input(shape=(1),
                                       dtype=tf.dtypes.float32,
                                       name='commands_input')
circuits_input = tf.keras.Input(shape=(),
                                # The circuit-tensor has dtype `tf.string` 
                                dtype=tf.dtypes.string,
                                name='circuits_input')
operators_input = tf.keras.Input(shape=(1,),
                                 dtype=tf.dtypes.string,
                                 name='operators_input')

dense_2 = controller(commands_input) # tf.keras.Sequential

full_circuit = tfq.layers.AddCircuit()(circuits_input, append=model_circuit)

expectation_output = tfq.layers.Expectation()(full_circuit,
                                              symbol_names=control_params,
                                              symbol_values=dense_2,
                                              operators=operators_input)

model = tf.keras.Model(
    inputs=[circuits_input, commands_input, operators_input],
    outputs=[expectation_output])


operator_data = tfq.convert_to_tensor(operators)
commands = np.array([[command_param] for i in range(len(operators))], dtype=np.float32)
expected_outputs = np.array([[0] for i in range(len(operators))], dtype=np.float32)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
loss = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=optimizer, loss=loss)
history = model.fit(
    x=[init_circuits, commands, operator_data],
    y=expected_outputs,
    epochs=200,
    verbose=1)
print('Expectation value', model([init_circuits, commands, operator_data]))
after_params = controller.predict(np.array([command_param]))[0]
print('after params: ', after_params)

param_dict = {}
for i in range(len(control_params)):
    param_dict.update([(control_params[i], after_params[i])])
resolver = cirq.ParamResolver(param_dict)

simulator = cirq.Simulator()
model_circuit.append(cirq.measure(*qubits, key = 'm'))
results = simulator.run(model_circuit, resolver, repetitions=100)

def calc_cost(state, operator):
    qubits = [cirq.GridQubit(i, 0) for i in range(len(state))]
    test_circuit = cirq.Circuit()
    for i in range(len(state)):
        if state[i] == '1':
            test_circuit.append(cirq.X(qubits[i]))
        else:
            test_circuit.append(cirq.I(qubits[i]))

    output_state_vector = cirq.Simulator().simulate(test_circuit).final_state
    qubit_map={qubits[0]: 0, qubits[1]: 1, qubits[2]: 2, qubits[3]: 3, qubits[4]: 4, qubits[5]: 5}
    return operator.expectation_from_wavefunction(output_state_vector, qubit_map).real

hist = results.histogram(key='m')
keys = list(hist.keys())
print('{:6}'.format('state'), '|', '{}'.format('count'), '|', '{}'.format('cost'))
for key in keys:
    binary = '{:0=6b}'.format(key)
    count = '{:5}'.format(hist[key])
    print(binary, '|',  count, '|', calc_cost(binary, operators[0][0]))