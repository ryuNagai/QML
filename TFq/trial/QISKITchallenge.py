import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# Parameters that the classical NN will feed values into.
txt =  'theta_{}x theta_{}y theta_{}z'
fulltxt = ''
for i in range(4):
    fulltxt += txt.format(i, i, i) + ' '
control_params = sympy.symbols(fulltxt)

# Create the parameterized circuit.
#qubit, qu = cirq.GridQubit.rect(1, 2)
qubit = cirq.GridQubit.rect(1, 4)
model_circuit = cirq.Circuit(
    cirq.rz(control_params[0])(qubit[0]),
    cirq.ry(control_params[1])(qubit[0]),
    cirq.rx(control_params[2])(qubit[0]),
    cirq.rz(control_params[3])(qubit[1]),
    cirq.ry(control_params[4])(qubit[1]),
    cirq.rx(control_params[5])(qubit[1]),
    cirq.rz(control_params[6])(qubit[2]),
    cirq.ry(control_params[7])(qubit[2]),
    cirq.rx(control_params[8])(qubit[2]),
    cirq.rz(control_params[9])(qubit[3]),
    cirq.ry(control_params[10])(qubit[3]),
    cirq.rx(control_params[11])(qubit[3]),
    cirq.CX(qubit[0], qubit[1]),
    cirq.CX(qubit[2], qubit[3]),
    cirq.CX(qubit[1], qubit[2])
    )

#operators = [[cirq.Z(qubit[0]) + cirq.Z(qubit[1]) + cirq.Z(qubit[2]) + cirq.Z(qubit[3])]]

# The classical neural network layers.
controller = tf.keras.Sequential([
    tf.keras.layers.Dense(36, activation='elu'),
    tf.keras.layers.Dense(12)
])

preparation = cirq.Circuit()
datapoint_circuits = tfq.convert_to_tensor([preparation])

circuits_input = tf.keras.Input(shape=(),
                                # The circuit-tensor has dtype `tf.string` 
                                dtype=tf.dtypes.string,
                                name='circuits_input')

commands_input = tf.keras.Input(shape=(1,),
                                dtype=tf.dtypes.float32,
                                name='commands_input')

dense_2 = controller(commands_input)
full_circuit = tfq.layers.AddCircuit()(circuits_input, append=model_circuit)

state_layer = tfq.layers.State()
state = state_layer(full_circuit,
                    symbol_names=control_params,
                    symbol_values=dense_2)

model = tf.keras.Model(
    inputs=[circuits_input, commands_input],
    outputs=state[0])

commands = np.array([0], dtype=np.float32)
expected_state = np.zeros(16, dtype=np.complex64)
expected_state[0] = 1.
expected_outputs = tf.constant([expected_state], dtype=tf.complex64)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
#loss = tf.keras.losses.MeanSquaredError()
def loss(predicted_y, target_y):
    print("predict: ", predicted_y)
    print("target: ", target_y)
    res = tf.math.reduce_sum(tf.math.abs(predicted_y - target_y))
    print("LOSS", res)
    return res

model.compile(optimizer=optimizer, loss=loss)
history = model.fit(x=[datapoint_circuits, commands],
                    y=expected_outputs,
                    epochs=30,
                    verbose=0)

after_params = controller.predict(np.array([0]))[0]
param_dict = {}
for i in range(len(control_params)):
    param_dict.update([(control_params[i], after_params[i])])

print(param_dict)