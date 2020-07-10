import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

# Quantum circuit for control
control_params = sympy.symbols('theta_1 theta_2 theta_3')
qubit = cirq.GridQubit(0, 0)
model_circuit = cirq.Circuit(
    cirq.rz(control_params[0])(qubit),
    cirq.ry(control_params[1])(qubit),
    cirq.rx(control_params[2])(qubit))

# Model
controller = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='elu'),
    tf.keras.layers.Dense(3)])

circuits_input = tf.keras.Input(shape=(),
                                dtype=tf.string, 
                                name='circuits_input')

commands_input = tf.keras.Input(shape=(1,),
                                dtype=tf.dtypes.float32,
                                name='commands_input')
dense_2 = controller(commands_input) # dense layer wity input

expectation_layer = tfq.layers.ControlledPQC(model_circuit,
                                             operators = cirq.Z(qubit))
expectation = expectation_layer([circuits_input, dense_2])
model = tf.keras.Model(inputs=[circuits_input, commands_input],
                       outputs=expectation)

# Input Circuit
random_rotations = np.random.uniform(0, 2 * np.pi, 3)
noisy_preparation = cirq.Circuit(
    cirq.rx(random_rotations[0])(qubit),
    cirq.ry(random_rotations[1])(qubit),
    cirq.rz(random_rotations[2])(qubit))
# convert to tensor and make copy
datapoint_circuits = tfq.convert_to_tensor([noisy_preparation] * 2)

# Train and target
commands = np.array([[0], [1]], dtype=np.float32)
expected_outputs = np.array([[1], [-1]], dtype=np.float32)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=loss)
history = model.fit(x=[datapoint_circuits, commands],
                    y=expected_outputs, epochs=30, verbose=1)

print("Trained params: ", controller(tf.constant(commands)).numpy())
print("Estimation: ", model([datapoint_circuits, commands]))