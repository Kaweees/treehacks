from __future__ import annotations

import sys
import time
from select import select
import termios
import threading
import tty
import pickle

from bri import Action, Controller
import zenoh
import numpy as np
import onnxruntime as ort


# ----- Label Mapping Dictionary -----
action_map = {
    0: Action.FORWARD,
    1: Action.LEFT,
    2: Action.RIGHT
}


def numpy_listener(sample: zenoh.Sample) -> None:
    # Unpickle the payload to get the dictionary with 'nirs' and 'eeg' arrays
    data = pickle.loads(sample.payload.to_bytes())
    nirs_array = data['nirs']
    eeg_array = data['eeg']

    print(f"nirs={nirs_array.shape}, eeg={eeg_array.shape}")

    # Concatenate both arrays
    combined_array = np.concatenate([nirs_array.flatten(), eeg_array.flatten()])

    # Reshape for ONNX model: (batch, channels, seq_len)
    x = combined_array.astype(np.float32).reshape(1, -1, 1)

    # Forward pass
    outputs = ort_session.run(None, {'input': x})
    probs = outputs[0]

    # Prediction
    pred_idx = np.argmax(probs, axis=-1).item()
    pred_action = action_map[pred_idx]
    print(f"Predicted action: {pred_action}")
    # ctrl.set_action(pred_action)

conf = zenoh.Config()
session = zenoh.open(conf)

# Load ONNX model
ort_session = ort.InferenceSession("models/model.onnx", providers=ort.get_available_providers())
print(f"Loaded ONNX model with input: {ort_session.get_inputs()[0].name}, shape: {ort_session.get_inputs()[0].shape}")
print(f"Output: {ort_session.get_outputs()[0].name}, shape: {ort_session.get_outputs()[0].shape}")

ctrl = None  # Will be initialized in main()

# Subscribe to key expression
key = "robot/sensor/nirs_observation"
sub = session.declare_subscriber(key, numpy_listener)

print(f"Subscribed to {key}, waiting for data...")

def main() -> None:
    global ctrl
    # ctrl = Controller(backend="sim", hold_s=0.3)
    # ctrl.start()

    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        session.close()
        # if ctrl:
        #     ctrl.stop()


if __name__ == "__main__":
    main()

