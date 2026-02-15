from __future__ import annotations

import sys
import time
from select import select
import termios
import threading
import tty

# from bri import Action, Controller
import zenoh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Label Mapping Dictionary -----
action_map = {
    0: Action.FORWARD,
    1: Action.LEFT,
    2: Action.RIGHT,
    3: Action.STOP
}


def numpy_listener(sample: zenoh.Sample) -> None:
    # Convert payload to numpy array (414720 bytes = 480 × 288 × 3 or 480 × 864 × 1)
    array = np.frombuffer(sample.payload.to_bytes(), dtype=np.uint8)
    # Convert array -> tensor, flatten, and add batch dimension
    x = torch.tensor(array, dtype=torch.float32).flatten().unsqueeze(0)

    # Forward pass
    probs = model(x)

    # Prediction
    pred_idx = torch.argmax(probs, dim=-1).item()
    pred_action = action_map[pred_idx]
    print(f"Predicted action: {pred_action}")
    ctrl.set_action(pred_action)

conf = zenoh.Config()
session = zenoh.open(conf)
input_dim = 414_720
model = lo
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
        pass
    finally:
        # session.close()
        # ctrl.stop()
        pass


if __name__ == "__main__":
    main()
