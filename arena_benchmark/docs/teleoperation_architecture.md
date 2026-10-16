# Teleoperation Architecture (Experimental)

The simulator-side teleoperation integration is experimental and its protocol may change. It is not a physical robot safety system.

## Process boundary

```text
VR input → ROS 2 Humble process → local transport → Isaac Sim / Arena process
                                      ← state feedback ←
```

ROS 2 and Isaac Sim run in separate Python environments. The simulator-side control step acts on the simulated articulation directly; the ROS process transports commands and state for integration with the companion application.

## Safety and deployment limits

- Local command timeouts should place the simulated base in a stopped state and hold or neutralize joint commands according to the current implementation.
- Emergency-stop behavior in this integration is simulation-side only. It does not replace a certified physical emergency stop or real-robot safety controller.
- Use the documented local transport defaults unless a secured remote setup has been explicitly configured.
- Validate message ordering, timeout behavior, reset behavior, and disconnect handling on the target runtime before collecting data.
- The full GUI/real-robot workflow requires a compatible companion repository that is not included here.

The wire format and runtime identifiers are implementation details, not a stable public protocol. Consult the matching source revision before building an external client.