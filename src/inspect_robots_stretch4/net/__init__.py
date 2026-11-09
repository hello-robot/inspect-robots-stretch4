"""Transport for driving a robot over the network: framing, request/reply with deadlines,
CURVE authentication, clock offsets. Knows nothing about Stretch; the only package here that
imports zmq."""
