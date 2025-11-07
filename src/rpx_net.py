"""Stubs for distributed RPX (WebSocket / Redis / gRPC)."""


class RPXTransport:
    """Abstract transport for RPX messages."""
    def send(self, topic: str, payload: dict) -> None:
        raise NotImplementedError

    def recv(self, topic: str) -> dict:
        raise NotImplementedError


class WebSocketRPX(RPXTransport):
    """WebSocket transport stub (requires `pip install websockets`)."""
    def __init__(self, uri: str):
        self.uri = uri

    async def send(self, topic: str, payload: dict) -> None:
        import json
        import websockets
        async with websockets.connect(self.uri + "/" + topic) as ws:
            await ws.send(json.dumps(payload))

    async def recv(self, topic: str) -> dict:
        import json
        import websockets
        async with websockets.connect(self.uri + "/" + topic) as ws:
            msg = await ws.recv()
            return json.loads(msg)


class RedisRPX(RPXTransport):
    """Redis transport stub (requires `pip install redis`)."""
    def __init__(self, host="localhost", port=6379):
        import redis
        self.r = redis.Redis(host=host, port=port, decode_responses=True)

    def send(self, topic: str, payload: dict) -> None:
        import json
        self.r.publish(topic, json.dumps(payload))

    def recv(self, topic: str) -> dict:
        import json
        p = self.r.pubsub()
        p.subscribe(topic)
        for msg in p.listen():
            if msg["type"] == "message":
                return json.loads(msg["data"])


class GRPCRPX(RPXTransport):
    """gRPC stub (requires proto definition and `grpcio`)."""
    # Placeholder: generate proto messages for pred, hid, deltaC
    def __init__(self, target: str):
        self.target = target

    def send(self, topic: str, payload: dict) -> None:
        # stub: implement grpc call
        pass

    def recv(self, topic: str) -> dict:
        # stub: implement grpc streaming recv
        pass