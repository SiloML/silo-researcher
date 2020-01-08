import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
import torch
import asyncio
import websocket
from syft.messaging.message import SearchMessage
import binascii
from syft.federated.train_config import TrainConfig

TIMEOUT_INTERVAL = 999_999

# hook = sy.TorchHook(torch)

class ResearcherWorker(WebsocketClientWorker):
    @property
    def url(self):
        return f"wss://{self.host}:{self.port}/research" if self.secure else f"ws://{self.host}:{self.port}/research"

    def connect(self):
        args = {"max_size": None, "timeout": TIMEOUT_INTERVAL, "url": self.url, "cookie": f"320984,{self.id}"}

        if self.secure:
            args["sslopt"] = {"cert_reqs": ssl.CERT_NONE}

        self.ws = websocket.create_connection(**args)

    def search(self, query):
        # Prepare a message requesting the websocket server to search among its objects
        message = SearchMessage(query)
        # print(message)
        serialized_message = sy.serde.serialize(message)
        # Send the message and return the deserialized response.
        response = self._send_msg(serialized_message)
        # print(response)
        return sy.serde.deserialize(response) # this worker = self fixes a pysyft bug

    # def _send_train_config(self, config: TrainConfig):
    #     return self._send_msg_and_deserialize("set_obj", obj = config)

    def set_train_config(self, *args, **kwargs):
        config = TrainConfig(*args, **kwargs)
        self._train_config_ptr = config.send(self)
        # return self._send_train_config(config)

    def __getitem__(self, *args, **kwargs):
        print("GOT A __GETITEM__ REQUEST")
        print(args)
        print(kwargs)
        return super().__getitem__(*args, **kwargs)

    def send_msg(self, message, location):
        print(message)
        print(location)
        # location.id = self.id
        # print(location)
        return super().send_msg(message, location)

    # def _forward_to_websocket_server_worker(self, message: bin) -> bin:
    #     print("GOT HERE")
    #     print(message)
    #     self.ws.send(str(binascii.hexlify(message)))
    #     response = binascii.unhexlify(self.ws.recv()[2:-1])
    #     return response

    # def _send_msg(self, message: bin, location=None) -> bin:
    #     print("AND THIS IS THE LOCATION")
    #     return self._recv_msg(message)

    def test_hello_world(self):
        return self._send_msg_and_deserialize("test_hello_world")

    # def _send_msg_and_deserialize(self, command_name: str, *args, **kwargs):
    #     print("WE'RE DOING A THING")
    #     message = self.create_message_execute_command(
    #         command_name=command_name, command_owner="self", *args, **kwargs
    #     )

    #     # Send the message and return the deserialized response.
    #     serialized_message = sy.serde.serialize(message)
    #     response = self._send_msg(serialized_message)
    #     return sy.serde.deserialize(response)

# should maybe do is_client_worker ?
# client1 = ResearcherWorker(hook, PROXY_URL, PROXY_PORT, verbose = VERBOSE, id = 125)
# # client2 = WebsocketClientWorker(hook, PROXY_URL, PROXY_PORT, verbose = VERBOSE, id = 625)

# # dataowner = DataownerWorker(hook, PROXY_URL, PROXY_PORT, verbose = VERBOSE, id = 32)

# x = torch.Tensor([1, 2, 3, 4, 5])
# x_ptr = x.send(client1)