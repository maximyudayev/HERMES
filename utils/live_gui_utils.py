"""
Class that is concerned with communicating with the live GUI. 
It is used to store a buffer of data and transmit it when full to the GUI via UDP specified by a given ip and port.
"""
import numpy as np
import socket


class LiveGUIPoster:
    def __init__(self,
                tag: str,
                ip_gui: str,
                port_gui: str,
                buffer_shape: tuple[int, int, int],
                buffer_dtype: type):
        
        self.tag = tag

        self._data_buffer_index = 0
        self._data_buffer_counter = 0
        self._data_buffer_shape = buffer_shape
        self._data_buffer = np.zeros(buffer_shape, dtype=buffer_dtype)

        self.GUIsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ip_gui = ip_gui
        self.port_gui = int(port_gui)

    def append_to_data_buffer(self, data: np.ndarray) -> bool:
        """Append a sample to the buffer. Return True if buffer is full."""
        idx = self._data_buffer_index
        self._data_buffer[:, :, idx] = data
        self._data_buffer_index = idx + 1

        if self._data_buffer_index >= self._data_buffer.shape[-1]:
            self.flush_data_buffer()
            return True
        return False
        
    def post_data_UDP(self) -> None:
        for i in range(self._data_buffer.shape[0]):
            samples = self._data_buffer[i]   
            payload = f"sensor||{self.tag}-{i+1}||{self._data_buffer_counter}||".encode() + samples.tobytes()
            self.GUIsocket.sendto(payload, (self.ip_gui, self.port_gui))
        
    def flush_data_buffer(self) -> None:
        # reset valid data index to 0
        self._data_buffer_index = 0
        self.post_data_UDP()
        # add data counter
        self._data_buffer_counter += 1
