from collections import namedtuple
import struct
import numpy as np


Event = namedtuple("Event", "x y polarity timestamp")


class EventStream:

    data = []
    fpath = ""

    event_buffer = []

    mask_6b = np.uint32(0x3F)
    mask_11b = np.uint32(0x7FF)
    mask_28b = np.uint32(0xFFFFFFF)

    # Start at 239 to skip EVT2 headers
    last_event_byte = 239

    time_high = np.uint64(0)

    def __init__(self, fpath):
        self.fpath = fpath

    def read(self) -> Event:
        if len(self.event_buffer) > 0:
            return self.event_buffer.pop(0)
        else:
            with open(self.fpath, "rb") as f:
                f.seek(self.last_event_byte)

                while len(self.event_buffer) < 1000:
                    byte_buffer = f.read(2048)
                    for data in struct.iter_unpack("I", byte_buffer):
                        self.last_event_byte += 4
                        data = data[0]

                        event_type = data >> 28
                        if event_type <= 0x1:  # Handle CD_ON & CD_OFF
                            # Combine lower half with upper half of timestamp
                            timestamp = self.time_high << 6 | ((data >> 22) & self.mask_6b)

                            event_x = data >> 11 & self.mask_11b
                            event_y = data & self.mask_11b

                            polarity = event_type

                            self.event_buffer.append(Event(event_x, event_y, polarity, timestamp))

                        elif (data >> 28) == 0x8:  # Handle EVT_TIME_HIGH
                            # Extract upper half of full timestamp
                            self.time_high = data & self.mask_28b

            return self.read()
