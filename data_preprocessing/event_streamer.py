from collections import namedtuple
import numpy as np


Event = namedtuple("Event", "x y polarity timestamp")


class EventStream:

    data = []

    event_buffer = []

    mask_6b = np.uint32(0x3F)
    mask_11b = np.uint32(0x7FF)
    mask_28b = np.uint32(0xFFFFFFF)

    last_event_idx = 0

    time_high = np.uint64(0)

    def __init__(self, fpath):
        self.data = np.memmap(filename=fpath, mode="r", dtype=np.uint32, offset=239)

    def read(self) -> Event:
        if len(self.event_buffer) > 0:
            return self.event_buffer.pop(0)
        else:
            i = 0
            while i < 1_000:
                next_wrd = self.data[self.last_event_idx]
                self.last_event_idx += 1

                event_type = next_wrd >> 28
                if event_type == 0x0 or event_type == 0x1:  # Handle CD_OFF & CD_ON
                    # print(f"CD_{event_type}")
                    timestamp = self.time_high << 6 | next_wrd >> 22 & self.mask_6b
                    # print(f"t: {timestamp:_}")
                    event_x = next_wrd >> 11 & self.mask_11b
                    event_y = next_wrd & self.mask_11b

                    polarity = event_type
                    # print(f"x: {event_x}")
                    # print(f"y: {event_y}")
                    # print(f"p: {polarity}")

                    self.event_buffer.append(Event(event_x, event_y, polarity, timestamp))
                    i += 1
                elif (next_wrd >> 28) == 0x8:  # Handle EVT_TIME_HIGH
                    # print("EVT_TIME_HIGH")
                    self.time_high = next_wrd & self.mask_28b

            return self.read()


# print("===========================================")


# with open("./events.raw", "rb") as f:
#     f.seek(239)
#     read = 1
#     total = 0

#     # time_high = np.uint32(0)
#     while read < 15:
#         byte_buffer = f.read1(2**16)
#         for i in range(0, (2**16), 4):
#             if i > 15:
#                 break
#             test = np.uint32(0)
#             test |= byte_buffer[i]
#             test |= byte_buffer[i + 1] << 8
#             test |= byte_buffer[i + 2] << 16
#             test |= byte_buffer[i + 3] << 24

#             event_type = test >> 28
#             if event_type <= 0x1:
#                 print(f"CD_{event_type}")
#                 timestamp = time_high << 6 | test >> 22 & mask_6b
#                 print(f"t: {timestamp:_}")
#                 event_x = test >> 11 & mask_11b
#                 event_y = test & mask_11b

#                 polarity = event_type
#                 print(f"x: {event_x}")
#                 print(f"y: {event_y}")
#                 print(f"p: {polarity}")
#             elif (test >> 28) == 0x8:
#                 print("EVT_TIME_HIGH")
#                 time_high = test & mask_28b

#             read += 1
