from ctypes import *

event_lib = CDLL("./event-reader.dll")


class Event(Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("polarity", c_bool),
        ("timestamp", c_long),
    ]


event_lib.read_window.argtypes = (
    POINTER(c_int),
    POINTER(c_long),
    POINTER(Event),
    c_int,
)

event_lib.read_window.restype = c_void_p


# void read_window(int *read_from, long *time_high, Event *event_buffer, int event_buffer_size)


def c_fill_event_buffer(buffer_size, last_read_from, last_time_high):
    buffer = (Event * buffer_size)(*[])
    read_from = c_int(last_read_from)
    time_high = c_long(last_time_high)
    event_lib.read_window(byref(read_from), byref(time_high), buffer, buffer_size)

    return buffer, read_from.value, time_high.value
