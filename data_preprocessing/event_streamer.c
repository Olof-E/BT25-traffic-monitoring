#include <stdbool.h>
#include <stdio.h>

#define CHUNK_SIZE 4096

enum EVENT_TYPE {
    CD_OFF = 0x0,
    CD_ON = 0x1,
    EVT_TIME_HIGH = 0x8,
};

struct Event {
    int x;
    int y;
    bool polarity;
    long timestamp;
} typedef Event;

unsigned int mask_6b = 0x3F;
unsigned int mask_11b = 0x7FF;
unsigned int mask_28b = 0xFFFFFFF;

void read_window(long *read_from, long *time_high, Event *event_buffer,
                 int event_buffer_size) {

    FILE *file_handle = fopen("events.raw", "rb");

    unsigned char buffer[CHUNK_SIZE];

    *time_high = 0;
    long window_start = 0;
    long timestamp = 0;
    int event_x, event_y;

    int event_idx = 0;

    fseek(file_handle, *read_from, 0);
    while (fread(buffer, sizeof(buffer), 1, file_handle) &&
           event_idx < event_buffer_size) {
        for (size_t i = 0; i < CHUNK_SIZE; i += 4) {
            if (event_idx >= event_buffer_size) {
                break;
            }
            *read_from += 4;
            unsigned int data = 0;
            data = buffer[i] | buffer[i + 1] << 8 | buffer[i + 2] << 16 |
                   buffer[i + 3] << 24;

            unsigned char event_type = data >> 28;

            if (event_type == CD_OFF || event_type == CD_ON) {
                // Combine lower half with upper half of timestamp
                timestamp = *time_high << 6 | ((data >> 22) & mask_6b);
                // if (timestamp < 1000) {
                //     continue;
                // } //  else if (window_start <= 0 && timestamp > 1000) {
                //     window_start = timestamp;
                // } else if (window_start > 0 &&
                //            (timestamp - window_start) > window_len) {
                //     break;
                // }

                event_x = data >> 11 & mask_11b;
                event_y = data & mask_11b;

                Event curr_event = {event_x, event_y, event_type, timestamp};
                event_buffer[event_idx++] = curr_event;

                // if (event_type) {
                //     printf("CD_ON\n");
                // } else {
                //     printf("CD_OFF\n");
                // }

                // printf("X: %d\n", event_x);
                // printf("Y: %d\n", event_y);
                // printf("timestamp: %ld\n", timestamp);
            } else if (event_type == EVT_TIME_HIGH) {
                // printf("EVT_TIME_HIGH\n");
                //  Extract upper half of full timestamp
                *time_high = data & mask_28b;
            }
        }
    }

    fclose(file_handle);
}

// int main(int argc, char const *argv[]) {
//     int a = 239;
//     long b = 0;
//     read_window(&a, 10000, &b, NULL);
//     return 0;
// }
