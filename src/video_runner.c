#include <stdlib.h>
#include <stdio.h>
#include "video_detect.h"

int main(int argc, char** argv){
    if(argc != 7 && argc != 8){
        printf("wrong argument count, get %d arguments, while expecting 7 or 8.\n", argc);
        printf("expected: %s <config_file> <weights_file> <video_path> <classes_name_file> <out_put_json_file> <decrypt_weight(int)> [<threshold> = 0.5]\n", argv[0]);
        exit(1);
    }
    int decrypt_weights = atoi(argv[6]);
    float thresh = 0.5;
    if (argc == 8) thresh = (float)atof(argv[7]);
    detect_in_video(argv[1], argv[2], argv[3], argv[4], thresh, 0.5, argv[5], decrypt_weights, NULL, -1);
}
