#include <stdlib.h>
#include <stdio.h>
#include "video_detect.h"

int main(int argc, char** argv){
    if(argc != 8 && argc != 9){
        printf("wrong argument count");
        printf("expected: %s <config_file> <weights_file> <video_path> <classes_name_file> <classes_count(int)> <out_put_json_file> <decrypt_weight(int)> [<threshold> = 0.5]\n", argv[0]);
        exit(1);
    }
    int classes_count = atoi(argv[5]);
    int decrypt_weights = atoi(argv[7]);
    float thresh = 0.5;
    if (argc == 9) thresh = (float)atof(argv[8]);
    detect_in_video(argv[1], argv[2], thresh, argv[3], argv[4], classes_count, 0.5, argv[6], decrypt_weights);
}
