#ifndef VIDEO_DETECT
#define VIDEO_DETECT

//#include "image.h"
#ifdef __cplusplus
extern "C" {
#endif
void detect_in_video(char *cfgfile, char *weightfile, char *video_filename,
                     char *classes_names_file, float thresh, float hier, char *json_output_file, int decrypt_weights,
                     const float * detectionTimeIntervalArray, int intervalCount);
#ifdef __cplusplus
}
#endif

#endif //VIDEO_DETECT
