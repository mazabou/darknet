#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "video_detect.h"
#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#include <libgen.h>
#include <limits.h>

#ifdef OPENCV
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH
#include <opencv2/videoio/videoio_c.h>
#endif
#include "http_stream.h"
#include "../include/darknet.h"

//image get_image_from_stream(CvCapture *cap);

static char **video_detect_names;
static image **video_detect_alphabet;
static int video_detect_classes;

static int nboxes = 0;
static detection *dets = NULL, *previousDets = NULL;

static network net;
static image in_s ;
static image det_s;
static CvCapture * cap;
static int cpp_video_capture = 0;
static float video_detect_thresh = 0;

//static float* predictions[NFRAMES];
//static int video_detect_index = 0;
//static image images[NFRAMES];
//static IplImage* ipl_images[NFRAMES];
//static float *avg;

//image get_image_from_stream_resize(CvCapture *cap, int w, int h, int c, IplImage** in_img, int cpp_video_capture, int dont_close);
//image get_image_from_stream_letterbox(CvCapture *cap, int w, int h, int c, IplImage** in_img, int cpp_video_capture, int dont_close);

IplImage* in_img;
IplImage* det_img;
//IplImage* show_img;

static int flag_video_end, flagDetectionEnd;
static int letter_box = 0;

struct detection_list_element{
    struct detection_list_element * next;
    detection * dets;
    int nboxes;
};
struct detection_list_element * detection_list_head = NULL;
static float video_width = 0.0;
static float video_height = 0.0;
static float video_fps = 0.0;

void *fetch_frame_in_thread(void *ptr)
{
    int dont_close_stream = 0;    // set 1 if your IP-camera periodically turns off and turns on video-stream
    if(letter_box)
        in_s = get_image_from_stream_letterbox(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
    else
        in_s = get_image_from_stream_resize(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
    if(!in_s.data){
        printf("Stream closed.\n");
        flag_video_end = 1;
        //exit(EXIT_FAILURE);
        return 0;
    }
    //in_s = resize_image(in, net.w, net.h);

    return 0;
}

void *detect_frame_in_thread(void *ptr)
{
//    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

//    memcpy(predictions[video_detect_index], prediction, l.outputs*sizeof(float));
//    mean_arrays(predictions, NFRAMES, l.outputs, avg);
//    l.output = avg;

    printf("prefree3\n");
    free_image(det_s);
    printf("postfree3\n");

//    ipl_images[video_detect_index] = det_img;
//    det_img = ipl_images[(video_detect_index + NFRAMES / 2 + 1) % NFRAMES];
//    video_detect_index = (video_detect_index + 1) % NFRAMES;

    if (letter_box)
        dets = get_network_boxes(&net, get_width_mat(in_img), get_height_mat(in_img), video_detect_thresh, video_detect_thresh, 0, 1, &nboxes, 1); // letter box
    else
        dets = get_network_boxes(&net, net.w, net.h, video_detect_thresh, video_detect_thresh, 0, 1, &nboxes, 0); // resized

    return 0;
}

void detections_to_rois(detection * dets, int det_count, char * rois, char * signs)
{
    int i,j;
    char is_first_sign = 1;

    for(i = 0; i < det_count; ++i){
        int class_id = -1;
        for(j = 0; j < video_detect_classes; ++j){
            if (dets[i].prob[j] > video_detect_thresh){
                class_id = j;
                break;
            }
        }
        if(class_id >= 0){
            box b = dets[i].bbox;

            int left   = (int)((b.x - (b.w / 2.f)) * video_width);
            int top    = (int)((b.y - (b.h / 2.f)) * video_height);
            int width  = (int)(b.w * video_width);
            int height = (int)(b.h * video_height);

            if(left < 0) left = 0;
            if(left + width > (int)video_width - 1) width = (int)video_width - 1 - left;
            if(top < 0) top = 0;
            if(top + height > (int)video_height - 1) height = (int)video_height - 1 - top;

            if(is_first_sign == 1){
                is_first_sign = 0;
            }
            else{
                strcat(signs, ",");
            }
            sprintf(signs,"%s\n"
                          "                    {\"coordinates\": [%d,%d,%d,%d],\n"
                          "                     \"detection_confidence\": %f,\n"
                          "                     \"class\": \"%s\"\n"
                          "                    }",
                    signs, left, top, width, height, dets[i].prob[j], video_detect_names[class_id]);

            sprintf(rois, "%s%s,%d,%d,%d,%d;", rois, video_detect_names[class_id], left, top, width, height);
        }
    }
}

struct write_in_thread_args{
    struct detection_list_element * list_first_element;
    char * output_json_file;
    void * cap;
    char * weightsPath;
};

void *write_in_thread(void * raw_args)
{
    struct write_in_thread_args * args = (struct write_in_thread_args *)raw_args;
    struct detection_list_element * cur_element = args->list_first_element;
    FILE *json = fopen(args->output_json_file, "w");
    if(json == NULL){
        printf("Cannot open file: '%s' !\n", args->output_json_file);
        exit(1);
    }

    // write basic header:
    time_t now;
    time (&now);
    struct tm * timeinfo;
    timeinfo = localtime (&now);
    char timeText[128];
    strftime(timeText, 128, "%A %d %B %Y, %H:%M", timeinfo);
    double video_fps = get_cap_property(args->cap, CV_CAP_PROP_FPS);
    fprintf(json, "{\n"
                  "    \"output\": {\n"
                  "        \"video_cfg\": {\n"
                  "            \"datetime\": \"\",\n"
                  "            \"route\": \"\",\n"
                  "            \"com_pos\": \"\",\n"
                  "            \"video_fps\": %f,\n"
                  "            \"resolution\": \"%dx%d\"\n"
                  "        },\n"
                  "        \"framework\": {\n"
                  "            \"name\": \"darknet\",\n"
                  "            \"version\": \"%s\",\n"
                  "            \"test_date\": \"%s\",\n"
                  "            \"weights\": \"%s\"\n"
                  "        },\n"
                  "        \"frames\": [\n",
            video_fps, (int)video_width, (int)video_height, __DATE__, timeText, basename(args->weightsPath));

    int frame_number = 0;

    while(flagDetectionEnd != 1 || cur_element->next != NULL){
        if(cur_element->next == NULL){
            sleep(1); // if list already empty, sleep one second
        }
        else{
            struct detection_list_element * old_element = cur_element;
            cur_element = cur_element->next;

            //clean old element:
            printf("prefree4\n");
            free_detections(old_element->dets, old_element->nboxes);
            free(old_element);
            printf("postfree4\n");

            char rois[512] = "";
            char signs[4096] = "";
            detections_to_rois(cur_element->dets, cur_element->nboxes, rois, signs);

            if(frame_number != 0){
                fprintf(json, ",\n");
            }
            fprintf(json, "            {\n"
                          "                \"frame_number\": \"%07d.jpg\",\n"
                          "                \"RoIs\": \"%s\",\n"
                          "                \"signs\": [%s]\n"
                          "            }", frame_number, rois, signs);

            frame_number++;
        }
    }
    fprintf(json, "\n        ]\n"
                  "    }\n"
                  "}");

    fclose(json);
    return 0;
}

int ms_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (int)time.tv_sec * 1000000 + (int)time.tv_usec;
}

void * feedDetectionListFromPreviousDets(){
    const float nms = .45;    // 0.4F
    int local_nboxes = nboxes;

//    printf("nms\n");
//    if (nms){
//        int classes_count = net.layers[net.n-1].classes;
//        printf("read\n");
//        printf("previousDets: %d len: %d\n", previousDets, local_nboxes);
//        if(local_nboxes > 0){
//            detection det = previousDets[local_nboxes - 1];
//            printf("last? : %d\n", det.classes);
//        }
//        else {
//            printf("nothing\n");
//        }
//        do_nms_sort(previousDets, local_nboxes, classes_count, nms);
//    }

    printf("append\n");
    // add previous detection to the list
    struct detection_list_element * new_detection;
    new_detection = (struct detection_list_element *) malloc(sizeof(struct detection_list_element));
    new_detection->next = NULL;
    new_detection->dets = previousDets;
    new_detection->nboxes = local_nboxes;
    detection_list_head->next = new_detection;
    detection_list_head = new_detection;
    printf("done\n");
}

void detect_in_video(char *cfgfile, char *weightfile, char *video_filename,
                     char *classes_names_file, float thresh, float hier, char *json_output_file, int decrypt_weights,
                     const float * detectionTimeIntervalArray, int intervalCount)
{
    setbuf(stdout, NULL);
    in_img = det_img = NULL;
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    video_detect_names = get_labels(classes_names_file);
    video_detect_alphabet = alphabet;
    video_detect_thresh = thresh;
    printf("Video Detector\n");
    net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    detection_list_head = (struct detection_list_element *) malloc(sizeof(struct detection_list_element));
    detection_list_head->next = NULL;
    detection_list_head->dets = NULL;
    detection_list_head->nboxes = 0;

    srand(2222222);

    printf("video file: %s\n", video_filename);
    cpp_video_capture = 1;
    cap = get_capture_video_stream(video_filename);

    if (!cap) {
#ifdef WIN32
        printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
        error("Couldn't open file.\n");
    }
    video_height = (float)get_cap_property(cap, CV_CAP_PROP_FRAME_HEIGHT);
    video_width = (float)get_cap_property(cap, CV_CAP_PROP_FRAME_WIDTH);
    video_fps = (float)get_cap_property(cap, CV_CAP_PROP_FPS);
    int videoFrameCount = (int)get_cap_property(cap, CV_CAP_PROP_FRAME_COUNT);

    // convert time of detection into frames
    int frameDetectionInterval[intervalCount*2];
    for(int i=0 ; i<(intervalCount * 2) ; i++){
        frameDetectionInterval[i] = (int)(video_fps * detectionTimeIntervalArray[i]);
        if(frameDetectionInterval[i] > videoFrameCount){
            if(i % 2 == 0){
                intervalCount = i / 2;
            }
            else{
                intervalCount = i / 2 + 1;
            }
            break;
        }
    }
    int currentDetectionIntervalIndex = 0;
    int nextIntervalStart, nextIntervalEnd;
    if(intervalCount > 0){
        // if interval available setup the first one
        nextIntervalStart = frameDetectionInterval[0];
        nextIntervalEnd = frameDetectionInterval[1];
        printf("%d intervals to run detection on in the video\n", intervalCount);
        printf("first interval start at frame %d and end at %d\n", nextIntervalStart, nextIntervalEnd);
    }
    else if(intervalCount < 0){
        // interval disabled, run the full video
        nextIntervalStart = 0;
        nextIntervalEnd = INT_MAX;
        printf("Intervals disabled, running on every frame\n");
    } else{
        // if no interval, run nothing (but the first frames...)
        nextIntervalStart = (int)get_cap_property(cap, CV_CAP_PROP_FRAME_COUNT);
        nextIntervalEnd = INT_MAX;
        printf("No intervals, running nothing\n");
    }

    layer l = net.layers[net.n-1];
    video_detect_classes = l.classes;

//    int j;
//    avg = (float *) calloc(l.outputs, sizeof(float));
//    for(j = 0; j < NFRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
//    for(j = 0; j < NFRAMES; ++j) images[j] = make_image(1,1,3);

    flag_video_end = 0;
    flagDetectionEnd = 0;

    pthread_t fetch_thread;
    pthread_t detect_thread;
    pthread_t write_thread;

    int detection_time = ms_time();

    fetch_frame_in_thread(0);
    det_img = in_img;
    det_s = in_s;


    fetch_frame_in_thread(0);
    detect_frame_in_thread(0);
    det_img = in_img;
    det_s = in_s;
    previousDets = dets;

//    for (j = 0; j < NFRAMES / 2; ++j) {
//        fetch_frame_in_thread(0);
//        detect_frame_in_thread(0);
//        det_img = in_img;
//        det_s = in_s;
//    }

    struct write_in_thread_args writer_args;
    writer_args.list_first_element = detection_list_head;
    writer_args.output_json_file = json_output_file;
    writer_args.cap = cap;
    writer_args.weightsPath = weightfile;
//    if(pthread_create(&write_thread, 0, write_in_thread, &writer_args)) error("Thread creation failed");

    int frameNumber = 1; // last image to be read was image number 1 (0 and 1 had been read)
    int frameSkipped = 0;

    while(1){
        ++frameNumber;
        {
            printf("loop\n");
            if(frameNumber < nextIntervalStart){
                // handle previous image detection
                feedDetectionListFromPreviousDets();
                // start loading next frame for detection
                set_cap_property(cap, CV_CAP_PROP_POS_FRAMES, (double)(nextIntervalStart-1 < videoFrameCount ? nextIntervalStart-1 : videoFrameCount));
                printf("start thread\n");
                fetch_frame_in_thread(0);
//                if(pthread_create(&fetch_thread, 0, fetch_frame_in_thread, 0)) error("Thread creation failed");
                frameSkipped += nextIntervalStart - frameNumber - 1;
                printf("skipping frames\n");
                for(; frameNumber<nextIntervalStart; frameNumber++){
                    // add fake empty detection
                    struct detection_list_element * new_detection;
                    new_detection = (struct detection_list_element *) malloc(sizeof(struct detection_list_element));
                    new_detection->next = NULL;
                    new_detection->dets = NULL;
                    new_detection->nboxes = 0;
                    detection_list_head->next = new_detection;
                    detection_list_head = new_detection;
                }
                // clean previous loaded image that were not used for detection
                printf("prefree\n");
                free_image(det_s);
                release_mat(&det_img);
                printf("postfree\n");
                // join frame loading thread
//                pthread_join(fetch_thread, 0);
                if (flag_video_end == 1) break;
                // update prediction pointers
                det_img = in_img;
                det_s = in_s;
                printf("end\n");
            }
            else{
                fetch_frame_in_thread(0);
                detect_frame_in_thread(0);
//                if(pthread_create(&fetch_thread, 0, fetch_frame_in_thread, 0)) error("Thread creation failed");
//                if(pthread_create(&detect_thread, 0, detect_frame_in_thread, 0)) error("Thread creation failed");

                //if we are at the end on the currrant section, setup the value for the next one
                if(frameNumber > nextIntervalEnd) {
                    currentDetectionIntervalIndex++;
                    // if this was the last section set the value such as the rest of the video is filled with empty detection
                    if (currentDetectionIntervalIndex >= intervalCount || videoFrameCount < frameDetectionInterval[currentDetectionIntervalIndex * 2]) {
                        nextIntervalStart = videoFrameCount;
                        nextIntervalEnd = INT_MAX;
                    } else {
                        nextIntervalStart = frameDetectionInterval[currentDetectionIntervalIndex * 2];
                        nextIntervalEnd = frameDetectionInterval[currentDetectionIntervalIndex * 2 + 1];
                    }
                }

                // clear memory of previous frame
                printf("prefree2\n");
                release_mat(&det_img);
                printf("postfree2\n");

                // add previous detection to the list if a detection was done on previous frame
                if(frameNumber != nextIntervalStart) {
                    feedDetectionListFromPreviousDets();
                }
                printf("feed\n");

//                int cur_time = ms_time();
//                printf("\rFPS:%.2f  ",1e6/(double)(cur_time - detection_time + 1)); // prevent 0 div error
//                detection_time = cur_time;

//                pthread_join(fetch_thread, 0);
//                pthread_join(detect_thread, 0);

                if (flag_video_end == 1) break;

                det_img = in_img; // in_img is the full size version of in_s, we don't need it here
                det_s = in_s;
                previousDets = dets;
            }
        }
    }
    printf("\ninput video stream closed. \n");
    // process last detection
    previousDets = dets;
    feedDetectionListFromPreviousDets();
    flagDetectionEnd = 1;

    printf("During this run, %d frames were skipped (%d%%)\n", frameSkipped, frameSkipped * 100 / videoFrameCount );

    write_in_thread(&writer_args);
//    pthread_join(write_thread, 0);
    printf("Write finished.\n");

    // free memory
    free_detections(detection_list_head->dets, detection_list_head->nboxes);
    free_image(in_s);

//    free(avg);
//    for (j = 0; j < NFRAMES; ++j) free(predictions[j]);
//    for (j = 0; j < NFRAMES; ++j) free_image(images[j]);

    const int nsize = 8;
    for (int j = 0; j < nsize; ++j) {
        for (int i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);
    free_network(net);
    //cudaProfilerStop();
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
    int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif
