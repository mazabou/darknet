//
// Created by nicolas on 11/12/18.
//


#include <stdio.h>

#include "darknet.h"

int main(int argc, char ** argv)
{
    if(argc != 4){
        printf("Wrong argument");
        printf("Expect call is: %s <path_to_config> <path_to_weights> <path_to_encrypted_weight>\n", argv[0]);
        return 1;
    }

    network * net = load_network(argv[1], NULL, 0);
    load_weights_decrypt(net, argv[2], 0);
    save_weights_encrypt(*net, argv[3], 1);

    return 0;
}
