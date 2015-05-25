#include "tf.hpp"
#include "stdio.h"
#include "stdlib.h"
#include <helper_math.h>

TransferFunction::TransferFunction(void)
{
	    otf_table = NULL;
        tf_size = 0;
}


TransferFunction::~TransferFunction(void)
{

        if(otf_table != NULL){
             delete[] otf_table;
        }

}

void TransferFunction::Set_TF_table(unsigned int tf_start, unsigned int tf_middle1, unsigned int tf_middle2, unsigned int tf_end,int size)
{

        otf_table = new TF[size];

        tf_size = size;
        float maximumColorR = 1.0f; float maximumColorG = 1.0f; float maximumColorB = 1.0f;
        float minimumColorR = 1.0f; float minimumColorG = 0.3f; float minimumColorB = 0.3f;

                for(int i=0; i<=tf_start; i++){
                    otf_table[i].w = 0.0f;
                    otf_table[i].x = minimumColorR;
                    otf_table[i].y = minimumColorG;
                    otf_table[i].z = minimumColorB;

                }
                for(int i=tf_start+1; i<=tf_middle1; i++){
                    otf_table[i].w = (1.0 / (tf_middle1-tf_start)) * ( i - tf_start);
                    otf_table[i].x = minimumColorR * ( ((maximumColorR - minimumColorR) / (tf_middle1-tf_start)) * ( i - tf_start) + minimumColorR);
                    otf_table[i].y = minimumColorG * ( ((maximumColorG - minimumColorG) / (tf_middle1-tf_start)) * ( i - tf_start) + minimumColorG);
                    otf_table[i].z = minimumColorB * ( ((maximumColorB - minimumColorB) / (tf_middle1-tf_start)) * ( i - tf_start) + minimumColorB);

                }
                for(int i=tf_middle1+1; i<=tf_middle2; i++){
                    otf_table[i].w =1.0f;
                    otf_table[i].x =maximumColorR;
                    otf_table[i].y =maximumColorG;
                    otf_table[i].z =maximumColorB;
                }
                for(int i=tf_middle2+1; i<=tf_end; i++){
                    otf_table[i].w = (1.0 / (tf_end-tf_middle2)) * ( tf_end -i );
                    otf_table[i].x = minimumColorR * ( ((maximumColorR - minimumColorR) / (tf_end-tf_middle2)) * ( tf_end -i ) + minimumColorR);
                    otf_table[i].y = minimumColorG * ( ((maximumColorG - minimumColorG) / (tf_end-tf_middle2)) * ( tf_end -i ) + minimumColorG);
                    otf_table[i].z = minimumColorB * ( ((maximumColorB - minimumColorB) / (tf_end-tf_middle2)) * ( tf_end -i ) + minimumColorB);
                }
                for(int i=tf_end+1; i<tf_size; i++){
                    otf_table[i].w = 0.0f;
                    otf_table[i].x = minimumColorR;
                    otf_table[i].y = minimumColorG;
                    otf_table[i].z = minimumColorB;

                }

}


