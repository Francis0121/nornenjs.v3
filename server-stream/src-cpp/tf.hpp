#include <cuda.h>
#include <stdio.h>
#include <helper_math.h>

struct TF{
    float x;
    float y;
    float z;
    float w;
};

class TransferFunction
{

    public:

        TransferFunction(void);
        ~TransferFunction(void);

    private:

        TF *otf_table;
        int tf_size;

    public:

        void Set_TF_table(unsigned int tf_start, unsigned int tf_middle1, unsigned int tf_middle2, unsigned int tf_end, int tf_size);
        TF* GetTablePointer(void){ return otf_table;}
        int GetTableSize(void) { return tf_size; }

 };