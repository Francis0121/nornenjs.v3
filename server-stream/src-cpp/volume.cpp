#include "volume.hpp"
#include "stdio.h"
#include "stdlib.h"

Volume::Volume(void)
{
	    m_density = NULL;
}

Volume::Volume(int dim[3], char *filename)
{

        size_t size = dim[0]*dim[1]*dim[2] * sizeof(unsigned char);

        FILE *fp = fopen(filename, "rb");
        m_density = new uchar[size];

        size_t read = fread(m_density, 1, size, fp);
        fclose(fp);

        m_dim[0] = dim[0];
        m_dim[1] = dim[1];
        m_dim[2] = dim[2];

}

Volume::~Volume(void)
{

        if(m_density != NULL){
            delete[] m_density;
        }
        m_density = NULL;

}

void Volume::SetVolume(int dim[3], char *filename)
{

	    size_t size = dim[0]*dim[1]*dim[2] * sizeof(unsigned char);

        FILE *fp = fopen(filename, "rb");
      	m_density = new uchar[size];

        size_t read = fread(m_density, 1, size, fp);
        fclose(fp);

    	m_dim[0] = dim[0];
    	m_dim[1] = dim[1];
    	m_dim[2] = dim[2];
}

Volume *Volume::GetVolume(void)
{
        Volume *pVol;

        pVol->m_density = this->m_density;
        pVol->m_dim[3]  = this->m_dim[3];

        return pVol;
}
