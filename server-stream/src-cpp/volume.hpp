typedef unsigned char uchar;

class Volume
{
    public:

        Volume(void);
        Volume(int dim[3], char* filename);
        ~Volume(void);

    private:

        uchar *m_density;
        int m_dim[3];

    public:

        void SetVolume(int dim[3], char *filename);

        uchar* GetDensityPointer(void) { return m_density; }
        int* GetDimension(void) { return m_dim; }

        Volume *GetVolume(void);

 };