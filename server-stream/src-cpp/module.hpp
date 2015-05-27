#ifndef MODULE_HPP
#define MODULE_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "volume.hpp"
#include "tf.hpp"
#include "bindings.hpp"

namespace NodeCuda {

  class Module : public ObjectWrap {
    public:
      static void Initialize(Handle<Object> target);
      static Handle<Value> GetFunction(const Arguments& args);
      static Handle<Value> DestroyTexRef(const Arguments& args);
      static Handle<Value> DestroyOtfTexRef(const Arguments& args);
      CUmodule m_module;


    protected:
      static Persistent<FunctionTemplate> constructor_template;
      static Handle<Value> VolumeTextureAlloc(const Arguments& args);
      static Handle<Value> OTFTextureAlloc(const Arguments& args);
      static Handle<Value> BlockTextureAlloc(const Arguments& args);
      static Handle<Value> Load(const Arguments& args);

      Module() : ObjectWrap(), m_module(0) {}

      ~Module() {}

    private:
      static Handle<Value> New(const Arguments& args);
      static void volumeTextureLoad(unsigned char* h_data, int dim[3], Module *pmodule);
      static void otfTableTextureLoad(TF *input_float_1D, unsigned int otf_size, Module *pmodule);
      static void BlockTextureLoad(unsigned char *blockVolume, int dim[3], Module *pmodule);
      static float4 * getOTFtable(unsigned int tf_start, unsigned int tf_middle1, unsigned int tf_middle2, unsigned int tf_end, unsigned int tf_size);

      CUarray cu_volumeArray;
      CUarray cu_blockArray;
      CUarray otf_array;


  };

}

#endif
