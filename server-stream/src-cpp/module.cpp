#include "module.hpp"
#include "function.hpp"
#include <stdio.h>
#include <cstring>
#include <helper_math.h>
#include <node_buffer.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
using namespace NodeCuda;

Persistent<FunctionTemplate> Module::constructor_template;
extern "C" void setTextureFilterMode();

unsigned int pre_tf_start;
unsigned int pre_tf_middle1;
unsigned int pre_tf_middle2;
unsigned int pre_tf_end;

void Module::Initialize(Handle<Object> target) {
      HandleScope scope;

      Local<FunctionTemplate> t = FunctionTemplate::New(Module::New);
      constructor_template = Persistent<FunctionTemplate>::New(t);
      constructor_template->InstanceTemplate()->SetInternalFieldCount(1);
      constructor_template->SetClassName(String::NewSymbol("CudaModule"));

      // Module objects can only be created by load functions
      NODE_SET_METHOD(target, "moduleLoad", Module::Load);
      NODE_SET_PROTOTYPE_METHOD(constructor_template, "getFunction", Module::GetFunction);
      NODE_SET_PROTOTYPE_METHOD(constructor_template, "destroyTexRef", Module::DestroyTexRef);
      NODE_SET_PROTOTYPE_METHOD(constructor_template, "destroyOtfTexRef", Module::DestroyOtfTexRef);
      NODE_SET_PROTOTYPE_METHOD(constructor_template, "memVolumeTextureAlloc", Module::VolumeTextureAlloc);
      NODE_SET_PROTOTYPE_METHOD(constructor_template, "memOTFTextureAlloc", Module::OTFTextureAlloc);
      NODE_SET_PROTOTYPE_METHOD(constructor_template, "memBlockTextureAlloc", Module::BlockTextureAlloc);

}
Handle<Value> Module::New(const Arguments& args) {
      HandleScope scope;

      Module *pmem = new Module();
      pmem->Wrap(args.This());

      return args.This();
}

Handle<Value> Module::Load(const Arguments& args) {
      HandleScope scope;
      Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
      Module *pmodule = ObjectWrap::Unwrap<Module>(result);

      String::AsciiValue fname(args[0]);
      CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);

      result->Set(String::New("fname"), args[0]);
      result->Set(String::New("error"), Integer::New(error));

      return scope.Close(result);
}
void Module::volumeTextureLoad(unsigned char * h_data, int dim[3], Module *pmodule){

       CUDA_ARRAY3D_DESCRIPTOR desc;
       desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
       desc.NumChannels = 1;
       desc.Width = dim[0];
       desc.Height = dim[1];
       desc.Depth =  dim[2];
       desc.Flags=0;

       cuArray3DCreate(&(pmodule->cu_volumeArray), &desc);

       CUDA_MEMCPY3D copyParam;
       memset(&copyParam, 0, sizeof(copyParam));
       copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
       copyParam.srcHost = h_data;
       copyParam.srcPitch =  dim[0] * sizeof(unsigned char);
       copyParam.srcHeight =  dim[1];
       copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
       copyParam.dstArray = pmodule->cu_volumeArray;
       copyParam.dstHeight= dim[1];
       copyParam.WidthInBytes =  dim[0] * sizeof(unsigned char);
       copyParam.Height =  dim[1];
       copyParam.Depth =  dim[2];

       cuMemcpy3D(&copyParam);

       CUtexref m_cu_volumeref;
       cuTexRefCreate(&(m_cu_volumeref));
       cuModuleGetTexRef(&(m_cu_volumeref), pmodule->m_module, "tex");
       cuTexRefSetArray(m_cu_volumeref, pmodule->cu_volumeArray, CU_TRSA_OVERRIDE_FORMAT);
       cuTexRefSetAddressMode(m_cu_volumeref, 0, CU_TR_ADDRESS_MODE_BORDER);
       cuTexRefSetAddressMode(m_cu_volumeref, 1, CU_TR_ADDRESS_MODE_BORDER);
       cuTexRefSetFilterMode(m_cu_volumeref, CU_TR_FILTER_MODE_LINEAR);
       cuTexRefSetFlags(m_cu_volumeref, CU_TRSF_NORMALIZED_COORDINATES);
       cuTexRefSetFormat(m_cu_volumeref, CU_AD_FORMAT_UNSIGNED_INT8, 1);

}
void Module::BlockTextureLoad(unsigned char *blockVolume, int dim[3], Module *pmodule){

       size_t size = dim[0] * dim[1] *dim[2] * sizeof(unsigned char);

       CUDA_ARRAY3D_DESCRIPTOR desc;
       desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
       desc.NumChannels = 1;
       desc.Width = dim[0];
       desc.Height = dim[1];
       desc.Depth = dim[2];
       desc.Flags=0;

       cuArray3DCreate(&(pmodule->cu_blockArray), &desc);

       CUDA_MEMCPY3D copyParam;
       memset(&copyParam, 0, sizeof(copyParam));
       copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
       copyParam.srcHost = blockVolume;
       copyParam.srcPitch = dim[0] * sizeof(unsigned char);
       copyParam.srcHeight = dim[1];
       copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
       copyParam.dstArray = pmodule->cu_blockArray;
       copyParam.dstHeight=dim[1];
       copyParam.WidthInBytes = dim[0] * sizeof(unsigned char);
       copyParam.Height = dim[1];
       copyParam.Depth = dim[2];

       cuMemcpy3D(&copyParam);

       CUtexref m_cu_blockref;
       cuTexRefCreate(&(m_cu_blockref));
       cuModuleGetTexRef(&(m_cu_blockref), pmodule->m_module, "tex_block");
       cuTexRefSetArray(m_cu_blockref, pmodule->cu_blockArray, CU_TRSA_OVERRIDE_FORMAT);
       cuTexRefSetAddressMode(m_cu_blockref, 0, CU_TR_ADDRESS_MODE_BORDER);
       cuTexRefSetAddressMode(m_cu_blockref, 1, CU_TR_ADDRESS_MODE_BORDER);
       cuTexRefSetFilterMode(m_cu_blockref, CU_TR_FILTER_MODE_LINEAR);
       cuTexRefSetFlags(m_cu_blockref, CU_TRSF_NORMALIZED_COORDINATES);
       cuTexRefSetFormat(m_cu_blockref, CU_AD_FORMAT_UNSIGNED_INT8, 1);

}
void Module::otfTableTextureLoad(TF *input_float_1D, unsigned int otf_size, Module *pmodule){

       CUDA_ARRAY_DESCRIPTOR ad;
       ad.Format = CU_AD_FORMAT_FLOAT;
       ad.Width = otf_size;
       ad.Height = 1;
       ad.NumChannels = 4;
       cuArrayCreate(&(pmodule->otf_array), &ad);

       // Copy the host input to the array
       cuMemcpyHtoA(pmodule->otf_array,0,input_float_1D,otf_size*sizeof(float4));

       // Texture Binding
       CUtexref m_cu_tf2Dref;

       cuTexRefCreate(&(m_cu_tf2Dref));
       cuModuleGetTexRef(&(m_cu_tf2Dref), pmodule->m_module, "texture_float_1D");
       cuTexRefSetFilterMode(m_cu_tf2Dref, CU_TR_FILTER_MODE_POINT);
       cuTexRefSetAddressMode(m_cu_tf2Dref, 0, CU_TR_ADDRESS_MODE_CLAMP);
       cuTexRefSetFlags(m_cu_tf2Dref, CU_TRSF_READ_AS_INTEGER);
       cuTexRefSetFormat(m_cu_tf2Dref, CU_AD_FORMAT_FLOAT, 4);
       cuTexRefSetArray(m_cu_tf2Dref, pmodule->otf_array, CU_TRSA_OVERRIDE_FORMAT);
}
Handle<Value> Module::VolumeTextureAlloc(const Arguments& args) {
  
       HandleScope scope;
       Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
       Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());

       /*volume binding*/
       int dim[3];
       v8::String::Utf8Value param1(args[0]->ToString());
       char *filename = *param1;

       dim[0]= args[1]->Uint32Value();
       dim[1]= args[2]->Uint32Value();
       dim[2]= args[3]->Uint32Value();

       Volume *m_vol = new Volume;
       m_vol->SetVolume(dim, filename);

       unsigned char * h_data = m_vol->GetDensityPointer();
       int *dimension = m_vol->GetDimension();

       volumeTextureLoad(h_data, dimension, pmodule);   //volume Texture Generation

       delete m_vol;

       return scope.Close(result);
}
Handle<Value> Module::BlockTextureAlloc(const Arguments& args) {

       HandleScope scope;
       Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
       Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());

        /*Blockvolume binding*/
       int dim[3];
       Local<Object> buf = args[0]->ToObject();
       unsigned char *blockVolume = (unsigned char *)Buffer::Data(buf);

       dim[0]= args[1]->Uint32Value();
       dim[1]= args[2]->Uint32Value();
       dim[2]= args[3]->Uint32Value();

       BlockTextureLoad(blockVolume, dim, pmodule);  //otf 생성.

       return scope.Close(result);
}
Handle<Value> Module::OTFTextureAlloc(const Arguments& args) {

       HandleScope scope;
       Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
       Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());

       /*OTF binding*/
       unsigned int tf_start   =  args[0]->Uint32Value();
       unsigned int tf_middle1 =  args[1]->Uint32Value();
       unsigned int tf_middle2 =  args[2]->Uint32Value();
       unsigned int tf_end     =  args[3]->Uint32Value();
       unsigned int tf_size    =  256;

       TransferFunction *m_TF_table = new TransferFunction;

       if(tf_start<0 || tf_middle1<0 || tf_middle2<0 || tf_end<0 ||
          tf_start>255 || tf_start>255 || tf_start>255 || tf_end>255){

            m_TF_table->Set_TF_table(pre_tf_start, pre_tf_middle1, pre_tf_middle2, pre_tf_end, tf_size);

            TF *input_float_1D = m_TF_table->GetTablePointer();
            int real_tf_size = m_TF_table->GetTableSize();

            otfTableTextureLoad(input_float_1D, real_tf_size, pmodule);  //otf 생성.

       }else{

           m_TF_table->Set_TF_table(tf_start, tf_middle1, tf_middle2, tf_end, tf_size);

           TF *input_float_1D = m_TF_table->GetTablePointer();
           int real_tf_size = m_TF_table->GetTableSize();

           otfTableTextureLoad(input_float_1D, real_tf_size, pmodule);  //otf 생성.
       }

       pre_tf_start = tf_start;
       pre_tf_middle1 = tf_middle1;
       pre_tf_middle2 = tf_middle2;
       pre_tf_end = tf_end;

       delete m_TF_table;

       return scope.Close(result);
}

Handle<Value> Module::GetFunction(const Arguments& args) {
      HandleScope scope;
      Local<Object> result = NodeCuda::Function::constructor_template->InstanceTemplate()->NewInstance();
      Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());
      NodeCuda::Function *pfunction = ObjectWrap::Unwrap<NodeCuda::Function>(result);

      String::AsciiValue name(args[0]);
      CUresult error = cuModuleGetFunction(&(pfunction->m_function), pmodule->m_module, *name);

      result->Set(String::New("name"), args[0]);
      result->Set(String::New("error"), Integer::New(error));

      return scope.Close(result);
}
Handle<Value> Module::DestroyTexRef(const Arguments& args) {

      HandleScope scope;
      Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
      Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());

      cuArrayDestroy(pmodule->cu_volumeArray);
      cuArrayDestroy(pmodule->cu_blockArray);
      cuArrayDestroy(pmodule->otf_array);

      return scope.Close(result);
}
Handle<Value> Module::DestroyOtfTexRef(const Arguments& args) {

      HandleScope scope;
      Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
      Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());

      cuArrayDestroy(pmodule->otf_array);

      return scope.Close(result);
}

