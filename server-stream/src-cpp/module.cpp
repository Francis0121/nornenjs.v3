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
void Module::volumeTextureLoad(unsigned int width, unsigned int height, unsigned int depth, char * filename, Module *pmodule){

       size_t size = width * height *depth * sizeof(unsigned char);

       FILE *fp = fopen(filename, "rb");
       void *h_data = (void *) malloc(size);

       size_t read = fread(h_data, 1, size, fp);
       fclose(fp);

       CUDA_ARRAY3D_DESCRIPTOR desc;
       desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
       desc.NumChannels = 1;
       desc.Width = width;
       desc.Height = height;
       desc.Depth = depth;
       desc.Flags=0;

       cuArray3DCreate(&(pmodule->cu_volumeArray), &desc);

       CUDA_MEMCPY3D copyParam;
       memset(&copyParam, 0, sizeof(copyParam));
       copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
       copyParam.srcHost = h_data;
       copyParam.srcPitch = width * sizeof(unsigned char);
       copyParam.srcHeight = height;
       copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
       copyParam.dstArray = pmodule->cu_volumeArray;
       copyParam.dstHeight=height;
       copyParam.WidthInBytes = width * sizeof(unsigned char);
       copyParam.Height = height;
       copyParam.Depth = depth;

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

       free(h_data);
}
void Module::BlockTextureLoad(char *blockVolume, unsigned int width, unsigned int height, unsigned int depth, Module *pmodule){

       size_t size = width * height *depth * sizeof(unsigned char);

       CUDA_ARRAY3D_DESCRIPTOR desc;
       desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
       desc.NumChannels = 1;
       desc.Width = width;
       desc.Height = height;
       desc.Depth = depth;
       desc.Flags=0;

       cuArray3DCreate(&(pmodule->cu_blockArray), &desc);

       CUDA_MEMCPY3D copyParam;
       memset(&copyParam, 0, sizeof(copyParam));
       copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
       copyParam.srcHost = blockVolume;
       copyParam.srcPitch = width * sizeof(unsigned char);
       copyParam.srcHeight = height;
       copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
       copyParam.dstArray = pmodule->cu_blockArray;
       copyParam.dstHeight=height;
       copyParam.WidthInBytes = width * sizeof(unsigned char);
       copyParam.Height = height;
       copyParam.Depth = depth;

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
void Module::otfTableTextureLoad(float4 *input_float_1D, unsigned int otf_size, Module *pmodule){
       // Create the array on the device
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
float4 *Module::getOTFtable(unsigned int tf_start, unsigned int tf_middle1, unsigned int tf_middle2, unsigned int tf_end, unsigned int tf_size){

        float4 *otf_table= (float4 *)malloc(sizeof(float4)*tf_size);
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
        return otf_table;
}
Handle<Value> Module::VolumeTextureAlloc(const Arguments& args) {
  
       HandleScope scope;
       Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
       Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());

       /*volume binding*/
       v8::String::Utf8Value param1(args[0]->ToString());
       char *filename = *param1;

       unsigned int width = args[1]->Uint32Value();
       unsigned int height = args[2]->Uint32Value();
       unsigned int depth = args[3]->Uint32Value();

       volumeTextureLoad(width, height, depth, filename, pmodule);   //volume Texture 생성
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

       float4 *input_float_1D = NULL;
       input_float_1D =getOTFtable(tf_start, tf_middle1, tf_middle2, tf_end, tf_size);

       otfTableTextureLoad(input_float_1D, tf_size, pmodule);  //otf 생성.
       free(input_float_1D);

       return scope.Close(result);
}
Handle<Value> Module::BlockTextureAlloc(const Arguments& args) {

       HandleScope scope;
       Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
       Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());

       /*OTF binding*/
       Local<Object> buf = args[0]->ToObject();
       char *blockVolume = Buffer::Data(buf);

       unsigned int block_width =  args[1]->Uint32Value();
       unsigned int block_height =  args[2]->Uint32Value();
       unsigned int block_depth =  args[3]->Uint32Value();

       BlockTextureLoad(blockVolume, block_width,block_height,block_depth, pmodule);  //otf 생성.

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

