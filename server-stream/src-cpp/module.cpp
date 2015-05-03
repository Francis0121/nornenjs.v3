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
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "memVolumeTextureAlloc", Module::VolumeTextureAlloc);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "memOTFTextureAlloc", Module::OTFTextureAlloc);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "memOTF2DTextureAlloc", Module::OTF2DTextureAlloc);
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

   CUarray cu_array;
   CUDA_ARRAY3D_DESCRIPTOR desc;
   desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
   desc.NumChannels = 1;
   desc.Width = width;
   desc.Height = height;
   desc.Depth = depth;
   desc.Flags=0;

   cuArray3DCreate(&cu_array, &desc);

   CUDA_MEMCPY3D copyParam;
   memset(&copyParam, 0, sizeof(copyParam));
   copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
   copyParam.srcHost = h_data;
   copyParam.srcPitch = width * sizeof(unsigned char);
   copyParam.srcHeight = height;
   copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
   copyParam.dstArray = cu_array;
   copyParam.dstHeight=height;
   copyParam.WidthInBytes = width * sizeof(unsigned char);
   copyParam.Height = height;
   copyParam.Depth = depth;

   cuMemcpy3D(&copyParam);

   CUtexref cu_texref;
   cuModuleGetTexRef(&cu_texref, pmodule->m_module, "tex");
   cuTexRefSetArray(cu_texref, cu_array, CU_TRSA_OVERRIDE_FORMAT);
   cuTexRefSetAddressMode(cu_texref, 0, CU_TR_ADDRESS_MODE_BORDER);
   cuTexRefSetAddressMode(cu_texref, 1, CU_TR_ADDRESS_MODE_BORDER);
   cuTexRefSetFilterMode(cu_texref, CU_TR_FILTER_MODE_LINEAR);
   cuTexRefSetFlags(cu_texref, CU_TRSF_NORMALIZED_COORDINATES);
   cuTexRefSetFormat(cu_texref, CU_AD_FORMAT_UNSIGNED_INT8, 1);

   free(h_data);
}
void Module::otf2DTableTextureLoad(char *TF2d, unsigned int otf_size, Module *pmodule){

   size_t size = otf_size * otf_size * sizeof(float4);

   CUarray otf_2Darray;
   CUDA_ARRAY3D_DESCRIPTOR desc_tf;
   desc_tf.Format = CU_AD_FORMAT_FLOAT;
   desc_tf.NumChannels = 4;
   desc_tf.Width = otf_size;
   desc_tf.Height = otf_size;
   desc_tf.Depth = 1;
   desc_tf.Flags=0;

   cuArray3DCreate(&otf_2Darray, &desc_tf);

   CUDA_MEMCPY3D copyParam;
   memset(&copyParam, 0, sizeof(copyParam));
   copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
   copyParam.srcHost = TF2d;
   copyParam.srcPitch = otf_size * sizeof(float4);
   copyParam.srcHeight = otf_size;
   copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
   copyParam.dstArray = otf_2Darray;
   copyParam.dstHeight= otf_size;
   copyParam.WidthInBytes = otf_size * sizeof(float4);
   copyParam.Height = otf_size;
   copyParam.Depth = 1;

   cuMemcpy3D(&copyParam);

   CUtexref cu_tf2Dref;
   cuModuleGetTexRef(&cu_tf2Dref, pmodule->m_module, "tex_TF2d");
   cuTexRefSetArray(cu_tf2Dref, otf_2Darray, CU_TRSA_OVERRIDE_FORMAT);
   cuTexRefSetAddressMode(cu_tf2Dref, 0, CU_TR_ADDRESS_MODE_BORDER);
   cuTexRefSetAddressMode(cu_tf2Dref, 1, CU_TR_ADDRESS_MODE_BORDER);
   cuTexRefSetAddressMode(cu_tf2Dref, 2, CU_TR_ADDRESS_MODE_BORDER);
   cuTexRefSetFilterMode(cu_tf2Dref, CU_TR_FILTER_MODE_LINEAR);
   cuTexRefSetFlags(cu_tf2Dref, CU_TRSF_NORMALIZED_COORDINATES);
   cuTexRefSetFormat(cu_tf2Dref, CU_AD_FORMAT_FLOAT, 4);

}
void Module::otfTableTextureLoad(float4 *input_float_1D, unsigned int otf_size, Module *pmodule){
   // Create the array on the device
   CUarray otf_array;
   CUDA_ARRAY_DESCRIPTOR ad;
   ad.Format = CU_AD_FORMAT_FLOAT;
   ad.Width = otf_size;
   ad.Height = 1;
   ad.NumChannels = 4;
   cuArrayCreate(&otf_array, &ad);

   // Copy the host input to the array
   cuMemcpyHtoA(otf_array,0,input_float_1D,otf_size*sizeof(float4));

   // Texture Binding
   CUtexref otf_texref;
   cuModuleGetTexRef(&otf_texref, pmodule->m_module, "texture_float_1D");
   cuTexRefSetFilterMode(otf_texref, CU_TR_FILTER_MODE_POINT);
   cuTexRefSetAddressMode(otf_texref, 0, CU_TR_ADDRESS_MODE_CLAMP);
   cuTexRefSetFlags(otf_texref, CU_TRSF_READ_AS_INTEGER);
   cuTexRefSetFormat(otf_texref, CU_AD_FORMAT_FLOAT, 4);
   cuTexRefSetArray(otf_texref, otf_array, CU_TRSA_OVERRIDE_FORMAT);
}

float4 *Module::getOTFtable(unsigned int tf_start, unsigned int tf_middle1, unsigned int tf_middle2, unsigned int tf_end, unsigned int tf_size){

    float4 *otf_table= (float4 *)malloc(sizeof(float4)*tf_size);
    memset(otf_table, 0, sizeof(float4)*tf_size);

	for(int i=tf_start+1; i<=tf_middle1; i++){
		otf_table[i].x = (1.0f / ((float)tf_end-(float)tf_start)) * ( i - (float)tf_start);
		otf_table[i].y = (1.0f / ((float)tf_end-(float)tf_start)) * ( i - (float)tf_start);
		otf_table[i].z = (1.0f / ((float)tf_end-(float)tf_start)) * ( i - (float)tf_start);
		otf_table[i].w = (1.0f / ((float)tf_end-(float)tf_start)) * ( i - (float)tf_start);
	}
	for(int i=tf_middle1+1; i<=tf_middle2; i++){
    	otf_table[i].w =1.0f;
    	otf_table[i].x =1.0f;
    	otf_table[i].y =1.0f;
    	otf_table[i].z =1.0f;
    }
    for(int i=tf_middle2+1; i<=tf_end; i++){
    	otf_table[i].w = (1.0 / ((float)tf_end-(float)tf_middle2)) * ((float)tf_end -i);
    	otf_table[i].x = (1.0 / ((float)tf_end-(float)tf_middle2)) * ((float)tf_end -i);
    	otf_table[i].y = (1.0 / ((float)tf_end-(float)tf_middle2)) * ((float)tf_end -i);
    	otf_table[i].z = (1.0 / ((float)tf_end-(float)tf_middle2)) * ((float)tf_end -i);
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
Handle<Value> Module::OTF2DTextureAlloc(const Arguments& args) {

   HandleScope scope;
   Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
   Module *pmodule = ObjectWrap::Unwrap<Module>(args.This());

   /*OTF binding*/
   Local<Object> buf = args[0]->ToObject();
   char *TF2d = Buffer::Data(buf);
   unsigned int tf_size =  args[1]->Uint32Value();

   otf2DTableTextureLoad(TF2d, tf_size, pmodule);  //otf 생성.
   //free(TF2d);

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

