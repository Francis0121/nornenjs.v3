cmd_Release/obj.target/jpeg.node := flock ./Release/linker.lock g++ -shared -pthread -rdynamic -m32  -Wl,-soname=jpeg.node -o Release/obj.target/jpeg.node -Wl,--start-group Release/obj.target/jpeg/src/common.o Release/obj.target/jpeg/src/jpeg_encoder.o Release/obj.target/jpeg/src/jpeg.o Release/obj.target/jpeg/src/fixed_jpeg_stack.o Release/obj.target/jpeg/src/dynamic_jpeg_stack.o Release/obj.target/jpeg/src/module.o -Wl,--end-group -ljpeg
