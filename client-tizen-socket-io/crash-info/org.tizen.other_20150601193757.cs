S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 20671
Date: 2015-06-01 19:37:57+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xad15d6a8

Register Information
r0   = 0xaccdda00, r1   = 0xad157688
r2   = 0x00000c00, r3   = 0xad15d6a8
r4   = 0x000001b0, r5   = 0xad157688
r6   = 0xaccdd800, r7   = 0x0000028a
r8   = 0x00000c00, r9   = 0xad157008
r10  = 0x00000300, fp   = 0x000000d0
ip   = 0x00000000, sp   = 0xbeef97e8
lr   = 0xb3acb924, pc   = 0xb3acbae4
cpsr = 0x20000010

Memory Information
MemTotal:   797840 KB
MemFree:    406436 KB
Buffers:     38464 KB
Cached:     132216 KB
VmPeak:     175180 KB
VmSize:     137008 KB
VmLck:           0 KB
VmHWM:       36576 KB
VmRSS:       33816 KB
VmData:      81444 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24904 KB
VmPTE:         112 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 20671 TID = 20671
20671 20675 20676 20677 20678 20679 20680 20681 

Maps Information
b17a6000 b17a7000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b17af000 b17b6000 r-xp /usr/lib/libfeedback.so.0.1.4
b17c9000 b17ca000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b17d2000 b17e9000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b196f000 b1974000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b31bd000 b3208000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b3211000 b321b000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b3224000 b3280000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3a8d000 b3a93000 r-xp /usr/lib/libUMP.so
b3a9b000 b3aae000 r-xp /usr/lib/libEGL_platform.so
b3ab7000 b3b8e000 r-xp /usr/lib/libMali.so
b3b99000 b3bb0000 r-xp /usr/lib/libEGL.so.1.4
b3bb9000 b3bbe000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3bc7000 b3bc8000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3bd1000 b3be9000 r-xp /usr/lib/libpng12.so.0.50.0
b3bf1000 b3c2f000 r-xp /usr/lib/libGLESv2.so.2.0
b3c37000 b3c3b000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3c44000 b3c46000 r-xp /usr/lib/libdri2.so.0.0.0
b3c4e000 b3c55000 r-xp /usr/lib/libdrm.so.2.4.0
b3c5e000 b3d14000 r-xp /usr/lib/libcairo.so.2.11200.14
b3d1f000 b3d35000 r-xp /usr/lib/libtts.so
b3d3e000 b3d45000 r-xp /usr/lib/libtbm.so.1.0.0
b3d4d000 b3d52000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3d5a000 b3d6b000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3d73000 b3d7a000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3d82000 b3d87000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3d8f000 b3d91000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3d99000 b3da1000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3da9000 b3dac000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3db5000 b3e5c000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3e67000 b3e71000 r-xp /lib/libnss_files-2.13.so
b3e7f000 b3e81000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b408a000 b40ab000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b40b4000 b40d1000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b40da000 b41a8000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b41bf000 b41e5000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b41ef000 b41f1000 r-xp /usr/lib/libiniparser.so.0
b41fb000 b4201000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b420a000 b4210000 r-xp /usr/lib/libappsvc.so.0.1.0
b4219000 b421b000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b4224000 b4228000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b4230000 b4234000 r-xp /usr/lib/libogg.so.0.7.1
b423c000 b425e000 r-xp /usr/lib/libvorbis.so.0.4.3
b4266000 b434a000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b435e000 b438f000 r-xp /usr/lib/libFLAC.so.8.2.0
b4398000 b439a000 r-xp /usr/lib/libXau.so.6.0.0
b43a2000 b43ee000 r-xp /usr/lib/libssl.so.1.0.0
b43fb000 b4429000 r-xp /usr/lib/libidn.so.11.5.44
b4431000 b443b000 r-xp /usr/lib/libcares.so.2.1.0
b4443000 b4488000 r-xp /usr/lib/libsndfile.so.1.0.25
b4496000 b449d000 r-xp /usr/lib/libsensord-share.so
b44a5000 b44bb000 r-xp /lib/libexpat.so.1.5.2
b44c9000 b44cc000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b44d4000 b4508000 r-xp /usr/lib/libicule.so.51.1
b4511000 b4524000 r-xp /usr/lib/libxcb.so.1.1.0
b452c000 b4567000 r-xp /usr/lib/libcurl.so.4.3.0
b4570000 b4579000 r-xp /usr/lib/libethumb.so.1.7.99
b5ae7000 b5b7b000 r-xp /usr/lib/libstdc++.so.6.0.16
b5b8e000 b5b90000 r-xp /usr/lib/libctxdata.so.0.0.0
b5b98000 b5ba5000 r-xp /usr/lib/libremix.so.0.0.0
b5bad000 b5bae000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5bb6000 b5bcd000 r-xp /usr/lib/liblua-5.1.so
b5bd6000 b5bdd000 r-xp /usr/lib/libembryo.so.1.7.99
b5be5000 b5c08000 r-xp /usr/lib/libjpeg.so.8.0.2
b5c20000 b5c36000 r-xp /usr/lib/libsensor.so.1.1.0
b5c3f000 b5c95000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5ca2000 b5cc5000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5cce000 b5d14000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5d1d000 b5d30000 r-xp /usr/lib/libfribidi.so.0.3.1
b5d38000 b5d88000 r-xp /usr/lib/libfreetype.so.6.8.1
b5d93000 b5d96000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5d9e000 b5da2000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5daa000 b5daf000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5db8000 b5dc2000 r-xp /usr/lib/libXext.so.6.4.0
b5dca000 b5eab000 r-xp /usr/lib/libX11.so.6.3.0
b5eb6000 b5eb9000 r-xp /usr/lib/libXtst.so.6.1.0
b5ec1000 b5ec7000 r-xp /usr/lib/libXrender.so.1.3.0
b5ecf000 b5ed4000 r-xp /usr/lib/libXrandr.so.2.2.0
b5edc000 b5edd000 r-xp /usr/lib/libXinerama.so.1.0.0
b5ee6000 b5eee000 r-xp /usr/lib/libXi.so.6.1.0
b5eef000 b5ef2000 r-xp /usr/lib/libXfixes.so.3.1.0
b5efa000 b5efc000 r-xp /usr/lib/libXgesture.so.7.0.0
b5f04000 b5f06000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5f0e000 b5f0f000 r-xp /usr/lib/libXdamage.so.1.1.0
b5f18000 b5f1e000 r-xp /usr/lib/libXcursor.so.1.0.2
b5f27000 b5f40000 r-xp /usr/lib/libecore_con.so.1.7.99
b5f4a000 b5f50000 r-xp /usr/lib/libecore_imf.so.1.7.99
b5f58000 b5f60000 r-xp /usr/lib/libethumb_client.so.1.7.99
b5f68000 b5f6c000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b5f75000 b5f8b000 r-xp /usr/lib/libefreet.so.1.7.99
b5f94000 b5f9d000 r-xp /usr/lib/libedbus.so.1.7.99
b5fa5000 b608a000 r-xp /usr/lib/libicuuc.so.51.1
b609f000 b61de000 r-xp /usr/lib/libicui18n.so.51.1
b61ee000 b624a000 r-xp /usr/lib/libedje.so.1.7.99
b6254000 b6265000 r-xp /usr/lib/libecore_input.so.1.7.99
b626d000 b6272000 r-xp /usr/lib/libecore_file.so.1.7.99
b627a000 b6293000 r-xp /usr/lib/libeet.so.1.7.99
b62a4000 b62a8000 r-xp /usr/lib/libappcore-common.so.1.1
b62b1000 b637d000 r-xp /usr/lib/libevas.so.1.7.99
b63a2000 b63c3000 r-xp /usr/lib/libecore_evas.so.1.7.99
b63cc000 b63fb000 r-xp /usr/lib/libecore_x.so.1.7.99
b6405000 b6539000 r-xp /usr/lib/libelementary.so.1.7.99
b6551000 b6552000 r-xp /usr/lib/libjournal.so.0.1.0
b655b000 b6626000 r-xp /usr/lib/libxml2.so.2.7.8
b6634000 b6644000 r-xp /lib/libresolv-2.13.so
b6648000 b665e000 r-xp /lib/libz.so.1.2.5
b6666000 b6668000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b6670000 b6675000 r-xp /usr/lib/libffi.so.5.0.10
b667e000 b667f000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b6687000 b668a000 r-xp /lib/libattr.so.1.1.0
b6692000 b683a000 r-xp /usr/lib/libcrypto.so.1.0.0
b685a000 b6874000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b687d000 b68e6000 r-xp /lib/libm-2.13.so
b68ef000 b692f000 r-xp /usr/lib/libeina.so.1.7.99
b6938000 b6940000 r-xp /usr/lib/libvconf.so.0.2.45
b6948000 b694b000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b6953000 b6987000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b6990000 b6a64000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6a70000 b6a76000 r-xp /lib/librt-2.13.so
b6a7f000 b6a84000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6a8d000 b6a94000 r-xp /lib/libcrypt-2.13.so
b6ac4000 b6ac7000 r-xp /lib/libcap.so.2.21
b6acf000 b6ad1000 r-xp /usr/lib/libiri.so
b6ad9000 b6af8000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6b00000 b6b16000 r-xp /usr/lib/libecore.so.1.7.99
b6b2c000 b6b31000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6b3a000 b6c0a000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6c0b000 b6c19000 r-xp /usr/lib/libail.so.0.1.0
b6c21000 b6c38000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6c41000 b6c4b000 r-xp /lib/libunwind.so.8.0.1
b6c79000 b6d94000 r-xp /lib/libc-2.13.so
b6da2000 b6daa000 r-xp /lib/libgcc_s-4.6.4.so.1
b6db2000 b6ddc000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6de5000 b6de8000 r-xp /usr/lib/libbundle.so.0.1.22
b6df0000 b6df2000 r-xp /lib/libdl-2.13.so
b6dfb000 b6dfe000 r-xp /usr/lib/libsmack.so.1.0.0
b6e06000 b6e68000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6e72000 b6e84000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6e8d000 b6ea1000 r-xp /lib/libpthread-2.13.so
b6eae000 b6eb2000 r-xp /usr/lib/libappcore-efl.so.1.1
b6ebc000 b6ebe000 r-xp /usr/lib/libdlog.so.0.0.0
b6ec6000 b6ed1000 r-xp /usr/lib/libaul.so.0.1.0
b6edb000 b6edf000 r-xp /usr/lib/libsys-assert.so
b6ee8000 b6f05000 r-xp /lib/ld-2.13.so
b6f0e000 b6f14000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6f1c000 b6f48000 rw-p [heap]
b6f48000 b7848000 rw-p [heap]
beeda000 beefb000 rwxp [stack]
End of Maps Information

Callstack Information (PID:20671)
Call Stack Count: 1
 0: (0xb3acbae4) [/usr/lib/libMali.so] + 0x14ae4
End of Call Stack

Package Information
Package Name: org.tizen.other
Package ID : org.tizen.other
Version: 1.0.0
Package Type: coretpk
App Name: Nornenjs
App ID: org.tizen.other
Type: capp
Categories: 

Latest Debug Message Information
--------- beginning of /dev/log_main
: 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:51.695+0900 F/sio_packet(20671): accept()
06-01 19:37:51.695+0900 E/socket.io(20671): 743: encoded paylod length: 77
06-01 19:37:51.695+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:51.705+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:51.705+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:51.710+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:51.715+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:51.715+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:51.715+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:51.720+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:51.730+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60982756 button=1
06-01 19:37:51.730+0900 F/sio_packet(20671): accept()
06-01 19:37:51.735+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:51.735+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:51.810+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:51.810+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:51.825+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:51.920+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60982937 button=1
06-01 19:37:51.920+0900 F/sio_packet(20671): accept()
06-01 19:37:51.920+0900 E/socket.io(20671): 743: encoded paylod length: 79
06-01 19:37:51.920+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:51.950+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:51.950+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:51.950+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:51.955+0900 F/sio_packet(20671): accept()
06-01 19:37:51.955+0900 E/socket.io(20671): 743: encoded paylod length: 79
06-01 19:37:51.955+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:51.975+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:51.975+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:51.980+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:51.985+0900 F/sio_packet(20671): accept()
06-01 19:37:51.985+0900 E/socket.io(20671): 743: encoded paylod length: 79
06-01 19:37:51.985+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.005+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.005+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.010+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.020+0900 F/sio_packet(20671): accept()
06-01 19:37:52.020+0900 E/socket.io(20671): 743: encoded paylod length: 79
06-01 19:37:52.020+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.040+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.040+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.040+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:52.040+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.055+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60983076 button=1
06-01 19:37:52.055+0900 F/sio_packet(20671): accept()
06-01 19:37:52.055+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:52.055+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.140+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.140+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.155+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.235+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60983242 button=1
06-01 19:37:52.235+0900 F/sio_packet(20671): accept()
06-01 19:37:52.235+0900 E/socket.io(20671): 743: encoded paylod length: 79
06-01 19:37:52.235+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.265+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.265+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.265+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.270+0900 F/sio_packet(20671): accept()
06-01 19:37:52.270+0900 E/socket.io(20671): 743: encoded paylod length: 79
06-01 19:37:52.270+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.285+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.285+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.290+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.300+0900 F/sio_packet(20671): accept()
06-01 19:37:52.300+0900 E/socket.io(20671): 743: encoded paylod length: 79
06-01 19:37:52.300+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.320+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.320+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.320+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.325+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:52.340+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60983362 button=1
06-01 19:37:52.340+0900 F/sio_packet(20671): accept()
06-01 19:37:52.340+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:52.340+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.420+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.420+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.435+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.530+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60983551 button=1
06-01 19:37:52.555+0900 F/sio_packet(20671): accept()
06-01 19:37:52.560+0900 E/socket.io(20671): 743: encoded paylod length: 79
06-01 19:37:52.560+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.585+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.585+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.585+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.590+0900 F/sio_packet(20671): accept()
06-01 19:37:52.590+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:52.590+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.620+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.620+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.620+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.625+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:52.630+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60983660 button=1
06-01 19:37:52.630+0900 F/sio_packet(20671): accept()
06-01 19:37:52.630+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:52.630+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.710+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.710+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.730+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.815+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60983837 button=1
06-01 19:37:52.815+0900 F/sio_packet(20671): accept()
06-01 19:37:52.815+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:52.815+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.840+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.840+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.845+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.845+0900 F/sio_packet(20671): accept()
06-01 19:37:52.845+0900 E/socket.io(20671): 743: encoded paylod length: 82
06-01 19:37:52.845+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.865+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.865+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.865+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.880+0900 F/sio_packet(20671): accept()
06-01 19:37:52.880+0900 E/socket.io(20671): 743: encoded paylod length: 82
06-01 19:37:52.885+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.900+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:52.905+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.905+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:52.910+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:52.915+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60983935 button=1
06-01 19:37:52.915+0900 F/sio_packet(20671): accept()
06-01 19:37:52.915+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:52.920+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:52.995+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:52.995+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.010+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.080+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60984102 button=1
06-01 19:37:53.110+0900 F/sio_packet(20671): accept()
06-01 19:37:53.110+0900 E/socket.io(20671): 743: encoded paylod length: 82
06-01 19:37:53.115+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.145+0900 F/sio_packet(20671): accept()
06-01 19:37:53.145+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:53.145+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.160+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.160+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.160+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.165+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:53.175+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.175+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.175+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.180+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60984199 button=1
06-01 19:37:53.180+0900 F/sio_packet(20671): accept()
06-01 19:37:53.180+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:53.180+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.260+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.260+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.280+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.345+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60984377 button=1
06-01 19:37:53.375+0900 F/sio_packet(20671): accept()
06-01 19:37:53.375+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:53.380+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.405+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.405+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.410+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.415+0900 F/sio_packet(20671): accept()
06-01 19:37:53.415+0900 E/socket.io(20671): 743: encoded paylod length: 77
06-01 19:37:53.415+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.425+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:53.435+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60984463 button=1
06-01 19:37:53.435+0900 F/sio_packet(20671): accept()
06-01 19:37:53.435+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.435+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.435+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:53.440+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.440+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.520+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.520+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.540+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.650+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60984652 button=1
06-01 19:37:53.650+0900 F/sio_packet(20671): accept()
06-01 19:37:53.650+0900 E/socket.io(20671): 743: encoded paylod length: 77
06-01 19:37:53.650+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.680+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.680+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.680+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.685+0900 F/sio_packet(20671): accept()
06-01 19:37:53.690+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:53.690+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.705+0900 F/sio_packet(20671): accept()
06-01 19:37:53.705+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:53.705+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.710+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.710+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.710+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.725+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.725+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.725+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:53.730+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.730+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60984761 button=1
06-01 19:37:53.730+0900 F/sio_packet(20671): accept()
06-01 19:37:53.730+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:53.735+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.805+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.805+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.825+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.925+0900 D/RESOURCED( 2375): counter-process.c: check_net_blocked(99) > [check_net_blocked,99] net_blocked 0, state 0
06-01 19:37:53.925+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60984928 button=1
06-01 19:37:53.925+0900 F/sio_packet(20671): accept()
06-01 19:37:53.925+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:53.925+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.950+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.950+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.955+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.955+0900 F/sio_packet(20671): accept()
06-01 19:37:53.955+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:53.955+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:53.970+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:53.970+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:53.970+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:53.990+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:53.995+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60985025 button=1
06-01 19:37:53.995+0900 F/sio_packet(20671): accept()
06-01 19:37:53.995+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:53.995+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.070+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.070+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.085+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.185+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60985217 button=1
06-01 19:37:54.215+0900 F/sio_packet(20671): accept()
06-01 19:37:54.215+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:54.215+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.245+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.245+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.245+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.255+0900 F/sio_packet(20671): accept()
06-01 19:37:54.255+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:54.255+0900 F/sio_packet(20671): accept()
06-01 19:37:54.255+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.255+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:54.255+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.270+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.270+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.270+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.280+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.280+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.285+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.295+0900 F/sio_packet(20671): accept()
06-01 19:37:54.295+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:54.295+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.310+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:54.315+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.315+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.315+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60985345 button=1
06-01 19:37:54.320+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.320+0900 F/sio_packet(20671): accept()
06-01 19:37:54.320+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:54.320+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.390+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.390+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.405+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.505+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60985523 button=1
06-01 19:37:54.530+0900 F/sio_packet(20671): accept()
06-01 19:37:54.535+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:54.535+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.560+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.560+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.565+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.565+0900 F/sio_packet(20671): accept()
06-01 19:37:54.565+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:54.565+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.585+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.585+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.585+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:54.585+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.590+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60985621 button=1
06-01 19:37:54.590+0900 F/sio_packet(20671): accept()
06-01 19:37:54.590+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:54.590+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.670+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.670+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.685+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.775+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60985801 button=1
06-01 19:37:54.805+0900 F/sio_packet(20671): accept()
06-01 19:37:54.805+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:54.805+0900 F/sio_packet(20671): accept()
06-01 19:37:54.805+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.805+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:54.810+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.830+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.830+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.830+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.850+0900 F/sio_packet(20671): accept()
06-01 19:37:54.850+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:54.850+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.850+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.855+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.855+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.865+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.870+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.870+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.875+0900 F/sio_packet(20671): accept()
06-01 19:37:54.875+0900 E/socket.io(20671): 743: encoded paylod length: 81
06-01 19:37:54.875+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.895+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.895+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:54.895+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:54.895+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:54.910+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60985930 button=1
06-01 19:37:54.910+0900 F/sio_packet(20671): accept()
06-01 19:37:54.910+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:54.910+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:54.985+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:54.985+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.005+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.070+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60986096 button=1
06-01 19:37:55.100+0900 F/sio_packet(20671): accept()
06-01 19:37:55.100+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:55.105+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.130+0900 F/sio_packet(20671): accept()
06-01 19:37:55.135+0900 E/socket.io(20671): 743: encoded paylod length: 76
06-01 19:37:55.135+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.145+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.150+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.155+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.160+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:55.160+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.165+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.165+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.170+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60986194 button=1
06-01 19:37:55.170+0900 F/sio_packet(20671): accept()
06-01 19:37:55.170+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:55.170+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.245+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.245+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.265+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.335+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60986352 button=1
06-01 19:37:55.335+0900 F/sio_packet(20671): accept()
06-01 19:37:55.335+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:55.335+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.365+0900 F/sio_packet(20671): accept()
06-01 19:37:55.365+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:55.365+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.380+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.380+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.380+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.390+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.390+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.395+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.395+0900 F/sio_packet(20671): accept()
06-01 19:37:55.400+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:55.400+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.415+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.415+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.420+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.435+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:55.435+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60986469 button=1
06-01 19:37:55.435+0900 F/sio_packet(20671): accept()
06-01 19:37:55.435+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:55.435+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.510+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.510+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.525+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.615+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60986647 button=1
06-01 19:37:55.645+0900 F/sio_packet(20671): accept()
06-01 19:37:55.645+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:55.645+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.675+0900 F/sio_packet(20671): accept()
06-01 19:37:55.675+0900 E/socket.io(20671): 743: encoded paylod length: 76
06-01 19:37:55.675+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.690+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.690+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.695+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.695+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:55.705+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.705+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.705+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.710+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60986734 button=1
06-01 19:37:55.710+0900 F/sio_packet(20671): accept()
06-01 19:37:55.710+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:55.710+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.785+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.785+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.805+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.880+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60986900 button=1
06-01 19:37:55.915+0900 F/sio_packet(20671): accept()
06-01 19:37:55.915+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:55.915+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.940+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.940+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.940+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.945+0900 F/sio_packet(20671): accept()
06-01 19:37:55.945+0900 E/socket.io(20671): 743: encoded paylod length: 76
06-01 19:37:55.945+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.960+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.960+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.960+0900 F/sio_packet(20671): accept()
06-01 19:37:55.960+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:55.960+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:55.965+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.970+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:55.975+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60987009 button=1
06-01 19:37:55.975+0900 F/sio_packet(20671): accept()
06-01 19:37:55.975+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:55.975+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:55.985+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:55.985+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:55.985+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.060+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.060+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.075+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.175+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60987198 button=1
06-01 19:37:56.205+0900 F/sio_packet(20671): accept()
06-01 19:37:56.205+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:56.205+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.235+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.235+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.240+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.240+0900 F/sio_packet(20671): accept()
06-01 19:37:56.245+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:56.245+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.260+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:56.260+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.260+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.265+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.265+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60987295 button=1
06-01 19:37:56.265+0900 F/sio_packet(20671): accept()
06-01 19:37:56.270+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:56.270+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.340+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.340+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.360+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.445+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60987476 button=1
06-01 19:37:56.475+0900 F/sio_packet(20671): accept()
06-01 19:37:56.475+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:56.475+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.505+0900 F/sio_packet(20671): accept()
06-01 19:37:56.505+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:56.505+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.515+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.515+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.520+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.530+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.530+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.530+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.535+0900 F/sio_packet(20671): accept()
06-01 19:37:56.535+0900 E/socket.io(20671): 743: encoded paylod length: 76
06-01 19:37:56.535+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.555+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.555+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.560+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.570+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:56.580+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60987604 button=1
06-01 19:37:56.580+0900 F/sio_packet(20671): accept()
06-01 19:37:56.580+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:56.580+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.660+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.660+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.675+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.775+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60987796 button=1
06-01 19:37:56.775+0900 F/sio_packet(20671): accept()
06-01 19:37:56.775+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:56.775+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.800+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.800+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.805+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.805+0900 F/sio_packet(20671): accept()
06-01 19:37:56.805+0900 E/socket.io(20671): 743: encoded paylod length: 76
06-01 19:37:56.805+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.820+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.820+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.825+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.835+0900 F/sio_packet(20671): accept()
06-01 19:37:56.835+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:56.835+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.855+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.855+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.860+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:56.865+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:56.870+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=60987902 button=1
06-01 19:37:56.870+0900 F/sio_packet(20671): accept()
06-01 19:37:56.870+0900 E/socket.io(20671): 743: encoded paylod length: 21
06-01 19:37:56.875+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:56.945+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:56.945+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:56.960+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:57.060+0900 D/EFL     (20671): ecore_x<20671> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=60988082 button=1
06-01 19:37:57.090+0900 F/sio_packet(20671): accept()
06-01 19:37:57.090+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:57.090+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:57.120+0900 F/sio_packet(20671): accept()
06-01 19:37:57.120+0900 E/socket.io(20671): 743: encoded paylod length: 78
06-01 19:37:57.125+0900 E/socket.io(20671): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 19:37:57.135+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:57.135+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:57.140+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:57.140+0900 D/PROCESSMGR( 2148): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600032  register trigger_timer!  pointed_win=0x658c36 
06-01 19:37:57.150+0900 E/socket.io(20671): 669: Received Message type(Event)
06-01 19:37:57.150+0900 F/get_binary(20671): in get binary_message()...
06-01 19:37:57.150+0900 D/CAPI_MEDIA_IMAGE_UTIL(20671): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 19:37:57.195+0900 W/CRASH_MANAGER(20655): worker.c: worker_job(1189) > 11206716f7468143315507
