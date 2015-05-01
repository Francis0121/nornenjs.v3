S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 11571
Date: 2015-04-14 16:47:48+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 2
      invalid permissions for mapped object
      si_addr = 0xb0e3f77c

Register Information
r0   = 0xbea32a30, r1   = 0x00000001
r2   = 0xb0e3f77c, r3   = 0x000027e6
r4   = 0xbea32a30, r5   = 0x00000002
r6   = 0xb723aaf0, r7   = 0x00000446
r8   = 0xb5cd7644, r9   = 0x000027e7
r10  = 0x0000000a, fp   = 0xb70fc308
ip   = 0xb5cd76b0, sp   = 0xbea32760
lr   = 0xb5cbddc9, pc   = 0xb5cbeed8
cpsr = 0x20000030

Memory Information
MemTotal:   797840 KB
MemFree:    512508 KB
Buffers:     13808 KB
Cached:      84852 KB
VmPeak:     149228 KB
VmSize:     119412 KB
VmLck:           0 KB
VmHWM:       16352 KB
VmRSS:       16352 KB
VmData:      63508 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       25000 KB
VmPTE:          90 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 11571 TID = 11571
11571 11575 11576 11577 11578 11579 11580 11581 

Maps Information
b181a000 b181b000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b1858000 b1859000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b1861000 b1868000 r-xp /usr/lib/libfeedback.so.0.1.4
b187b000 b187c000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b1884000 b189b000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b1a21000 b1a26000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b326f000 b32ba000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b32c3000 b32cd000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b32d6000 b3332000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3b3f000 b3b45000 r-xp /usr/lib/libUMP.so
b3b4d000 b3b60000 r-xp /usr/lib/libEGL_platform.so
b3b69000 b3c40000 r-xp /usr/lib/libMali.so
b3c4b000 b3c62000 r-xp /usr/lib/libEGL.so.1.4
b3c6b000 b3c70000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3c79000 b3c7a000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3c83000 b3c9b000 r-xp /usr/lib/libpng12.so.0.50.0
b3ca3000 b3ce1000 r-xp /usr/lib/libGLESv2.so.2.0
b3ce9000 b3ced000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3cf6000 b3cf8000 r-xp /usr/lib/libdri2.so.0.0.0
b3d00000 b3d07000 r-xp /usr/lib/libdrm.so.2.4.0
b3d10000 b3dc6000 r-xp /usr/lib/libcairo.so.2.11200.14
b3dd1000 b3de7000 r-xp /usr/lib/libtts.so
b3df0000 b3df7000 r-xp /usr/lib/libtbm.so.1.0.0
b3dff000 b3e04000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3e0c000 b3e1d000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3e25000 b3e2c000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3e34000 b3e39000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3e41000 b3e43000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3e4b000 b3e53000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3e5b000 b3e5e000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3e67000 b3f25000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3f2f000 b3f39000 r-xp /lib/libnss_files-2.13.so
b3f47000 b3f49000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b4152000 b4173000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b417c000 b4199000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b41a2000 b4270000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b4287000 b42ad000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b42b7000 b42b9000 r-xp /usr/lib/libiniparser.so.0
b42c3000 b42c9000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b42d2000 b42d8000 r-xp /usr/lib/libappsvc.so.0.1.0
b42e1000 b42e3000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b42ec000 b42f0000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b42f8000 b42fc000 r-xp /usr/lib/libogg.so.0.7.1
b4304000 b4326000 r-xp /usr/lib/libvorbis.so.0.4.3
b432e000 b4412000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b4426000 b4457000 r-xp /usr/lib/libFLAC.so.8.2.0
b4460000 b4462000 r-xp /usr/lib/libXau.so.6.0.0
b446a000 b44b6000 r-xp /usr/lib/libssl.so.1.0.0
b44c3000 b44f1000 r-xp /usr/lib/libidn.so.11.5.44
b44f9000 b4503000 r-xp /usr/lib/libcares.so.2.1.0
b450b000 b4550000 r-xp /usr/lib/libsndfile.so.1.0.25
b455e000 b4565000 r-xp /usr/lib/libsensord-share.so
b456d000 b4583000 r-xp /lib/libexpat.so.1.5.2
b4591000 b4594000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b459c000 b45d0000 r-xp /usr/lib/libicule.so.51.1
b45d9000 b45ec000 r-xp /usr/lib/libxcb.so.1.1.0
b45f4000 b462f000 r-xp /usr/lib/libcurl.so.4.3.0
b4638000 b4641000 r-xp /usr/lib/libethumb.so.1.7.99
b5baf000 b5c43000 r-xp /usr/lib/libstdc++.so.6.0.16
b5c56000 b5c58000 r-xp /usr/lib/libctxdata.so.0.0.0
b5c60000 b5c6d000 r-xp /usr/lib/libremix.so.0.0.0
b5c75000 b5c76000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5c7e000 b5c95000 r-xp /usr/lib/liblua-5.1.so
b5c9e000 b5ca5000 r-xp /usr/lib/libembryo.so.1.7.99
b5cad000 b5cd0000 r-xp /usr/lib/libjpeg.so.8.0.2
b5ce8000 b5cfe000 r-xp /usr/lib/libsensor.so.1.1.0
b5d07000 b5d5d000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5d6a000 b5d8d000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5d96000 b5ddc000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5de5000 b5df8000 r-xp /usr/lib/libfribidi.so.0.3.1
b5e00000 b5e50000 r-xp /usr/lib/libfreetype.so.6.8.1
b5e5b000 b5e5e000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5e66000 b5e6a000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5e72000 b5e77000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5e80000 b5e8a000 r-xp /usr/lib/libXext.so.6.4.0
b5e92000 b5f73000 r-xp /usr/lib/libX11.so.6.3.0
b5f7e000 b5f81000 r-xp /usr/lib/libXtst.so.6.1.0
b5f89000 b5f8f000 r-xp /usr/lib/libXrender.so.1.3.0
b5f97000 b5f9c000 r-xp /usr/lib/libXrandr.so.2.2.0
b5fa4000 b5fa5000 r-xp /usr/lib/libXinerama.so.1.0.0
b5fae000 b5fb6000 r-xp /usr/lib/libXi.so.6.1.0
b5fb7000 b5fba000 r-xp /usr/lib/libXfixes.so.3.1.0
b5fc2000 b5fc4000 r-xp /usr/lib/libXgesture.so.7.0.0
b5fcc000 b5fce000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5fd6000 b5fd7000 r-xp /usr/lib/libXdamage.so.1.1.0
b5fe0000 b5fe6000 r-xp /usr/lib/libXcursor.so.1.0.2
b5fef000 b6008000 r-xp /usr/lib/libecore_con.so.1.7.99
b6012000 b6018000 r-xp /usr/lib/libecore_imf.so.1.7.99
b6020000 b6028000 r-xp /usr/lib/libethumb_client.so.1.7.99
b6030000 b6034000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b603d000 b6053000 r-xp /usr/lib/libefreet.so.1.7.99
b605c000 b6065000 r-xp /usr/lib/libedbus.so.1.7.99
b606d000 b6152000 r-xp /usr/lib/libicuuc.so.51.1
b6167000 b62a6000 r-xp /usr/lib/libicui18n.so.51.1
b62b6000 b6312000 r-xp /usr/lib/libedje.so.1.7.99
b631c000 b632d000 r-xp /usr/lib/libecore_input.so.1.7.99
b6335000 b633a000 r-xp /usr/lib/libecore_file.so.1.7.99
b6342000 b635b000 r-xp /usr/lib/libeet.so.1.7.99
b636c000 b6370000 r-xp /usr/lib/libappcore-common.so.1.1
b6379000 b6445000 r-xp /usr/lib/libevas.so.1.7.99
b646a000 b648b000 r-xp /usr/lib/libecore_evas.so.1.7.99
b6494000 b64c3000 r-xp /usr/lib/libecore_x.so.1.7.99
b64cd000 b6601000 r-xp /usr/lib/libelementary.so.1.7.99
b6619000 b661a000 r-xp /usr/lib/libjournal.so.0.1.0
b6623000 b66ee000 r-xp /usr/lib/libxml2.so.2.7.8
b66fc000 b670c000 r-xp /lib/libresolv-2.13.so
b6710000 b6726000 r-xp /lib/libz.so.1.2.5
b672e000 b6730000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b6738000 b673d000 r-xp /usr/lib/libffi.so.5.0.10
b6746000 b6747000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b674f000 b6752000 r-xp /lib/libattr.so.1.1.0
b675a000 b6902000 r-xp /usr/lib/libcrypto.so.1.0.0
b6922000 b693c000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b6945000 b69ae000 r-xp /lib/libm-2.13.so
b69b7000 b69f7000 r-xp /usr/lib/libeina.so.1.7.99
b6a00000 b6a08000 r-xp /usr/lib/libvconf.so.0.2.45
b6a10000 b6a13000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b6a1b000 b6a4f000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b6a58000 b6b2c000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6b38000 b6b3e000 r-xp /lib/librt-2.13.so
b6b47000 b6b4c000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6b55000 b6b5c000 r-xp /lib/libcrypt-2.13.so
b6b8c000 b6b8f000 r-xp /lib/libcap.so.2.21
b6b97000 b6b99000 r-xp /usr/lib/libiri.so
b6ba1000 b6bc0000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6bc8000 b6bde000 r-xp /usr/lib/libecore.so.1.7.99
b6bf4000 b6bf9000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6c02000 b6cd2000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6cd3000 b6ce1000 r-xp /usr/lib/libail.so.0.1.0
b6ce9000 b6d00000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6d09000 b6d13000 r-xp /lib/libunwind.so.8.0.1
b6d41000 b6e5c000 r-xp /lib/libc-2.13.so
b6e6a000 b6e72000 r-xp /lib/libgcc_s-4.6.4.so.1
b6e7a000 b6ea4000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6ead000 b6eb0000 r-xp /usr/lib/libbundle.so.0.1.22
b6eb8000 b6eba000 r-xp /lib/libdl-2.13.so
b6ec3000 b6ec6000 r-xp /usr/lib/libsmack.so.1.0.0
b6ece000 b6f30000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6f3a000 b6f4c000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6f55000 b6f69000 r-xp /lib/libpthread-2.13.so
b6f76000 b6f7a000 r-xp /usr/lib/libappcore-efl.so.1.1
b6f84000 b6f86000 r-xp /usr/lib/libdlog.so.0.0.0
b6f8e000 b6f99000 r-xp /usr/lib/libaul.so.0.1.0
b6fa3000 b6fa7000 r-xp /usr/lib/libsys-assert.so
b6fb0000 b6fcd000 r-xp /lib/ld-2.13.so
b6fd6000 b6fdc000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6fe4000 b7010000 rw-p [heap]
b7010000 b729c000 rw-p [heap]
bea13000 bea34000 rwxp [stack]
End of Maps Information

Callstack Information (PID:11571)
Call Stack Count: 23
 0: (0xb5cbeed8) [/usr/lib/libjpeg.so.8] + 0x11ed8
 1: (0xb5cbddc9) [/usr/lib/libjpeg.so.8] + 0x10dc9
 2: jpeg_consume_input + 0x3e (0xb5cbac4b) [/usr/lib/libjpeg.so.8] + 0xdc4b
 3: jpeg_read_header + 0x24 (0xb5cbadbd) [/usr/lib/libjpeg.so.8] + 0xddbd
 4: (0xb3e3518d) [/usr/lib/libmmutil_jpeg.so.0] + 0x118d
 5: mm_util_decode_from_jpeg_memory + 0x50 (0xb3e37a29) [/usr/lib/libmmutil_jpeg.so.0] + 0x3a29
 6: image_util_decode_jpeg_from_memory + 0x38 (0xb3e5d299) [/usr/lib/libcapi-media-image-util.so.0] + 0x2299
 7: draw_gl + 0x7a (0xb3e70353) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9353
 8: (0xb6565553) [/usr/lib/libelementary.so.1] + 0x98553
 9: (0xb63a18e9) [/usr/lib/libevas.so.1] + 0x288e9
10: (0xb63d1fa3) [/usr/lib/libevas.so.1] + 0x58fa3
11: (0xb63d30a1) [/usr/lib/libevas.so.1] + 0x5a0a1
12: (0xb647b049) [/usr/lib/libecore_evas.so.1] + 0x11049
13: (0xb64787f5) [/usr/lib/libecore_evas.so.1] + 0xe7f5
14: (0xb6bd1ebd) [/usr/lib/libecore.so.1] + 0x9ebd
15: (0xb6bd32e7) [/usr/lib/libecore.so.1] + 0xb2e7
16: ecore_main_loop_begin + 0x30 (0xb6bd3851) [/usr/lib/libecore.so.1] + 0xb851
17: appcore_efl_main + 0x2a2 (0xb6f786f3) [/usr/lib/libappcore-efl.so.1] + 0x26f3
18: ui_app_main + 0xb0 (0xb4592c79) [/usr/lib/libcapi-appfw-application.so.0] + 0x1c79
19: main + 0x1a2 (0xb3e70deb) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9deb
20: (0xb6fd8c6f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2c6f
21: __libc_start_main + 0x114 (0xb6d5882c) [/lib/libc.so.6] + 0x1782c
22: (0xb6fd935c) [/opt/usr/apps/org.tizen.other/bin/other] + 0x335c
End of Call Stack

Package Information
Package Name: org.tizen.other
Package ID : org.tizen.other
Version: 1.0.0
Package Type: coretpk
App Name: other
App ID: org.tizen.other
Type: capp
Categories: 

Latest Debug Message Information
--------- beginning of /dev/log_main
0 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-14 16:47:47.175+0900 D/sio_packet(11571): from json
04-14 16:47:47.175+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.175+0900 D/sio_packet(11571): from json
04-14 16:47:47.175+0900 D/sio_packet(11571): from json
04-14 16:47:47.175+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.175+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.175+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.175+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.195+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.210+0900 D/sio_packet(11571): from json
04-14 16:47:47.210+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.210+0900 D/sio_packet(11571): from json
04-14 16:47:47.210+0900 D/sio_packet(11571): from json
04-14 16:47:47.210+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.210+0900 D/sio_packet(11571): from json
04-14 16:47:47.210+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.210+0900 D/sio_packet(11571): from json
04-14 16:47:47.210+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.210+0900 D/sio_packet(11571): from json
04-14 16:47:47.210+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.210+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.210+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.210+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.210+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.215+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.225+0900 D/sio_packet(11571): from json
04-14 16:47:47.225+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.225+0900 D/sio_packet(11571): from json
04-14 16:47:47.225+0900 D/sio_packet(11571): from json
04-14 16:47:47.225+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.225+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.225+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.225+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.230+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.250+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.255+0900 D/sio_packet(11571): from json
04-14 16:47:47.255+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.255+0900 D/sio_packet(11571): from json
04-14 16:47:47.255+0900 D/sio_packet(11571): from json
04-14 16:47:47.255+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.255+0900 D/sio_packet(11571): from json
04-14 16:47:47.255+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.255+0900 D/sio_packet(11571): from json
04-14 16:47:47.255+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.255+0900 D/sio_packet(11571): from json
04-14 16:47:47.255+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.255+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.255+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.255+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.255+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.265+0900 D/sio_packet(11571): from json
04-14 16:47:47.265+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.265+0900 D/sio_packet(11571): from json
04-14 16:47:47.265+0900 D/sio_packet(11571): from json
04-14 16:47:47.265+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.265+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.265+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.265+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.270+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.285+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.300+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.310+0900 D/sio_packet(11571): from json
04-14 16:47:47.310+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.310+0900 D/sio_packet(11571): from json
04-14 16:47:47.310+0900 D/sio_packet(11571): from json
04-14 16:47:47.310+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.310+0900 D/sio_packet(11571): from json
04-14 16:47:47.310+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.310+0900 D/sio_packet(11571): from json
04-14 16:47:47.310+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.310+0900 D/sio_packet(11571): from json
04-14 16:47:47.310+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.310+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.310+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.310+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.310+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.315+0900 D/sio_packet(11571): from json
04-14 16:47:47.315+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.315+0900 D/sio_packet(11571): from json
04-14 16:47:47.315+0900 D/sio_packet(11571): from json
04-14 16:47:47.315+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.315+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.315+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.315+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.320+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.340+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.355+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.360+0900 D/sio_packet(11571): from json
04-14 16:47:47.360+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.360+0900 D/sio_packet(11571): from json
04-14 16:47:47.360+0900 D/sio_packet(11571): from json
04-14 16:47:47.360+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.360+0900 D/sio_packet(11571): from json
04-14 16:47:47.360+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.360+0900 D/sio_packet(11571): from json
04-14 16:47:47.360+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.360+0900 D/sio_packet(11571): from json
04-14 16:47:47.360+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.360+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.360+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.360+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.360+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.365+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-14 16:47:47.365+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-14 16:47:47.375+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-14 16:47:47.375+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-14 16:47:47.385+0900 D/sio_packet(11571): from json
04-14 16:47:47.385+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.385+0900 D/sio_packet(11571): from json
04-14 16:47:47.385+0900 D/sio_packet(11571): from json
04-14 16:47:47.385+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.385+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.385+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.385+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.400+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.405+0900 D/sio_packet(11571): from json
04-14 16:47:47.405+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.405+0900 D/sio_packet(11571): from json
04-14 16:47:47.405+0900 D/sio_packet(11571): from json
04-14 16:47:47.405+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.405+0900 D/sio_packet(11571): from json
04-14 16:47:47.405+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.405+0900 D/sio_packet(11571): from json
04-14 16:47:47.405+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.405+0900 D/sio_packet(11571): from json
04-14 16:47:47.405+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.405+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.405+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.405+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.405+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.405+0900 D/sio_packet(11571): from json
04-14 16:47:47.405+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.405+0900 D/sio_packet(11571): from json
04-14 16:47:47.405+0900 D/sio_packet(11571): from json
04-14 16:47:47.405+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.405+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.405+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.405+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.415+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.435+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.450+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.455+0900 D/sio_packet(11571): from json
04-14 16:47:47.455+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.455+0900 D/sio_packet(11571): from json
04-14 16:47:47.455+0900 D/sio_packet(11571): from json
04-14 16:47:47.455+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.455+0900 D/sio_packet(11571): from json
04-14 16:47:47.455+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.455+0900 D/sio_packet(11571): from json
04-14 16:47:47.455+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.455+0900 D/sio_packet(11571): from json
04-14 16:47:47.455+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.455+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.455+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.455+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.455+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.460+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-14 16:47:47.460+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-14 16:47:47.475+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-14 16:47:47.475+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-14 16:47:47.490+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-14 16:47:47.495+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-14 16:47:47.500+0900 D/sio_packet(11571): from json
04-14 16:47:47.500+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.500+0900 D/sio_packet(11571): from json
04-14 16:47:47.500+0900 D/sio_packet(11571): from json
04-14 16:47:47.500+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.500+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.500+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.500+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.515+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.520+0900 D/sio_packet(11571): from json
04-14 16:47:47.520+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.520+0900 D/sio_packet(11571): from json
04-14 16:47:47.520+0900 D/sio_packet(11571): from json
04-14 16:47:47.520+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.520+0900 D/sio_packet(11571): from json
04-14 16:47:47.520+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.520+0900 D/sio_packet(11571): from json
04-14 16:47:47.520+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.520+0900 D/sio_packet(11571): from json
04-14 16:47:47.520+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.520+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.520+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.520+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.520+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.525+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-14 16:47:47.525+0900 D/sio_packet(11571): from json
04-14 16:47:47.525+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.525+0900 D/sio_packet(11571): from json
04-14 16:47:47.525+0900 D/sio_packet(11571): from json
04-14 16:47:47.525+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.525+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.525+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.525+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.525+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-14 16:47:47.550+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.550+0900 D/sio_packet(11571): from json
04-14 16:47:47.550+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.550+0900 D/sio_packet(11571): from json
04-14 16:47:47.550+0900 D/sio_packet(11571): from json
04-14 16:47:47.550+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.550+0900 D/sio_packet(11571): from json
04-14 16:47:47.550+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.550+0900 D/sio_packet(11571): from json
04-14 16:47:47.550+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.550+0900 D/sio_packet(11571): from json
04-14 16:47:47.550+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.550+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.550+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.550+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.550+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.555+0900 D/sio_packet(11571): from json
04-14 16:47:47.555+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.555+0900 D/sio_packet(11571): from json
04-14 16:47:47.555+0900 D/sio_packet(11571): from json
04-14 16:47:47.555+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.555+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.555+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.555+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.570+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.585+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.600+0900 D/sio_packet(11571): from json
04-14 16:47:47.600+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.600+0900 D/sio_packet(11571): from json
04-14 16:47:47.600+0900 D/sio_packet(11571): from json
04-14 16:47:47.600+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.600+0900 D/sio_packet(11571): from json
04-14 16:47:47.600+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.600+0900 D/sio_packet(11571): from json
04-14 16:47:47.600+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.600+0900 D/sio_packet(11571): from json
04-14 16:47:47.600+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.600+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.600+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.600+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.600+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.600+0900 D/sio_packet(11571): from json
04-14 16:47:47.600+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.600+0900 D/sio_packet(11571): from json
04-14 16:47:47.600+0900 D/sio_packet(11571): from json
04-14 16:47:47.600+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.600+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.600+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.600+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.605+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.620+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.635+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.645+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-14 16:47:47.645+0900 D/sio_packet(11571): from json
04-14 16:47:47.645+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.645+0900 D/sio_packet(11571): from json
04-14 16:47:47.645+0900 D/sio_packet(11571): from json
04-14 16:47:47.645+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.645+0900 D/sio_packet(11571): from json
04-14 16:47:47.645+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.645+0900 D/sio_packet(11571): from json
04-14 16:47:47.645+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.645+0900 D/sio_packet(11571): from json
04-14 16:47:47.645+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.645+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.645+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.645+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.645+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.650+0900 D/sio_packet(11571): from json
04-14 16:47:47.650+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.650+0900 D/sio_packet(11571): from json
04-14 16:47:47.650+0900 D/sio_packet(11571): from json
04-14 16:47:47.650+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.650+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.650+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.650+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.650+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-14 16:47:47.670+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.685+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.695+0900 D/sio_packet(11571): from json
04-14 16:47:47.695+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.695+0900 D/sio_packet(11571): from json
04-14 16:47:47.695+0900 D/sio_packet(11571): from json
04-14 16:47:47.695+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.695+0900 D/sio_packet(11571): from json
04-14 16:47:47.695+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.695+0900 D/sio_packet(11571): from json
04-14 16:47:47.695+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.695+0900 D/sio_packet(11571): from json
04-14 16:47:47.695+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.695+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.695+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.695+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.695+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.695+0900 D/sio_packet(11571): from json
04-14 16:47:47.695+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.695+0900 D/sio_packet(11571): from json
04-14 16:47:47.695+0900 D/sio_packet(11571): from json
04-14 16:47:47.695+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.695+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.695+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.695+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.705+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.725+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.740+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.745+0900 D/sio_packet(11571): from json
04-14 16:47:47.745+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.745+0900 D/sio_packet(11571): from json
04-14 16:47:47.745+0900 D/sio_packet(11571): from json
04-14 16:47:47.745+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.745+0900 D/sio_packet(11571): from json
04-14 16:47:47.745+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.745+0900 D/sio_packet(11571): from json
04-14 16:47:47.745+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.745+0900 D/sio_packet(11571): from json
04-14 16:47:47.745+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.745+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.745+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.745+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.745+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.750+0900 D/sio_packet(11571): from json
04-14 16:47:47.750+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.750+0900 D/sio_packet(11571): from json
04-14 16:47:47.750+0900 D/sio_packet(11571): from json
04-14 16:47:47.750+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.750+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.750+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.750+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.755+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.775+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.790+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.795+0900 D/sio_packet(11571): from json
04-14 16:47:47.795+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.795+0900 D/sio_packet(11571): from json
04-14 16:47:47.795+0900 D/sio_packet(11571): from json
04-14 16:47:47.795+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.795+0900 D/sio_packet(11571): from json
04-14 16:47:47.795+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.795+0900 D/sio_packet(11571): from json
04-14 16:47:47.795+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.795+0900 D/sio_packet(11571): from json
04-14 16:47:47.795+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.795+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.795+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.795+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.795+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.795+0900 D/sio_packet(11571): from json
04-14 16:47:47.795+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.795+0900 D/sio_packet(11571): from json
04-14 16:47:47.795+0900 D/sio_packet(11571): from json
04-14 16:47:47.795+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.795+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.795+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.795+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.810+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.825+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.845+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.850+0900 D/sio_packet(11571): from json
04-14 16:47:47.850+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.850+0900 D/sio_packet(11571): from json
04-14 16:47:47.850+0900 D/sio_packet(11571): from json
04-14 16:47:47.850+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.850+0900 D/sio_packet(11571): from json
04-14 16:47:47.850+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.850+0900 D/sio_packet(11571): from json
04-14 16:47:47.850+0900 D/sio_packet(11571): IsInt64
04-14 16:47:47.850+0900 D/sio_packet(11571): from json
04-14 16:47:47.850+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.850+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.850+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.850+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.850+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:47.850+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-14 16:47:47.855+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-14 16:47:47.865+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] NO_SUCH_FILE(0xfffffffe)
04-14 16:47:47.865+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg] ERROR_NONE(0x00000000)
04-14 16:47:47.870+0900 D/sio_packet(11571): from json
04-14 16:47:47.870+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:47.870+0900 D/sio_packet(11571): from json
04-14 16:47:47.870+0900 D/sio_packet(11571): from json
04-14 16:47:47.870+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:47.870+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:47.870+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:47.870+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:47.890+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.905+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.920+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.940+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.955+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.970+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:47.990+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:48.005+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:48.025+0900 D/CAPI_MEDIA_IMAGE_UTIL(11571): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
04-14 16:47:48.030+0900 D/sio_packet(11571): from json
04-14 16:47:48.030+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.030+0900 D/sio_packet(11571): from json
04-14 16:47:48.030+0900 D/sio_packet(11571): from json
04-14 16:47:48.030+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.030+0900 D/sio_packet(11571): from json
04-14 16:47:48.030+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.030+0900 D/sio_packet(11571): from json
04-14 16:47:48.030+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.030+0900 D/sio_packet(11571): from json
04-14 16:47:48.030+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.030+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.030+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.030+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.030+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.035+0900 D/sio_packet(11571): from json
04-14 16:47:48.035+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.035+0900 D/sio_packet(11571): from json
04-14 16:47:48.035+0900 D/sio_packet(11571): from json
04-14 16:47:48.035+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.035+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.035+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.035+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.040+0900 D/sio_packet(11571): from json
04-14 16:47:48.040+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.040+0900 D/sio_packet(11571): from json
04-14 16:47:48.040+0900 D/sio_packet(11571): from json
04-14 16:47:48.040+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.040+0900 D/sio_packet(11571): from json
04-14 16:47:48.040+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.040+0900 D/sio_packet(11571): from json
04-14 16:47:48.040+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.040+0900 D/sio_packet(11571): from json
04-14 16:47:48.040+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.040+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.040+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.040+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.040+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.040+0900 D/sio_packet(11571): from json
04-14 16:47:48.040+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.040+0900 D/sio_packet(11571): from json
04-14 16:47:48.040+0900 D/sio_packet(11571): from json
04-14 16:47:48.040+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.040+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.040+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.040+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.045+0900 D/sio_packet(11571): from json
04-14 16:47:48.045+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.045+0900 D/sio_packet(11571): from json
04-14 16:47:48.045+0900 D/sio_packet(11571): from json
04-14 16:47:48.045+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.045+0900 D/sio_packet(11571): from json
04-14 16:47:48.045+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.045+0900 D/sio_packet(11571): from json
04-14 16:47:48.045+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.045+0900 D/sio_packet(11571): from json
04-14 16:47:48.045+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.045+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.045+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.045+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.045+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.045+0900 D/sio_packet(11571): from json
04-14 16:47:48.045+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.045+0900 D/sio_packet(11571): from json
04-14 16:47:48.045+0900 D/sio_packet(11571): from json
04-14 16:47:48.045+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.045+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.045+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.045+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.050+0900 D/sio_packet(11571): from json
04-14 16:47:48.050+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.050+0900 D/sio_packet(11571): from json
04-14 16:47:48.050+0900 D/sio_packet(11571): from json
04-14 16:47:48.050+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.050+0900 D/sio_packet(11571): from json
04-14 16:47:48.050+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.050+0900 D/sio_packet(11571): from json
04-14 16:47:48.050+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.050+0900 D/sio_packet(11571): from json
04-14 16:47:48.050+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.050+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.050+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.050+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.050+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.055+0900 D/sio_packet(11571): from json
04-14 16:47:48.055+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.055+0900 D/sio_packet(11571): from json
04-14 16:47:48.055+0900 D/sio_packet(11571): from json
04-14 16:47:48.055+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.055+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.055+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.055+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.090+0900 D/sio_packet(11571): from json
04-14 16:47:48.090+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.090+0900 D/sio_packet(11571): from json
04-14 16:47:48.090+0900 D/sio_packet(11571): from json
04-14 16:47:48.090+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.090+0900 D/sio_packet(11571): from json
04-14 16:47:48.090+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.090+0900 D/sio_packet(11571): from json
04-14 16:47:48.090+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.090+0900 D/sio_packet(11571): from json
04-14 16:47:48.090+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.090+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.090+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.090+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.090+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.095+0900 D/sio_packet(11571): from json
04-14 16:47:48.095+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.095+0900 D/sio_packet(11571): from json
04-14 16:47:48.095+0900 D/sio_packet(11571): from json
04-14 16:47:48.095+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.095+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.095+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.095+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.140+0900 D/sio_packet(11571): from json
04-14 16:47:48.140+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.140+0900 D/sio_packet(11571): from json
04-14 16:47:48.140+0900 D/sio_packet(11571): from json
04-14 16:47:48.140+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.140+0900 D/sio_packet(11571): from json
04-14 16:47:48.140+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.140+0900 D/sio_packet(11571): from json
04-14 16:47:48.140+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.140+0900 D/sio_packet(11571): from json
04-14 16:47:48.140+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.140+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.140+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.140+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.140+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.165+0900 D/sio_packet(11571): from json
04-14 16:47:48.165+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.165+0900 D/sio_packet(11571): from json
04-14 16:47:48.165+0900 D/sio_packet(11571): from json
04-14 16:47:48.165+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.165+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.165+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.165+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.190+0900 D/sio_packet(11571): from json
04-14 16:47:48.190+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.190+0900 D/sio_packet(11571): from json
04-14 16:47:48.190+0900 D/sio_packet(11571): from json
04-14 16:47:48.190+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.190+0900 D/sio_packet(11571): from json
04-14 16:47:48.190+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.190+0900 D/sio_packet(11571): from json
04-14 16:47:48.190+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.190+0900 D/sio_packet(11571): from json
04-14 16:47:48.190+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.190+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.190+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.190+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.190+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.195+0900 D/sio_packet(11571): from json
04-14 16:47:48.195+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.195+0900 D/sio_packet(11571): from json
04-14 16:47:48.195+0900 D/sio_packet(11571): from json
04-14 16:47:48.195+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.195+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.195+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.195+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.235+0900 D/sio_packet(11571): from json
04-14 16:47:48.235+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.235+0900 D/sio_packet(11571): from json
04-14 16:47:48.235+0900 D/sio_packet(11571): from json
04-14 16:47:48.235+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.235+0900 D/sio_packet(11571): from json
04-14 16:47:48.235+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.235+0900 D/sio_packet(11571): from json
04-14 16:47:48.235+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.235+0900 D/sio_packet(11571): from json
04-14 16:47:48.235+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.235+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.235+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.235+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.235+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.240+0900 D/sio_packet(11571): from json
04-14 16:47:48.240+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.240+0900 D/sio_packet(11571): from json
04-14 16:47:48.240+0900 D/sio_packet(11571): from json
04-14 16:47:48.240+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.240+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.240+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.240+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.285+0900 D/sio_packet(11571): from json
04-14 16:47:48.285+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.285+0900 D/sio_packet(11571): from json
04-14 16:47:48.285+0900 D/sio_packet(11571): from json
04-14 16:47:48.285+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.285+0900 D/sio_packet(11571): from json
04-14 16:47:48.285+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.285+0900 D/sio_packet(11571): from json
04-14 16:47:48.285+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.285+0900 D/sio_packet(11571): from json
04-14 16:47:48.285+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.285+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.285+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.285+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.285+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.285+0900 D/sio_packet(11571): from json
04-14 16:47:48.285+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.285+0900 D/sio_packet(11571): from json
04-14 16:47:48.285+0900 D/sio_packet(11571): from json
04-14 16:47:48.285+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.285+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.285+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.285+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.350+0900 D/sio_packet(11571): from json
04-14 16:47:48.350+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.350+0900 D/sio_packet(11571): from json
04-14 16:47:48.350+0900 D/sio_packet(11571): from json
04-14 16:47:48.350+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.350+0900 D/sio_packet(11571): from json
04-14 16:47:48.350+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.350+0900 D/sio_packet(11571): from json
04-14 16:47:48.350+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.350+0900 D/sio_packet(11571): from json
04-14 16:47:48.350+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.350+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.350+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.350+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.350+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.350+0900 D/sio_packet(11571): from json
04-14 16:47:48.350+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.350+0900 D/sio_packet(11571): from json
04-14 16:47:48.350+0900 D/sio_packet(11571): from json
04-14 16:47:48.350+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.350+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.350+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.350+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.385+0900 D/sio_packet(11571): from json
04-14 16:47:48.385+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.385+0900 D/sio_packet(11571): from json
04-14 16:47:48.385+0900 D/sio_packet(11571): from json
04-14 16:47:48.385+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.385+0900 D/sio_packet(11571): from json
04-14 16:47:48.385+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.385+0900 D/sio_packet(11571): from json
04-14 16:47:48.385+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.385+0900 D/sio_packet(11571): from json
04-14 16:47:48.385+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.385+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.385+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.385+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.385+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.385+0900 D/sio_packet(11571): from json
04-14 16:47:48.385+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.385+0900 D/sio_packet(11571): from json
04-14 16:47:48.385+0900 D/sio_packet(11571): from json
04-14 16:47:48.385+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.385+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.385+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.385+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.440+0900 D/sio_packet(11571): from json
04-14 16:47:48.440+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.440+0900 D/sio_packet(11571): from json
04-14 16:47:48.440+0900 D/sio_packet(11571): from json
04-14 16:47:48.440+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.440+0900 D/sio_packet(11571): from json
04-14 16:47:48.440+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.440+0900 D/sio_packet(11571): from json
04-14 16:47:48.440+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.440+0900 D/sio_packet(11571): from json
04-14 16:47:48.440+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.440+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.440+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.440+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.440+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.445+0900 D/sio_packet(11571): from json
04-14 16:47:48.445+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.445+0900 D/sio_packet(11571): from json
04-14 16:47:48.445+0900 D/sio_packet(11571): from json
04-14 16:47:48.445+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.445+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.445+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.445+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.485+0900 D/sio_packet(11571): from json
04-14 16:47:48.485+0900 D/value.IsArray()(11571): if arr from json
04-14 16:47:48.485+0900 D/sio_packet(11571): from json
04-14 16:47:48.485+0900 D/sio_packet(11571): from json
04-14 16:47:48.485+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.485+0900 D/sio_packet(11571): from json
04-14 16:47:48.485+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.485+0900 D/sio_packet(11571): from json
04-14 16:47:48.485+0900 D/sio_packet(11571): IsInt64
04-14 16:47:48.485+0900 D/sio_packet(11571): from json
04-14 16:47:48.485+0900 D/value.IsObject()(11571): if binary from json
04-14 16:47:48.485+0900 F/value.IsObject()(11571): num!!!!!0
04-14 16:47:48.485+0900 F/value.IsObject()(11571): buf size!!!!!1
04-14 16:47:48.485+0900 E/socket.io(11571): 669: Received Message type(Event)
04-14 16:47:48.485+0900 F/get_binary(11571): in get binary_message()...
04-14 16:47:48.535+0900 W/CRASH_MANAGER(11665): worker.c: worker_job(1189) > 11115716f7468142899766
