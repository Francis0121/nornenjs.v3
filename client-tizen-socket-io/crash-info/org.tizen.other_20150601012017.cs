S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 13015
Date: 2015-06-01 01:20:17+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xae680004

Register Information
r0   = 0xae680008, r1   = 0xb3ddf4f3
r2   = 0x000000e4, r3   = 0x00000000
r4   = 0xb3e891cc, r5   = 0xae680008
r6   = 0x00000241, r7   = 0xbeb4a000
r8   = 0xb6edd2e8, r9   = 0xb6edd2e4
r10  = 0x00000000, fp   = 0xb6f34f40
ip   = 0xb3e89244, sp   = 0xbeb49fe8
lr   = 0xb3ddf4f3, pc   = 0xb6d0b150
cpsr = 0xa0000010

Memory Information
MemTotal:   797840 KB
MemFree:    423856 KB
Buffers:     16568 KB
Cached:     112756 KB
VmPeak:     176452 KB
VmSize:     142884 KB
VmLck:           0 KB
VmHWM:       43984 KB
VmRSS:       40052 KB
VmData:      87060 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24920 KB
VmPTE:         110 KB
VmSwap:          0 KB

Threads Information
Threads: 6
PID = 13015 TID = 13015
13015 13020 13021 13022 13024 13025 

Maps Information
b1789000 b178a000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b17c7000 b17c8000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b17d0000 b17d7000 r-xp /usr/lib/libfeedback.so.0.1.4
b17ea000 b17eb000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b17f3000 b180a000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b1990000 b1995000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b31de000 b3229000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b3232000 b323c000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b3245000 b32a1000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3aae000 b3ab4000 r-xp /usr/lib/libUMP.so
b3abc000 b3acf000 r-xp /usr/lib/libEGL_platform.so
b3ad8000 b3baf000 r-xp /usr/lib/libMali.so
b3bba000 b3bd1000 r-xp /usr/lib/libEGL.so.1.4
b3bda000 b3bdf000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3be8000 b3be9000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3bf2000 b3c0a000 r-xp /usr/lib/libpng12.so.0.50.0
b3c12000 b3c50000 r-xp /usr/lib/libGLESv2.so.2.0
b3c58000 b3c5c000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3c65000 b3c67000 r-xp /usr/lib/libdri2.so.0.0.0
b3c6f000 b3c76000 r-xp /usr/lib/libdrm.so.2.4.0
b3c7f000 b3d35000 r-xp /usr/lib/libcairo.so.2.11200.14
b3d40000 b3d56000 r-xp /usr/lib/libtts.so
b3d5f000 b3d66000 r-xp /usr/lib/libtbm.so.1.0.0
b3d6e000 b3d73000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3d7b000 b3d8c000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3d94000 b3d9b000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3da3000 b3da8000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3db0000 b3db2000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3dba000 b3dc2000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3dca000 b3dcd000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3dd6000 b3e80000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3e8a000 b3e94000 r-xp /lib/libnss_files-2.13.so
b3ea2000 b3ea4000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b40ad000 b40ce000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b40d7000 b40f4000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b40fd000 b41cb000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b41e2000 b4208000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b4212000 b4214000 r-xp /usr/lib/libiniparser.so.0
b421e000 b4224000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b422d000 b4233000 r-xp /usr/lib/libappsvc.so.0.1.0
b423c000 b423e000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b4247000 b424b000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b4253000 b4257000 r-xp /usr/lib/libogg.so.0.7.1
b425f000 b4281000 r-xp /usr/lib/libvorbis.so.0.4.3
b4289000 b436d000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b4381000 b43b2000 r-xp /usr/lib/libFLAC.so.8.2.0
b43bb000 b43bd000 r-xp /usr/lib/libXau.so.6.0.0
b43c5000 b4411000 r-xp /usr/lib/libssl.so.1.0.0
b441e000 b444c000 r-xp /usr/lib/libidn.so.11.5.44
b4454000 b445e000 r-xp /usr/lib/libcares.so.2.1.0
b4466000 b44ab000 r-xp /usr/lib/libsndfile.so.1.0.25
b44b9000 b44c0000 r-xp /usr/lib/libsensord-share.so
b44c8000 b44de000 r-xp /lib/libexpat.so.1.5.2
b44ec000 b44ef000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b44f7000 b452b000 r-xp /usr/lib/libicule.so.51.1
b4534000 b4547000 r-xp /usr/lib/libxcb.so.1.1.0
b454f000 b458a000 r-xp /usr/lib/libcurl.so.4.3.0
b4593000 b459c000 r-xp /usr/lib/libethumb.so.1.7.99
b5b0a000 b5b9e000 r-xp /usr/lib/libstdc++.so.6.0.16
b5bb1000 b5bb3000 r-xp /usr/lib/libctxdata.so.0.0.0
b5bbb000 b5bc8000 r-xp /usr/lib/libremix.so.0.0.0
b5bd0000 b5bd1000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5bd9000 b5bf0000 r-xp /usr/lib/liblua-5.1.so
b5bf9000 b5c00000 r-xp /usr/lib/libembryo.so.1.7.99
b5c08000 b5c2b000 r-xp /usr/lib/libjpeg.so.8.0.2
b5c43000 b5c59000 r-xp /usr/lib/libsensor.so.1.1.0
b5c62000 b5cb8000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5cc5000 b5ce8000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5cf1000 b5d37000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5d40000 b5d53000 r-xp /usr/lib/libfribidi.so.0.3.1
b5d5b000 b5dab000 r-xp /usr/lib/libfreetype.so.6.8.1
b5db6000 b5db9000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5dc1000 b5dc5000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5dcd000 b5dd2000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5ddb000 b5de5000 r-xp /usr/lib/libXext.so.6.4.0
b5ded000 b5ece000 r-xp /usr/lib/libX11.so.6.3.0
b5ed9000 b5edc000 r-xp /usr/lib/libXtst.so.6.1.0
b5ee4000 b5eea000 r-xp /usr/lib/libXrender.so.1.3.0
b5ef2000 b5ef7000 r-xp /usr/lib/libXrandr.so.2.2.0
b5eff000 b5f00000 r-xp /usr/lib/libXinerama.so.1.0.0
b5f09000 b5f11000 r-xp /usr/lib/libXi.so.6.1.0
b5f12000 b5f15000 r-xp /usr/lib/libXfixes.so.3.1.0
b5f1d000 b5f1f000 r-xp /usr/lib/libXgesture.so.7.0.0
b5f27000 b5f29000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5f31000 b5f32000 r-xp /usr/lib/libXdamage.so.1.1.0
b5f3b000 b5f41000 r-xp /usr/lib/libXcursor.so.1.0.2
b5f4a000 b5f63000 r-xp /usr/lib/libecore_con.so.1.7.99
b5f6d000 b5f73000 r-xp /usr/lib/libecore_imf.so.1.7.99
b5f7b000 b5f83000 r-xp /usr/lib/libethumb_client.so.1.7.99
b5f8b000 b5f8f000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b5f98000 b5fae000 r-xp /usr/lib/libefreet.so.1.7.99
b5fb7000 b5fc0000 r-xp /usr/lib/libedbus.so.1.7.99
b5fc8000 b60ad000 r-xp /usr/lib/libicuuc.so.51.1
b60c2000 b6201000 r-xp /usr/lib/libicui18n.so.51.1
b6211000 b626d000 r-xp /usr/lib/libedje.so.1.7.99
b6277000 b6288000 r-xp /usr/lib/libecore_input.so.1.7.99
b6290000 b6295000 r-xp /usr/lib/libecore_file.so.1.7.99
b629d000 b62b6000 r-xp /usr/lib/libeet.so.1.7.99
b62c7000 b62cb000 r-xp /usr/lib/libappcore-common.so.1.1
b62d4000 b63a0000 r-xp /usr/lib/libevas.so.1.7.99
b63c5000 b63e6000 r-xp /usr/lib/libecore_evas.so.1.7.99
b63ef000 b641e000 r-xp /usr/lib/libecore_x.so.1.7.99
b6428000 b655c000 r-xp /usr/lib/libelementary.so.1.7.99
b6574000 b6575000 r-xp /usr/lib/libjournal.so.0.1.0
b657e000 b6649000 r-xp /usr/lib/libxml2.so.2.7.8
b6657000 b6667000 r-xp /lib/libresolv-2.13.so
b666b000 b6681000 r-xp /lib/libz.so.1.2.5
b6689000 b668b000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b6693000 b6698000 r-xp /usr/lib/libffi.so.5.0.10
b66a1000 b66a2000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b66aa000 b66ad000 r-xp /lib/libattr.so.1.1.0
b66b5000 b685d000 r-xp /usr/lib/libcrypto.so.1.0.0
b687d000 b6897000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b68a0000 b6909000 r-xp /lib/libm-2.13.so
b6912000 b6952000 r-xp /usr/lib/libeina.so.1.7.99
b695b000 b6963000 r-xp /usr/lib/libvconf.so.0.2.45
b696b000 b696e000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b6976000 b69aa000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b69b3000 b6a87000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6a93000 b6a99000 r-xp /lib/librt-2.13.so
b6aa2000 b6aa7000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6ab0000 b6ab7000 r-xp /lib/libcrypt-2.13.so
b6ae7000 b6aea000 r-xp /lib/libcap.so.2.21
b6af2000 b6af4000 r-xp /usr/lib/libiri.so
b6afc000 b6b1b000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6b23000 b6b39000 r-xp /usr/lib/libecore.so.1.7.99
b6b4f000 b6b54000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6b5d000 b6c2d000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6c2e000 b6c3c000 r-xp /usr/lib/libail.so.0.1.0
b6c44000 b6c5b000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6c64000 b6c6e000 r-xp /lib/libunwind.so.8.0.1
b6c9c000 b6db7000 r-xp /lib/libc-2.13.so
b6dc5000 b6dcd000 r-xp /lib/libgcc_s-4.6.4.so.1
b6dd5000 b6dff000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6e08000 b6e0b000 r-xp /usr/lib/libbundle.so.0.1.22
b6e13000 b6e15000 r-xp /lib/libdl-2.13.so
b6e1e000 b6e21000 r-xp /usr/lib/libsmack.so.1.0.0
b6e29000 b6e8b000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6e95000 b6ea7000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6eb0000 b6ec4000 r-xp /lib/libpthread-2.13.so
b6ed1000 b6ed5000 r-xp /usr/lib/libappcore-efl.so.1.1
b6edf000 b6ee1000 r-xp /usr/lib/libdlog.so.0.0.0
b6ee9000 b6ef4000 r-xp /usr/lib/libaul.so.0.1.0
b6efe000 b6f02000 r-xp /usr/lib/libsys-assert.so
b6f0b000 b6f28000 r-xp /lib/ld-2.13.so
b6f31000 b6f37000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6f3f000 b6f6b000 rw-p [heap]
b6f6b000 b7189000 rw-p [heap]
beb2a000 beb4b000 rwxp [stack]
End of Maps Information

Callstack Information (PID:13015)
Call Stack Count: 9
 0: cfree + 0x30 (0xb6d0b150) [/lib/libc.so.6] + 0x6f150
 1: app_terminate + 0x2a (0xb3ddf4f3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x94f3
 2: (0xb44ed1e9) [/usr/lib/libcapi-appfw-application.so.0] + 0x11e9
 3: appcore_efl_main + 0x2c6 (0xb6ed3717) [/usr/lib/libappcore-efl.so.1] + 0x2717
 4: ui_app_main + 0xb0 (0xb44edc79) [/usr/lib/libcapi-appfw-application.so.0] + 0x1c79
 5: main + 0x10a (0xb3ddf693) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9693
 6: (0xb6f33c6f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2c6f
 7: __libc_start_main + 0x114 (0xb6cb382c) [/lib/libc.so.6] + 0x1782c
 8: (0xb6f3435c) [/opt/usr/apps/org.tizen.other/bin/other] + 0x335c
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
6-01 01:20:13.535+0900 F/sio_packet(13015): accept()
06-01 01:20:13.535+0900 E/socket.io(13015): 743: encoded paylod length: 78
06-01 01:20:13.540+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:13.540+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:13.565+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:13.565+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:13.570+0900 F/sio_packet(13015): accept()
06-01 01:20:13.570+0900 E/socket.io(13015): 743: encoded paylod length: 76
06-01 01:20:13.575+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:13.575+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:13.595+0900 D/PROCESSMGR( 2145): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600043  register trigger_timer!  pointed_win=0x613769 
06-01 01:20:13.600+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:13.600+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:13.605+0900 D/EFL     (13015): ecore_x<13015> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=92828467 button=1
06-01 01:20:13.610+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:13.635+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:13.640+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:13.650+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:13.820+0900 D/EFL     (13015): ecore_x<13015> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=92828681 button=1
06-01 01:20:13.825+0900 F/sio_packet(13015): accept()
06-01 01:20:13.825+0900 E/socket.io(13015): 743: encoded paylod length: 78
06-01 01:20:13.825+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:13.845+0900 F/sio_packet(13015): accept()
06-01 01:20:13.850+0900 E/socket.io(13015): 743: encoded paylod length: 78
06-01 01:20:13.850+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:13.890+0900 F/sio_packet(13015): accept()
06-01 01:20:13.890+0900 E/socket.io(13015): 743: encoded paylod length: 81
06-01 01:20:13.890+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:13.915+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:13.915+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:13.925+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:13.925+0900 F/sio_packet(13015): accept()
06-01 01:20:13.925+0900 E/socket.io(13015): 743: encoded paylod length: 79
06-01 01:20:13.925+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:13.960+0900 F/sio_packet(13015): accept()
06-01 01:20:13.960+0900 E/socket.io(13015): 743: encoded paylod length: 85
06-01 01:20:13.960+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:13.980+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:13.980+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:13.980+0900 F/sio_packet(13015): accept()
06-01 01:20:13.980+0900 E/socket.io(13015): 743: encoded paylod length: 85
06-01 01:20:13.990+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:13.995+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:13.995+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.005+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.005+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:14.010+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.015+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.025+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.025+0900 F/sio_packet(13015): accept()
06-01 01:20:14.025+0900 E/socket.io(13015): 743: encoded paylod length: 84
06-01 01:20:14.025+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:14.040+0900 D/PROCESSMGR( 2145): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600043  register trigger_timer!  pointed_win=0x613769 
06-01 01:20:14.040+0900 D/EFL     (13015): ecore_x<13015> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=92828910 button=1
06-01 01:20:14.045+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.045+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.055+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.080+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.080+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.090+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.105+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.105+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.120+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.210+0900 D/EFL     (13015): ecore_x<13015> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=92829076 button=1
06-01 01:20:14.210+0900 F/sio_packet(13015): accept()
06-01 01:20:14.215+0900 E/socket.io(13015): 743: encoded paylod length: 84
06-01 01:20:14.215+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:14.245+0900 F/sio_packet(13015): accept()
06-01 01:20:14.245+0900 E/socket.io(13015): 743: encoded paylod length: 81
06-01 01:20:14.245+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:14.265+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.265+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.270+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.280+0900 F/sio_packet(13015): accept()
06-01 01:20:14.280+0900 E/socket.io(13015): 743: encoded paylod length: 76
06-01 01:20:14.280+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:14.295+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.295+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.305+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.315+0900 F/sio_packet(13015): accept()
06-01 01:20:14.315+0900 E/socket.io(13015): 743: encoded paylod length: 78
06-01 01:20:14.315+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:14.330+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.330+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.340+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.345+0900 F/sio_packet(13015): accept()
06-01 01:20:14.345+0900 E/socket.io(13015): 743: encoded paylod length: 76
06-01 01:20:14.345+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:14.365+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.365+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.375+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.385+0900 F/sio_packet(13015): accept()
06-01 01:20:14.385+0900 E/socket.io(13015): 743: encoded paylod length: 78
06-01 01:20:14.385+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:14.400+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.400+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.410+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.415+0900 F/sio_packet(13015): accept()
06-01 01:20:14.415+0900 E/socket.io(13015): 743: encoded paylod length: 76
06-01 01:20:14.415+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:14.435+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.435+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.440+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.450+0900 F/sio_packet(13015): accept()
06-01 01:20:14.450+0900 E/socket.io(13015): 743: encoded paylod length: 76
06-01 01:20:14.450+0900 E/socket.io(13015): 800: ping exit, con is expired? 0, ec: Operation canceled
06-01 01:20:14.470+0900 D/PROCESSMGR( 2145): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600043  register trigger_timer!  pointed_win=0x613769 
06-01 01:20:14.470+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.470+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.480+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.480+0900 D/EFL     (13015): ecore_x<13015> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=92829341 button=1
06-01 01:20:14.500+0900 E/socket.io(13015): 669: Received Message type(Event)
06-01 01:20:14.500+0900 F/get_binary(13015): in get binary_message()...
06-01 01:20:14.510+0900 D/CAPI_MEDIA_IMAGE_UTIL(13015): image_util.c: _convert_image_util_error_code(111) > [image_util_decode_jpeg_from_memory] ERROR_NONE(0x00000000)
06-01 01:20:14.790+0900 D/PROCESSMGR( 2145): e_mod_processmgr.c: _e_mod_processmgr_anr_ping_confirm_handler(336) > [PROCESSMGR] last_pointed_win=0x613769 bd->visible=1
06-01 01:20:15.605+0900 D/STARTER ( 2224): hw_key.c: _key_press_cb(656) > [_key_press_cb:656] _key_press_cb : XF86Phone Pressed
06-01 01:20:15.605+0900 W/STARTER ( 2224): hw_key.c: _key_press_cb(668) > [_key_press_cb:668] Home Key is pressed
06-01 01:20:15.605+0900 W/STARTER ( 2224): hw_key.c: _key_press_cb(686) > [_key_press_cb:686] homekey count : 1
06-01 01:20:15.605+0900 D/STARTER ( 2224): hw_key.c: _key_press_cb(695) > [_key_press_cb:695] create long press timer
06-01 01:20:15.730+0900 D/STARTER ( 2224): hw_key.c: _key_release_cb(470) > [_key_release_cb:470] _key_release_cb : XF86Phone Released
06-01 01:20:15.730+0900 W/STARTER ( 2224): hw_key.c: _key_release_cb(498) > [_key_release_cb:498] Home Key is released
06-01 01:20:15.730+0900 D/STARTER ( 2224): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
06-01 01:20:15.730+0900 D/STARTER ( 2224): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
06-01 01:20:15.745+0900 D/STARTER ( 2224): hw_key.c: _key_release_cb(536) > [_key_release_cb:536] delete cancelkey timer
06-01 01:20:15.745+0900 D/STARTER ( 2224): hw_key.c: _key_release_cb(559) > [_key_release_cb:559] delete long press timer
06-01 01:20:15.810+0900 I/SYSPOPUP( 2242): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
06-01 01:20:15.810+0900 I/SYSPOPUP( 2242): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
06-01 01:20:15.810+0900 D/VOLUME  ( 2242): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
06-01 01:20:15.810+0900 E/VOLUME  ( 2242): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
06-01 01:20:15.810+0900 E/VOLUME  ( 2242): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
06-01 01:20:15.815+0900 D/VOLUME  ( 2242): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
06-01 01:20:15.815+0900 D/VOLUME  ( 2242): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
06-01 01:20:15.815+0900 D/VOLUME  ( 2242): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
06-01 01:20:15.815+0900 E/VOLUME  ( 2242): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
06-01 01:20:15.815+0900 D/VOLUME  ( 2242): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
06-01 01:20:15.880+0900 D/STARTER ( 2224): hw_key.c: _key_press_cb(656) > [_key_press_cb:656] _key_press_cb : XF86Phone Pressed
06-01 01:20:15.880+0900 W/STARTER ( 2224): hw_key.c: _key_press_cb(668) > [_key_press_cb:668] Home Key is pressed
06-01 01:20:15.880+0900 W/STARTER ( 2224): hw_key.c: _key_press_cb(686) > [_key_press_cb:686] homekey count : 2
06-01 01:20:15.880+0900 D/STARTER ( 2224): hw_key.c: _key_press_cb(692) > [_key_press_cb:692] delete homekey timer
06-01 01:20:15.880+0900 D/STARTER ( 2224): hw_key.c: _key_press_cb(695) > [_key_press_cb:695] create long press timer
06-01 01:20:16.140+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
06-01 01:20:16.140+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
06-01 01:20:16.180+0900 D/STARTER ( 2224): hw_key.c: _destroy_syspopup_cb(630) > [_destroy_syspopup_cb:630] timer for cancel key operation
06-01 01:20:16.290+0900 D/STARTER ( 2224): hw_key.c: _launch_taskmgr_cb(132) > [_launch_taskmgr_cb:132] Launch TASKMGR
06-01 01:20:16.290+0900 D/STARTER ( 2224): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
06-01 01:20:16.295+0900 D/STARTER ( 2224): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
06-01 01:20:16.295+0900 I/SYSPOPUP( 2242): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
06-01 01:20:16.295+0900 I/SYSPOPUP( 2242): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
06-01 01:20:16.295+0900 D/VOLUME  ( 2242): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
06-01 01:20:16.295+0900 E/VOLUME  ( 2242): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
06-01 01:20:16.295+0900 E/VOLUME  ( 2242): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
06-01 01:20:16.300+0900 D/VOLUME  ( 2242): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
06-01 01:20:16.300+0900 D/VOLUME  ( 2242): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
06-01 01:20:16.300+0900 D/VOLUME  ( 2242): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
06-01 01:20:16.300+0900 D/AUL     ( 2224): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : org.tizen.task-mgr
06-01 01:20:16.300+0900 D/AUL     ( 2224): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(0)
06-01 01:20:16.300+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 0
06-01 01:20:16.300+0900 E/VOLUME  ( 2242): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
06-01 01:20:16.300+0900 D/VOLUME  ( 2242): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
06-01 01:20:16.300+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.task-mgr
06-01 01:20:16.305+0900 D/AUL     ( 2170): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 2219, pid = 2224
06-01 01:20:16.305+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
06-01 01:20:16.310+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1770) > process_pool: false
06-01 01:20:16.310+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1773) > h/w acceleration: USE
06-01 01:20:16.310+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.task-mgr
06-01 01:20:16.310+0900 D/AUL_AMD ( 2170): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
06-01 01:20:16.310+0900 D/AUL     ( 2170): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
06-01 01:20:16.310+0900 D/AUL_PAD ( 2202): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.task-mgr
06-01 01:20:16.310+0900 D/AUL_PAD ( 2202): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
06-01 01:20:16.315+0900 D/AUL_PAD ( 2202): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 13113 /usr/apps/org.tizen.task-mgr/bin/task-mgr
06-01 01:20:16.315+0900 D/AUL_PAD ( 2202): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
06-01 01:20:16.315+0900 D/AUL_PAD (13113): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
06-01 01:20:16.315+0900 D/AUL_PAD (13113): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
06-01 01:20:16.315+0900 D/AUL_PAD (13113): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.task-mgr / pkg_type : rpm / app_path : /usr/apps/org.tizen.task-mgr/bin/task-mgr 
06-01 01:20:16.330+0900 D/AUL_PAD (13113): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
06-01 01:20:16.330+0900 D/AUL_PAD (13113): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /usr/apps/org.tizen.task-mgr/bin/task-mgr##
06-01 01:20:16.330+0900 D/AUL_PAD (13113): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : HIDE_LAUNCH##
06-01 01:20:16.330+0900 D/AUL_PAD (13113): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_STARTTIME__##
06-01 01:20:16.330+0900 D/AUL_PAD (13113): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 6 : __AUL_CALLER_PID__##
06-01 01:20:16.330+0900 D/LAUNCH  (13113): launchpad.c: __real_launch(229) > [SECURE_LOG] [/usr/apps/org.tizen.task-mgr/bin/task-mgr:Platform:launchpad:done]
06-01 01:20:16.330+0900 E/AUL_PAD (13113): preload.h: __preload_exec(124) > dlopen("/usr/apps/org.tizen.task-mgr/bin/task-mgr") failed
06-01 01:20:16.330+0900 E/AUL_PAD (13113): preload.h: __preload_exec(126) > dlopen error: /usr/apps/org.tizen.task-mgr/bin/task-mgr: cannot dynamically load executable
06-01 01:20:16.330+0900 D/AUL_PAD (13113): launchpad.c: __normal_fork_exec(190) > start real fork and exec
06-01 01:20:16.415+0900 D/AUL_PAD ( 2202): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
06-01 01:20:16.415+0900 D/AUL_PAD ( 2202): sigchild.h: __send_app_launch_signal(112) > send launch signal done
06-01 01:20:16.415+0900 D/AUL_PAD ( 2202): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
06-01 01:20:16.415+0900 D/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
06-01 01:20:16.415+0900 E/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(264) > access
06-01 01:20:16.415+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 13113
06-01 01:20:16.415+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.task-mgr
06-01 01:20:16.415+0900 D/AUL     ( 2224): launch.c: app_request_to_launchpad(295) > launch request result : 13113
06-01 01:20:16.420+0900 D/STARTER ( 2224): dbus-util.c: starter_dbus_set_oomadj(223) > [starter_dbus_set_oomadj:223] org.tizen.system.deviced.Process-oomadj_set : 0
06-01 01:20:16.420+0900 D/RESOURCED( 2369): proc-noti.c: recv_str(87) > [recv_str,87] str is null
06-01 01:20:16.420+0900 D/RESOURCED( 2369): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2170
06-01 01:20:16.420+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.task-mgr, pid 13113, type 4 
06-01 01:20:16.420+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.task-mgr, 13113
06-01 01:20:16.420+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.task-mgr with pkgname
06-01 01:20:16.420+0900 D/RESOURCED( 2369): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 13113
06-01 01:20:16.420+0900 D/RESOURCED( 2369): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 13113
06-01 01:20:16.420+0900 D/RESOURCED( 2369): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
06-01 01:20:16.420+0900 D/RESOURCED( 2369): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
06-01 01:20:16.420+0900 D/RESOURCED( 2369): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
06-01 01:20:16.420+0900 D/RESOURCED( 2369): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
06-01 01:20:16.420+0900 D/RESOURCED( 2369): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
06-01 01:20:16.420+0900 D/RESOURCED( 2369): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
06-01 01:20:16.420+0900 E/RESOURCED( 2369): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 545
06-01 01:20:16.420+0900 D/RESOURCED( 2369): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
06-01 01:20:16.430+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: main(461) > Application Main Function 
06-01 01:20:16.430+0900 D/LAUNCH  (13113): appcore-efl.c: appcore_efl_main(1569) > [task-mgr:Application:main:done]
06-01 01:20:16.440+0900 D/AUL     (13113): pkginfo.c: aul_app_get_appid_bypid(196) > [SECURE_LOG] appid for 13113 is org.tizen.task-mgr
06-01 01:20:16.605+0900 D/APP_CORE(13113): appcore-efl.c: __before_loop(1012) > elm_config_preferred_engine_set : opengl_x11
06-01 01:20:16.615+0900 D/AUL     (13113): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 13113 is org.tizen.task-mgr
06-01 01:20:16.615+0900 D/APP_CORE(13113): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.task-mgr/res/locale
06-01 01:20:16.615+0900 D/APP_CORE(13113): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
06-01 01:20:16.615+0900 D/AUL     (13113): app_sock.c: __create_server_sock(135) > pg path - already exists
06-01 01:20:16.615+0900 D/LAUNCH  (13113): appcore-efl.c: __before_loop(1035) > [task-mgr:Platform:appcore_init:done]
06-01 01:20:16.615+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: _create_app(331) > Application Create Callback 
06-01 01:20:16.795+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: create_layout(197) > create_layout
06-01 01:20:16.795+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: create_layout(216) > Resolution: HD
06-01 01:20:16.795+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: load_data(282) > load_data
06-01 01:20:16.795+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_app_list_create(625) > recent_app_list_create
06-01 01:20:16.800+0900 D/PKGMGR_INFO(13113): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_filter_foreach_appinfo(3109) > [SECURE_LOG] where = package_app_info.app_taskmanage IN ('true','True') and package_app_info.app_disable IN ('false','False')
06-01 01:20:16.800+0900 D/PKGMGR_INFO(13113): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_filter_foreach_appinfo(3115) > [SECURE_LOG] query = select DISTINCT package_app_info.*, package_app_localized_info.app_locale, package_app_localized_info.app_label, package_app_localized_info.app_icon from package_app_info LEFT OUTER JOIN package_app_localized_info ON package_app_info.app_id=package_app_localized_info.app_id and package_app_localized_info.app_locale IN ('No Locale', 'en-us') LEFT OUTER JOIN package_app_app_svc ON package_app_info.app_id=package_app_app_svc.app_id LEFT OUTER JOIN package_app_app_category ON package_app_info.app_id=package_app_app_category.app_id where package_app_info.app_taskmanage IN ('true','True') and package_app_info.app_disable IN ('false','False')
06-01 01:20:16.805+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 12
06-01 01:20:16.805+0900 D/AUL     (13113): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 406
06-01 01:20:16.805+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_app_list_create(653) > USP mode is disabled.
06-01 01:20:16.805+0900 E/RUA     (13113): rua.c: rua_history_load_db(278) > rua_history_load_db ok. nrows : 2, ncols : 5
06-01 01:20:16.805+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_app_list_create(680) > Apps in history: 2
06-01 01:20:16.805+0900 E/TASK_MGR_LITE(13113): recent_apps.c: list_retrieve_item(348) > Fail to get ai table !!
06-01 01:20:16.805+0900 D/TASK_MGR_LITE(13113): recent_apps.c: list_retrieve_item(444) > [org.tizen.other]
06-01 01:20:16.805+0900 D/TASK_MGR_LITE(13113): [/opt/usr/apps/org.tizen.other/shared/res/other.png]
06-01 01:20:16.805+0900 D/TASK_MGR_LITE(13113): [other]
06-01 01:20:16.805+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_app_list_create(741) > App added into the running_list : pkgid:org.tizen.other - appid:org.tizen.other
06-01 01:20:16.805+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_app_list_create(773) > HISTORY LIST: (nil) 0
06-01 01:20:16.805+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_app_list_create(774) > RUNNING LIST: 0x173dc8 1
06-01 01:20:16.810+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: load_data(292) > LISTs : 0x1bfbd0, 0x1bfc40
06-01 01:20:16.810+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: load_data(304) > App list should display !!
06-01 01:20:16.810+0900 D/TASK_MGR_LITE(13113): genlist.c: genlist_init(258) > in gen list: 0x173dc8 (nil)
06-01 01:20:16.810+0900 D/TASK_MGR_LITE(13113): genlist.c: recent_app_panel_create(98) > Creating task_mgr_genlist widget, noti count is : [-1]
06-01 01:20:16.825+0900 D/TASK_MGR_LITE(13113): genlist_item.c: genlist_item_class_create(308) > Genlist item class create
06-01 01:20:16.825+0900 D/TASK_MGR_LITE(13113): genlist_item.c: genlist_clear_all_class_create(323) > Genlist clear all class create
06-01 01:20:16.840+0900 D/TASK_MGR_LITE(13113): genlist.c: genlist_update(432) > genlist_update
06-01 01:20:16.840+0900 D/TASK_MGR_LITE(13113): genlist.c: genlist_clear_list(297) > genlist_clear_list
06-01 01:20:16.840+0900 D/TASK_MGR_LITE(13113): genlist.c: add_clear_btn_to_genlist(417) > add_clear_btn_to_genlist
06-01 01:20:16.840+0900 D/TASK_MGR_LITE(13113): genlist.c: genlist_add_item(165) > genlist_add_item
06-01 01:20:16.840+0900 D/TASK_MGR_LITE(13113): genlist.c: genlist_add_item(171) > Adding item: 0x1bcfb0
06-01 01:20:16.840+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_apps_find_by_appid(210) > recent_apps_find_by_appid
06-01 01:20:16.845+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 12
06-01 01:20:16.845+0900 D/AUL     (13113): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 406
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.lockscreen (2234)
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.volume (2242)
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.menu-screen (2243)
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.basic (7435)
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.other (13015)
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_apps_find_by_appid_cb(201) > FOUND
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.task-mgr (13113)
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): genlist.c: genlist_update(449) > Are there recent apps: (nil)
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): genlist.c: genlist_update(450) > Adding recent apps... 0
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): genlist.c: genlist_item_show(192) > genlist_item_show
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): genlist.c: genlist_item_show(199) > ELM COUNT = 2
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: load_data(323) > load_data END. 
06-01 01:20:16.845+0900 D/LAUNCH  (13113): appcore-efl.c: __before_loop(1045) > [task-mgr:Application:create:done]
06-01 01:20:16.845+0900 D/APP_CORE(13113): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
06-01 01:20:16.845+0900 D/APP_CORE(13113): appcore.c: __aul_handler(423) > [APP 13113]     AUL event: AUL_START
06-01 01:20:16.845+0900 D/APP_CORE(13113): appcore-efl.c: __do_app(470) > [APP 13113] Event: RESET State: CREATED
06-01 01:20:16.845+0900 D/APP_CORE(13113): appcore-efl.c: __do_app(496) > [APP 13113] RESET
06-01 01:20:16.845+0900 D/LAUNCH  (13113): appcore-efl.c: __do_app(498) > [task-mgr:Application:reset:start]
06-01 01:20:16.845+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: _reset_app(446) > _reset_app
06-01 01:20:16.845+0900 D/LAUNCH  (13113): appcore-efl.c: __do_app(501) > [task-mgr:Application:reset:done]
06-01 01:20:16.845+0900 E/PKGMGR_INFO(13113): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_get_appinfo(1308) > (appid == NULL) appid is NULL
06-01 01:20:16.845+0900 I/APP_CORE(13113): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
06-01 01:20:16.845+0900 I/APP_CORE(13113): appcore-efl.c: __do_app(509) > [APP 13113] Initial Launching, call the resume_cb
06-01 01:20:16.845+0900 D/APP_CORE(13113): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
06-01 01:20:16.850+0900 D/APP_CORE(13113): appcore.c: __prt_ltime(183) > [APP 13113] first idle after reset: 550 msec
06-01 01:20:16.855+0900 D/APP_CORE(13113): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:3200008
06-01 01:20:16.855+0900 D/APP_CORE(13113): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:3200008
06-01 01:20:16.875+0900 W/PROCESSMGR( 2145): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=13113
06-01 01:20:16.920+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 2800003"
06-01 01:20:16.920+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 3200008"
06-01 01:20:16.920+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
06-01 01:20:16.920+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 3200008"
06-01 01:20:16.920+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
06-01 01:20:16.975+0900 D/AUL_AMD ( 2170): amd_launch.c: __e17_status_handler(1888) > pid(13113) status(3)
06-01 01:20:16.975+0900 D/RESOURCED( 2369): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 13113, type = 0
06-01 01:20:16.975+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 13113
06-01 01:20:16.975+0900 I/RESOURCED( 2369): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/foreground/cgroup.procs, pid : 13113, oom : 200
06-01 01:20:16.975+0900 E/RESOURCED( 2369): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/foreground/cgroup.procs open failed
06-01 01:20:16.990+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _clear_all_btn_show_cb(243) > _clear_all_btn_show_cb
06-01 01:20:16.990+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
06-01 01:20:16.990+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.other/shared/res/other.png; other 0x21f9e8
06-01 01:20:16.990+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_set(169) > It is already set.
06-01 01:20:16.990+0900 D/TASK_MGR_LITE(13113): genlist_item.c: recent_app_item_create(543) > 
06-01 01:20:16.990+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_main_create(765) > 
06-01 01:20:17.005+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
06-01 01:20:17.005+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
06-01 01:20:17.005+0900 E/TASK_MGR_LITE(13113): genlist_item.c: del_cb(758) > Deleted
06-01 01:20:17.010+0900 D/APP_CORE(13113): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:3200008 fully_obscured 0
06-01 01:20:17.010+0900 D/APP_CORE(13113): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
06-01 01:20:17.010+0900 D/APP_CORE(13113): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
06-01 01:20:17.010+0900 D/APP_CORE(13113): appcore-efl.c: __do_app(470) > [APP 13113] Event: RESUME State: RUNNING
06-01 01:20:17.010+0900 D/LAUNCH  (13113): appcore-efl.c: __do_app(557) > [task-mgr:Application:resume:start]
06-01 01:20:17.010+0900 D/LAUNCH  (13113): appcore-efl.c: __do_app(567) > [task-mgr:Application:resume:done]
06-01 01:20:17.010+0900 D/LAUNCH  (13113): appcore-efl.c: __do_app(569) > [task-mgr:Application:Launching:done]
06-01 01:20:17.010+0900 D/APP_CORE(13113): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
06-01 01:20:17.010+0900 E/APP_CORE(13113): appcore-efl.c: __trm_app_info_send_socket(233) > access
06-01 01:20:17.015+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _clear_all_btn_show_cb(243) > _clear_all_btn_show_cb
06-01 01:20:17.015+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
06-01 01:20:17.020+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.other/shared/res/other.png; other 0x21f9e8
06-01 01:20:17.020+0900 D/TASK_MGR_LITE(13113): genlist_item.c: recent_app_item_create(543) > 
06-01 01:20:17.020+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_main_create(765) > 
06-01 01:20:17.020+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
06-01 01:20:17.020+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
06-01 01:20:17.185+0900 D/STARTER ( 2224): hw_key.c: _key_release_cb(470) > [_key_release_cb:470] _key_release_cb : XF86Phone Released
06-01 01:20:17.185+0900 W/STARTER ( 2224): hw_key.c: _key_release_cb(498) > [_key_release_cb:498] Home Key is released
06-01 01:20:17.185+0900 D/STARTER ( 2224): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
06-01 01:20:17.185+0900 D/STARTER ( 2224): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
06-01 01:20:17.195+0900 D/STARTER ( 2224): hw_key.c: _key_release_cb(536) > [_key_release_cb:536] delete cancelkey timer
06-01 01:20:17.265+0900 I/SYSPOPUP( 2242): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
06-01 01:20:17.265+0900 I/SYSPOPUP( 2242): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
06-01 01:20:17.265+0900 D/VOLUME  ( 2242): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
06-01 01:20:17.265+0900 E/VOLUME  ( 2242): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
06-01 01:20:17.265+0900 E/VOLUME  ( 2242): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
06-01 01:20:17.265+0900 D/VOLUME  ( 2242): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
06-01 01:20:17.265+0900 D/VOLUME  ( 2242): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
06-01 01:20:17.265+0900 D/VOLUME  ( 2242): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
06-01 01:20:17.270+0900 E/VOLUME  ( 2242): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
06-01 01:20:17.270+0900 D/VOLUME  ( 2242): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
06-01 01:20:17.425+0900 D/AUL_AMD ( 2170): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.task-mgr /usr/apps/org.tizen.task-mgr/bin/task-mgr
06-01 01:20:17.425+0900 D/RUA     ( 2170): rua.c: rua_add_history(179) > rua_add_history start
06-01 01:20:17.435+0900 D/RUA     ( 2170): rua.c: rua_add_history(247) > rua_add_history ok
06-01 01:20:17.550+0900 D/EFL     (13113): ecore_x<13113> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=92832423 button=1
06-01 01:20:17.615+0900 D/EFL     (13113): ecore_x<13113> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=92832485 button=1
06-01 01:20:17.615+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
06-01 01:20:17.615+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected list -0.000000 no_shutdown: 0
06-01 01:20:17.615+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
06-01 01:20:17.615+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected layout -0.000000 no_shutdown: 0
06-01 01:20:17.615+0900 D/TASK_MGR_LITE(13113): genlist_item.c: clear_all_btn_clicked_cb(218) > Removing all items...
06-01 01:20:17.615+0900 D/TASK_MGR_LITE(13113): genlist_item.c: clear_all_btn_clicked_cb(230) > On REDWOOD target (HD) : No Animation.
06-01 01:20:17.620+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_apps_kill_all(164) > recent_apps_kill_all
06-01 01:20:17.620+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 12
06-01 01:20:17.620+0900 D/AUL     (13113): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 406
06-01 01:20:17.625+0900 D/PROCESSMGR( 2145): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600043  register trigger_timer!  pointed_win=0x614ba0 
06-01 01:20:17.625+0900 D/TASK_MGR_LITE(13113): recent_apps.c: get_app_taskmanage(121) > app org.tizen.lockscreen is taskmanage: 0
06-01 01:20:17.630+0900 D/TASK_MGR_LITE(13113): recent_apps.c: get_app_taskmanage(121) > app org.tizen.volume is taskmanage: 0
06-01 01:20:17.640+0900 D/TASK_MGR_LITE(13113): recent_apps.c: get_app_taskmanage(121) > app org.tizen.menu-screen is taskmanage: 0
06-01 01:20:17.640+0900 D/PKGMGR_INFO(13113): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_get_appinfo(1338) > [SECURE_LOG] Appid[org.tizen.basic] not found in DB
06-01 01:20:17.650+0900 D/TASK_MGR_LITE(13113): recent_apps.c: get_app_taskmanage(121) > app org.tizen.other is taskmanage: 1
06-01 01:20:17.650+0900 D/AUL     (13113): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : 13015
06-01 01:20:17.650+0900 D/AUL     (13113): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(4)
06-01 01:20:17.650+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 4
06-01 01:20:17.655+0900 D/RESOURCED( 2369): proc-noti.c: recv_str(87) > [recv_str,87] str is null
06-01 01:20:17.655+0900 D/RESOURCED( 2369): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2170
06-01 01:20:17.655+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid (null), pid 13015, type 6 
06-01 01:20:17.655+0900 E/RESOURCED( 2369): datausage-common.c: app_terminate_cb(254) > [app_terminate_cb,254] No classid to terminate!
06-01 01:20:17.655+0900 D/RESOURCED( 2369): proc-main.c: proc_remove_process_list(363) > [proc_remove_process_list,363] found_pid 13015
06-01 01:20:17.655+0900 D/AUL_AMD ( 2170): amd_request.c: __app_process_by_pid(175) > __app_process_by_pid, cmd: 4, pid: 13015, 
06-01 01:20:17.655+0900 D/AUL     ( 2170): app_sock.c: __app_send_raw_with_delay_reply(434) > pid(13015) : cmd(4)
06-01 01:20:17.655+0900 D/AUL_AMD ( 2170): amd_launch.c: _term_app(878) > term done
06-01 01:20:17.655+0900 D/AUL_AMD ( 2170): amd_launch.c: __set_reply_handler(860) > listen fd : 27, send fd : 16
06-01 01:20:17.665+0900 D/AUL_AMD ( 2170): amd_launch.c: __reply_handler(784) > listen fd(27) , send fd(16), pid(13015), cmd(4)
06-01 01:20:17.665+0900 D/AUL     (13113): launch.c: app_request_to_launchpad(295) > launch request result : 0
06-01 01:20:17.665+0900 D/APP_CORE(13015): appcore.c: __aul_handler(443) > [APP 13015]     AUL event: AUL_TERMINATE
06-01 01:20:17.665+0900 D/APP_CORE(13015): appcore-efl.c: __do_app(470) > [APP 13015] Event: TERMINATE State: RUNNING
06-01 01:20:17.665+0900 D/APP_CORE(13015): appcore-efl.c: __do_app(486) > [APP 13015] TERMINATE
06-01 01:20:17.665+0900 D/AUL     (13015): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
06-01 01:20:17.665+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 22
06-01 01:20:17.665+0900 D/TASK_MGR_LITE(13113): recent_apps.c: kill_pid(141) > terminate pid = 13015
06-01 01:20:17.670+0900 I/CAPI_APPFW_APPLICATION(13015): app_main.c: _ui_app_appcore_terminate(577) > app_appcore_terminate
06-01 01:20:17.675+0900 D/TASK_MGR_LITE(13113): recent_apps.c: get_app_taskmanage(121) > app org.tizen.task-mgr is taskmanage: 0
06-01 01:20:17.675+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_del(189) > _recent_app_item_content_del
06-01 01:20:17.675+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_del(194) > Data exist: 0xb0b03678
06-01 01:20:17.675+0900 E/TASK_MGR_LITE(13113): genlist_item.c: del_cb(758) > Deleted
06-01 01:20:17.675+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _recent_app_item_content_del(198) > Item deleted
06-01 01:20:17.675+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _clear_all_button_del(206) > _clear_all_button_del
06-01 01:20:17.675+0900 D/TASK_MGR_LITE(13113): genlist_item.c: _clear_all_button_del(213) > _clear_all_button_del END
06-01 01:20:17.690+0900 D/AUL     (13113): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
06-01 01:20:17.690+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 22
06-01 01:20:17.690+0900 D/APP_CORE(13113): appcore-efl.c: __after_loop(1060) > [APP 13113] PAUSE before termination
06-01 01:20:17.690+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: _pause_app(453) > _pause_app
06-01 01:20:17.690+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: _terminate_app(393) > _terminate_app
06-01 01:20:17.690+0900 D/TASK_MGR_LITE(13113): genlist.c: genlist_clear_list(297) > genlist_clear_list
06-01 01:20:17.695+0900 D/TASK_MGR_LITE(13113): genlist.c: recent_app_panel_del_cb(84) > recent_app_panel_del_cb
06-01 01:20:17.695+0900 D/TASK_MGR_LITE(13113): genlist_item.c: genlist_item_class_destroy(340) > Genlist class free
06-01 01:20:17.695+0900 D/TASK_MGR_LITE(13113): task-mgr-lite.c: delete_layout(272) > delete_layout
06-01 01:20:17.695+0900 D/TASK_MGR_LITE(13113): recent_apps.c: recent_app_list_destroy(791) > recent_app_list_destroy
06-01 01:20:17.695+0900 D/TASK_MGR_LITE(13113): recent_apps.c: list_destroy(475) > START list_destroy
06-01 01:20:17.695+0900 D/TASK_MGR_LITE(13113): recent_apps.c: list_destroy(487) > FREE ALL list_destroy
06-01 01:20:17.695+0900 D/TASK_MGR_LITE(13113): recent_apps.c: list_destroy(493) > FREING: list_destroy org.tizen.other
06-01 01:20:17.695+0900 D/TASK_MGR_LITE(13113): recent_apps.c: list_unretrieve_item(295) > FREING 0x1bcfb0 org.tizen.other 's Item(list_type_default_s) in list_unretrieve_item
06-01 01:20:17.695+0900 D/TASK_MGR_LITE(13113): recent_apps.c: list_unretrieve_item(333) > list_unretrieve_item END 
06-01 01:20:17.695+0900 D/TASK_MGR_LITE(13113): recent_apps.c: list_destroy(500) > END list_destroy
06-01 01:20:17.705+0900 E/APP_CORE(13113): appcore.c: appcore_flush_memory(583) > Appcore not initialized
06-01 01:20:17.705+0900 W/PROCESSMGR( 2145): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=13015
06-01 01:20:17.715+0900 D/PROCESSMGR( 2145): e_mod_processmgr.c: _e_mod_processmgr_wininfo_del(141) > [PROCESSMGR] delete anr_trigger_timer!
06-01 01:20:17.720+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 3200008"
06-01 01:20:17.720+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2800003"
06-01 01:20:17.720+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
06-01 01:20:17.720+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2800003"
06-01 01:20:17.720+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
06-01 01:20:18.205+0900 W/CRASH_MANAGER(13118): worker.c: worker_job(1189) > 11130156f7468143308921
