S/W Version Information
Model: SM-Z130H
Tizen-Version: 2.3.0.1
Build-Number: Z130HDDU0BOD7
Build-Date: 2015.04.16 12:41:35

Crash Information
Process Name: other
PID: 16066
Date: 2015-07-06 14:01:40+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xb1b7f004

Register Information
r0   = 0xb1b7f008, r1   = 0xb4d40bf3
r2   = 0x000000e4, r3   = 0x00000000
r4   = 0xb4de7f1c, r5   = 0xb1b7f008
r6   = 0x00000245, r7   = 0xbef82d88
r8   = 0xbef82e14, r9   = 0xb6f9f2c4
r10  = 0x00000000, fp   = 0xb7c6c5a0
ip   = 0xb4de7f98, sp   = 0xbef82d70
lr   = 0xb4d40bf3, pc   = 0xb6cd6150
cpsr = 0xa0000010

Memory Information
MemTotal:   730748 KB
MemFree:    210744 KB
Buffers:     30136 KB
Cached:     214332 KB
VmPeak:     103444 KB
VmSize:     103440 KB
VmLck:           0 KB
VmPin:           0 KB
VmHWM:       17056 KB
VmRSS:       17052 KB
VmData:      47556 KB
VmStk:         136 KB
VmExe:          20 KB
VmLib:       25912 KB
VmPTE:          66 KB
VmSwap:          0 KB

Threads Information
Threads: 5
PID = 16066 TID = 16066
16066 16068 16069 16071 16072 

Maps Information
b2001000 b2800000 rw-p [stack:16072]
b2986000 b3185000 rw-p [stack:16071]
b3186000 b3985000 rw-p [stack:16070]
b3a18000 b3a19000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b3a21000 b3a28000 r-xp /usr/lib/libfeedback.so.0.1.4
b3a48000 b3a49000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b3a51000 b3a52000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b3a5a000 b3a71000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b3c18000 b3c1c000 r-xp /usr/lib/bufmgr/libtbm_sprd7727.so.0.0.0
b3c26000 b4425000 rw-p [stack:16068]
b4425000 b456c000 r-xp /usr/lib/driver/libMali.so
b4579000 b45c9000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b4900000 b49ca000 r-xp /usr/lib/libCOREGL.so.4.0
b49db000 b4b1a000 r-xp /usr/lib/libicui18n.so.51.1
b4b2a000 b4b2f000 r-xp /usr/lib/libxcb-render.so.0.0.0
b4b38000 b4b39000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b4b42000 b4b5a000 r-xp /usr/lib/libpng12.so.0.50.0
b4b62000 b4b65000 r-xp /usr/lib/libEGL.so.1.4
b4b6d000 b4b7b000 r-xp /usr/lib/libGLESv2.so.2.0
b4b84000 b4b86000 r-xp /usr/lib/libiniparser.so.0
b4b90000 b4b98000 r-xp /usr/lib/libui-extension.so.0.1.0
b4b99000 b4b9c000 r-xp /usr/lib/libnative-buffer.so.0.1.0
b4ba4000 b4c5a000 r-xp /usr/lib/libcairo.so.2.11200.14
b4c65000 b4c77000 r-xp /usr/lib/libtts.so
b4c7f000 b4c86000 r-xp /usr/lib/libtbm.so.1.0.0
b4c8e000 b4c93000 r-xp /usr/lib/libcapi-media-tool.so.0.1.3
b4c9b000 b4c9f000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b4ca8000 b4caa000 r-xp /usr/lib/libdri2.so.0.0.0
b4cb2000 b4cb9000 r-xp /usr/lib/libdrm.so.2.4.0
b4cc2000 b4cd8000 r-xp /usr/lib/libefl-assist.so.0.1.0
b4ce0000 b4ce8000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b4cf0000 b4cf5000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b4cfd000 b4cff000 r-xp /usr/lib/libefl-extension.so.0.1.0
b4d07000 b4d0e000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.4
b4d17000 b4d1a000 r-xp /usr/lib/libcapi-media-image-util.so.0.3.22
b4d24000 b4d2e000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b4d37000 b4ddf000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b4de9000 b4df3000 r-xp /lib/libnss_files-2.13.so
b4dfc000 b4e0e000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b4e16000 b4e2c000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b4e34000 b4f02000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b4f19000 b4f3d000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b4f46000 b4f4c000 r-xp /usr/lib/libappsvc.so.0.1.0
b4f54000 b4f62000 r-xp /usr/lib/libail.so.0.1.0
b4f6a000 b4f6c000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.3
b4f75000 b4f7a000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.3
b4f82000 b4f84000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b4f8d000 b4f8e000 r-xp /usr/lib/libosp-env-config.so.1.2.2.1
b4f97000 b4f9b000 r-xp /usr/lib/libogg.so.0.7.1
b4fa3000 b4fc5000 r-xp /usr/lib/libvorbis.so.0.4.3
b4fcd000 b50b1000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b50c5000 b50f6000 r-xp /usr/lib/libFLAC.so.8.2.0
b5a90000 b5b24000 r-xp /usr/lib/libstdc++.so.6.0.16
b5b37000 b5b39000 r-xp /usr/lib/libXau.so.6.0.0
b5b41000 b5b4b000 r-xp /usr/lib/libspdy.so.0.0.0
b5b54000 b5ba0000 r-xp /usr/lib/libssl.so.1.0.0
b5bad000 b5bdb000 r-xp /usr/lib/libidn.so.11.5.44
b5be3000 b5bed000 r-xp /usr/lib/libcares.so.2.1.0
b5bf5000 b5c16000 r-xp /usr/lib/libexif.so.12.3.3
b5c29000 b5c6e000 r-xp /usr/lib/libsndfile.so.1.0.25
b5c7c000 b5c92000 r-xp /lib/libexpat.so.1.5.2
b5c9b000 b5d80000 r-xp /usr/lib/libicuuc.so.51.1
b5d95000 b5dc9000 r-xp /usr/lib/libicule.so.51.1
b5dd2000 b5de5000 r-xp /usr/lib/libxcb.so.1.1.0
b5ded000 b5e2a000 r-xp /usr/lib/libcurl.so.4.3.0
b5e33000 b5e3c000 r-xp /usr/lib/libethumb.so.1.7.99
b5e45000 b5e47000 r-xp /usr/lib/libctxdata.so.0.0.0
b5e4f000 b5e5c000 r-xp /usr/lib/libremix.so.0.0.0
b5e64000 b5e65000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5e6d000 b5e84000 r-xp /usr/lib/liblua-5.1.so
b5e8d000 b5e94000 r-xp /usr/lib/libembryo.so.1.7.99
b5e9c000 b5ebf000 r-xp /usr/lib/libjpeg.so.8.0.2
b5ed7000 b5f2d000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5f3a000 b5f8d000 r-xp /usr/lib/libfreetype.so.6.8.1
b5f98000 b5fc0000 r-xp /usr/lib/libfontconfig.so.1.8.0
b5fc1000 b6007000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b6010000 b6023000 r-xp /usr/lib/libfribidi.so.0.3.1
b602b000 b602e000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b6036000 b603a000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b6042000 b6047000 r-xp /usr/lib/libecore_fb.so.1.7.99
b6050000 b605a000 r-xp /usr/lib/libXext.so.6.4.0
b6062000 b6143000 r-xp /usr/lib/libX11.so.6.3.0
b614e000 b6151000 r-xp /usr/lib/libXtst.so.6.1.0
b6159000 b615f000 r-xp /usr/lib/libXrender.so.1.3.0
b6167000 b616c000 r-xp /usr/lib/libXrandr.so.2.2.0
b6174000 b6175000 r-xp /usr/lib/libXinerama.so.1.0.0
b617e000 b6186000 r-xp /usr/lib/libXi.so.6.1.0
b6187000 b618a000 r-xp /usr/lib/libXfixes.so.3.1.0
b6192000 b6194000 r-xp /usr/lib/libXgesture.so.7.0.0
b619c000 b619e000 r-xp /usr/lib/libXcomposite.so.1.0.0
b61a6000 b61a7000 r-xp /usr/lib/libXdamage.so.1.1.0
b61b0000 b61b6000 r-xp /usr/lib/libXcursor.so.1.0.2
b61bf000 b61d8000 r-xp /usr/lib/libecore_con.so.1.7.99
b61e2000 b61e8000 r-xp /usr/lib/libecore_imf.so.1.7.99
b61f0000 b61f8000 r-xp /usr/lib/libethumb_client.so.1.7.99
b6200000 b6209000 r-xp /usr/lib/libedbus.so.1.7.99
b6211000 b626e000 r-xp /usr/lib/libedje.so.1.7.99
b6277000 b6288000 r-xp /usr/lib/libecore_input.so.1.7.99
b6290000 b6295000 r-xp /usr/lib/libecore_file.so.1.7.99
b629d000 b62b6000 r-xp /usr/lib/libeet.so.1.7.99
b62c7000 b62cb000 r-xp /usr/lib/libappcore-common.so.1.1
b62d3000 b639a000 r-xp /usr/lib/libevas.so.1.7.99
b63bf000 b63e0000 r-xp /usr/lib/libecore_evas.so.1.7.99
b63e9000 b6418000 r-xp /usr/lib/libecore_x.so.1.7.99
b6422000 b6559000 r-xp /usr/lib/libelementary.so.1.7.99
b656f000 b6570000 r-xp /usr/lib/osp/libappinfo.so.1.2.2.1
b6578000 b657c000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.3
b6587000 b658a000 r-xp /lib/libgpg-error.so.0.5.0
b6592000 b65ea000 r-xp /usr/lib/libgcrypt.so.11.5.3
b65f4000 b6620000 r-xp /usr/lib/libsystemd.so.0.0.1
b6629000 b662b000 r-xp /usr/lib/journal/libjournal.so.0.1.0
b6634000 b66ff000 r-xp /usr/lib/libxml2.so.2.7.8
b670d000 b671d000 r-xp /lib/libresolv-2.13.so
b6721000 b6737000 r-xp /lib/libz.so.1.2.5
b673f000 b6741000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b6749000 b674e000 r-xp /usr/lib/libffi.so.5.0.10
b6757000 b6758000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b6760000 b6763000 r-xp /lib/libattr.so.1.1.0
b676b000 b676e000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b6776000 b677d000 r-xp /usr/lib/libvconf.so.0.2.45
b6786000 b692e000 r-xp /usr/lib/libcrypto.so.1.0.0
b694f000 b6965000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b696d000 b69d6000 r-xp /lib/libm-2.13.so
b69df000 b6a1f000 r-xp /usr/lib/libeina.so.1.7.99
b6a28000 b6a5c000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b6a65000 b6b39000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6b45000 b6b4a000 r-xp /usr/lib/libcapi-base-common.so.0.1.6
b6b53000 b6b59000 r-xp /lib/librt-2.13.so
b6b62000 b6b69000 r-xp /lib/libcrypt-2.13.so
b6b99000 b6b9c000 r-xp /lib/libcap.so.2.21
b6ba4000 b6ba6000 r-xp /usr/lib/libiri.so
b6bae000 b6bcd000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6bd5000 b6beb000 r-xp /usr/lib/libecore.so.1.7.99
b6c01000 b6c06000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6c0f000 b6c26000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6c2f000 b6c3a000 r-xp /lib/libunwind.so.8.0.1
b6c67000 b6d82000 r-xp /lib/libc-2.13.so
b6d90000 b6d98000 r-xp /lib/libgcc_s-4.6.4.so.1
b6da0000 b6dca000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6dd3000 b6dd6000 r-xp /usr/lib/libbundle.so.0.1.22
b6dde000 b6de0000 r-xp /lib/libdl-2.13.so
b6de9000 b6dec000 r-xp /usr/lib/libsmack.so.1.0.0
b6df4000 b6ec4000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6ec5000 b6f2a000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6f34000 b6f46000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6f4e000 b6f62000 r-xp /lib/libpthread-2.13.so
b6f6d000 b6f6f000 r-xp /usr/lib/libdlog.so.0.0.0
b6f77000 b6f82000 r-xp /usr/lib/libaul.so.0.1.0
b6f91000 b6f94000 rw-p [stack:16069]
b6f94000 b6f97000 r-xp /usr/lib/libappcore-efl.so.1.1
b6fa1000 b6fa5000 r-xp /usr/lib/libsys-assert.so
b6fae000 b6fcb000 r-xp /lib/ld-2.13.so
b6fd4000 b6fd9000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b7c5a000 b7c84000 rw-p [heap]
b7c84000 b7f2f000 rw-p [heap]
bef63000 bef84000 rw-p [stack]
b7c84000 b7f2f000 rw-p [heap]
bef63000 bef84000 rw-p [stack]
End of Maps Information

Callstack Information (PID:16066)
Call Stack Count: 10
 0: cfree + 0x30 (0xb6cd6150) [/lib/libc.so.6] + 0x6f150
 1: app_terminate + 0x2a (0xb4d40bf3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9bf3
 2: (0xb6579ad1) [/usr/lib/libcapi-appfw-application.so.0] + 0x1ad1
 3: appcore_efl_main + 0x1c4 (0xb6f963cd) [/usr/lib/libappcore-efl.so.1] + 0x23cd
 4: ui_app_main + 0xb0 (0xb657a499) [/usr/lib/libcapi-appfw-application.so.0] + 0x2499
 5: main + 0x10c (0xb4d40d95) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9d95
 6: (0xb6fd6dc7) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2dc7
 7: (0xb6fd5d8f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x1d8f
 8: __libc_start_main + 0x114 (0xb6c7e82c) [/lib/libc.so.6] + 0x1782c
 9: (0xb6fd60d4) [/opt/usr/apps/org.tizen.other/bin/other] + 0x20d4
End of Call Stack

Package Information
Package Name: org.tizen.other
Package ID : org.tizen.other
Version: 1.0.0
Package Type: rpm
App Name: Nornenjs
App ID: org.tizen.other
Type: capp
Categories: 

Latest Debug Message Information
--------- beginning of /dev/log_main
RA(16019): camera_product.c: camera_preinit_framework(754) > argv[0] : camera
07-06 14:01:28.330+0900 I/TIZEN_N_CAMERA(16019): camera_product.c: camera_preinit_framework(754) > argv[1] : --gst-disable-registry-fork
07-06 14:01:28.340+0900 W/CAM_SERVICE( 1174): cam_service.c: __app_context_status_cb(89) > [33mEND[0m
07-06 14:01:28.340+0900 E/socket.io(15905): 588: Client Disconnected.
07-06 14:01:28.350+0900 E/socket.io(15905): 602: Close code 1000
07-06 14:01:28.350+0900 E/socket.io(15905): clear timers
07-06 14:01:28.370+0900 E/socket.io(15905): 800: ping exit, con is expired? 1, ec: Operation canceled
07-06 14:01:28.380+0900 I/EFL-ASSIST(16019): efl_assist_theme.c: _theme_changeable_ui_data_set(222) > changeable state [1] is set to ecore_evas [b7c70308]
07-06 14:01:28.400+0900 I/EFL-ASSIST(16019): efl_assist_theme_color.cpp: ea_theme_color_table_new(763) > color table (b21066b8) from (/usr/share/themes/ChangeableColorTable2.xml) is created
07-06 14:01:28.420+0900 I/EFL-ASSIST(16019): efl_assist_theme_color.cpp: ea_theme_color_table_free(781) > color table (b21066b8) is freed
07-06 14:01:28.430+0900 I/EFL-ASSIST(16019): efl_assist_theme_color.cpp: ea_theme_color_table_new(763) > color table (b21066a0) from (/usr/apps/com.samsung.camera-app-lite/shared/res/tables/com.samsung.camera-app-lite_ChangeableColorInfo.xml) is created
07-06 14:01:28.440+0900 I/TIZEN_N_CAMERA(16019): camera_product.c: camera_preinit_framework(772) > release - argv[0] : camera
07-06 14:01:28.440+0900 I/TIZEN_N_CAMERA(16019): camera_product.c: camera_preinit_framework(772) > release - argv[1] : --gst-disable-registry-fork
07-06 14:01:28.440+0900 I/TIZEN_N_CAMERA(16019): camera_product.c: camera_preload_framework(804) > start load plugin
07-06 14:01:28.450+0900 I/EFL-ASSIST(16019): efl_assist_theme_font.c: ea_theme_font_table_new(393) > font table (b210c350) from (/usr/apps/com.samsung.camera-app-lite/shared/res/tables/com.samsung.camera-app-lite_ChangeableFontInfo.xml) is created
07-06 14:01:28.470+0900 I/CAPI_CONTENT_MEDIA_CONTENT(16019): media_content.c: media_content_connect(854) > [32m[16019]ref count : 0
07-06 14:01:28.490+0900 I/CAPI_CONTENT_MEDIA_CONTENT(16019): media_content.c: media_content_connect(886) > [32m[16019]ref count changed to: 1
07-06 14:01:28.490+0900 W/CAM_APP (16019): cam.c: cam_create(223) > [33m############## cam_create END ##############[0m
07-06 14:01:28.490+0900 I/CAPI_APPFW_APPLICATION(14072): app_main.c: ui_app_remove_event_handler(764) > handler list is not initialized
07-06 14:01:28.510+0900 I/CAPI_APPFW_APPLICATION(16019): app_main.c: _ui_app_appcore_reset(642) > app_appcore_reset
07-06 14:01:28.510+0900 W/CAM_APP (16019): cam.c: cam_service(415) > [33m############## cam_service START ##############[0m
07-06 14:01:28.510+0900 W/CAPI_APPFW_APP_CONTROL(16019): app_control.c: app_control_error(135) > [app_control_get_extra_data] KEY_NOT_FOUND(0xffffff82)
07-06 14:01:28.510+0900 W/CAM_APP (16019): cam.c: cam_service(464) > [33mapp state is [0][0m
07-06 14:01:28.510+0900 W/CAM_APP (16019): cam.c: cam_service(509) > [33mapp_control_get_operation is CAM_APP_PRELAUNCH_STATE[0m
07-06 14:01:28.510+0900 W/CAPI_APPFW_APP_CONTROL(16019): app_control.c: app_control_error(135) > [app_control_get_extra_data] KEY_NOT_FOUND(0xffffff82)
07-06 14:01:28.510+0900 W/CAM_APP (16019): cam_app.c: cam_handle_init(1524) > [33mmode : 1 [0m
07-06 14:01:28.520+0900 W/CAM_APP (16019): cam_mm.c: cam_mm_create(1641) > [33mSTART[0m
07-06 14:01:28.520+0900 W/CAM_APP (16019): cam_sound_session_manager.c: cam_sound_session_create(49) > [33mcreate sound session[0m
07-06 14:01:28.520+0900 I/TIZEN_N_SOUND_MANAGER(16019): sound_manager_product.c: sound_manager_multi_session_create(585) > >> enter : type=2, session=0xb34ae370
07-06 14:01:28.520+0900 I/TIZEN_N_SOUND_MANAGER(16019): sound_manager_product.c: sound_manager_multi_session_create(637) > << leave : type=2, session=0xb34ae370
07-06 14:01:28.520+0900 W/TIZEN_N_CAMERA(16019): camera.c: camera_create(764) > [camera_create]device name = [0]
07-06 14:01:28.540+0900 W/TIZEN_N_CAMERA(16019): camera.c: camera_create(824) > camera handle 0xb7e76600
07-06 14:01:28.540+0900 W/TIZEN_N_RECORDER(16019): recorder.c: recorder_create_videorecorder(422) > permission check done
07-06 14:01:28.540+0900 W/CAM_APP (16019): cam_mm.c: cam_mm_create(1696) > [33mEND[0m
07-06 14:01:28.560+0900 W/CAM_APP (16019): cam_app.c: cam_app_start(684) > [33m############# cam_app_start - START #############[0m
07-06 14:01:28.560+0900 E/TIZEN_N_RECORDER(16019): recorder.c: __convert_recorder_error_code(192) > [recorder_set_video_resolution] ERROR_NONE(0x00000000) : core frameworks error code(0x00000000)
07-06 14:01:28.560+0900 E/TIZEN_N_RECORDER(16019): recorder.c: __convert_recorder_error_code(192) > [recorder_attr_set_recording_flip] ERROR_NONE(0x00000000) : core frameworks error code(0x00000000)
07-06 14:01:28.560+0900 E/TIZEN_N_CAMERA(16019): camera.c: camera_attr_enable_anti_shake(3351) > NOT SUPPORTED
07-06 14:01:28.570+0900 E/CAM_APP (16019): cam_mm.c: cam_mm_set_anti_hand_shake(329) > [31mcamera_attr_enable_anti_shake failed - code[c0000002][0m
07-06 14:01:28.570+0900 I/TIZEN_N_CAMERA(16019): camera.c: camera_get_recommended_preview_resolution(1968) > recommend resolution 800x480, type 1
07-06 14:01:28.570+0900 E/CAM_APP (16019): cam_app.c: cam_app_start(701) > [31mcam_app_init_attribute failed[0m
07-06 14:01:28.570+0900 W/CAM_APP (16019): cam_app.c: cam_app_start(711) > [33mapp state is CAM_APP_PRELAUNCH_STATE. so do not start preview[0m
07-06 14:01:28.570+0900 W/CAM_APP (16019): cam_app.c: cam_app_start(729) > [33m############# cam_app_start - END #############[0m
07-06 14:01:28.600+0900 I/AUL_PAD ( 1181): sigchild.h: __launchpad_sig_child(142) > dead_pid = 14072 pgid = 14072
07-06 14:01:28.600+0900 I/AUL_PAD ( 1181): sigchild.h: __sigchild_action(123) > dead_pid(14072)
07-06 14:01:28.600+0900 I/AUL_PAD ( 1181): sigchild.h: __sigchild_action(129) > __send_app_dead_signal(0)
07-06 14:01:28.600+0900 I/AUL_PAD ( 1181): sigchild.h: __launchpad_sig_child(150) > after __sigchild_action
07-06 14:01:28.600+0900 I/AUL_AMD (  452): amd_main.c: __app_dead_handler(256) > __app_dead_handler, pid: 14072
07-06 14:01:28.600+0900 I/Tizen::System( 1047): (246) > Terminated app [com.samsung.phone]
07-06 14:01:28.600+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 14:01:28.600+0900 I/Tizen::App( 1034): (243) > App[com.samsung.phone] pid[14072] terminate event is forwarded
07-06 14:01:28.600+0900 I/Tizen::System( 1034): (256) > osp.accessorymanager.service provider is found.
07-06 14:01:28.600+0900 I/Tizen::System( 1034): (196) > Accessory Owner is removed [com.samsung.phone, 14072, ]
07-06 14:01:28.600+0900 I/Tizen::System( 1034): (256) > osp.system.service provider is found.
07-06 14:01:28.610+0900 I/Tizen::App( 1034): (506) > TerminatedApp(com.samsung.phone)
07-06 14:01:28.610+0900 I/Tizen::App( 1034): (512) > Not registered pid(14072)
07-06 14:01:28.610+0900 I/Tizen::App( 1034): (782) > Finished invoking application event listener for com.samsung.phone, 14072.
07-06 14:01:28.620+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 14:01:28.620+0900 I/Tizen::App( 1047): (782) > Finished invoking application event listener for com.samsung.phone, 14072.
07-06 14:01:28.630+0900 I/Tizen::App( 1034): (499) > LaunchedApp(com.samsung.camera-app-lite)
07-06 14:01:28.640+0900 I/Tizen::App( 1034): (733) > Finished invoking application event listener for com.samsung.camera-app-lite, 16019.
07-06 14:01:28.650+0900 I/Tizen::App( 1047): (733) > Finished invoking application event listener for com.samsung.camera-app-lite, 16019.
07-06 14:01:28.870+0900 W/CAM_APP (16019): cam_app.c: cam_app_create_main_view(1257) > [33mSTART:[0][0m
07-06 14:01:28.890+0900 W/CAM_APP (16019): cam_app.c: cam_app_destroy_main_view(1309) > [33mSTART:[0][0m
07-06 14:01:28.890+0900 W/CAM_APP (16019): cam_standby_view.c: cam_standby_view_destroy(2398) > [33mstandby_view is NULL[0m
07-06 14:01:28.890+0900 W/CAM_APP (16019): cam_app.c: cam_app_destroy_main_view(1334) > [33mEND[0m
07-06 14:01:28.920+0900 I/AUL_PAD (  491): sigchild.h: __launchpad_sig_child(142) > dead_pid = 15905 pgid = 15905
07-06 14:01:28.920+0900 I/AUL_PAD (  491): sigchild.h: __sigchild_action(123) > dead_pid(15905)
07-06 14:01:28.930+0900 I/AUL_PAD (  491): sigchild.h: __sigchild_action(129) > __send_app_dead_signal(0)
07-06 14:01:28.930+0900 I/AUL_PAD (  491): sigchild.h: __launchpad_sig_child(150) > after __sigchild_action
07-06 14:01:28.930+0900 I/AUL_AMD (  452): amd_main.c: __app_dead_handler(256) > __app_dead_handler, pid: 15905
07-06 14:01:28.930+0900 I/Tizen::System( 1047): (246) > Terminated app [org.tizen.nornenjs]
07-06 14:01:28.930+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 14:01:28.930+0900 I/Tizen::App( 1034): (243) > App[org.tizen.nornenjs] pid[15905] terminate event is forwarded
07-06 14:01:28.930+0900 I/Tizen::System( 1034): (256) > osp.accessorymanager.service provider is found.
07-06 14:01:28.930+0900 I/Tizen::System( 1034): (196) > Accessory Owner is removed [org.tizen.nornenjs, 15905, ]
07-06 14:01:28.930+0900 I/Tizen::System( 1034): (256) > osp.system.service provider is found.
07-06 14:01:28.930+0900 I/Tizen::App( 1034): (506) > TerminatedApp(org.tizen.nornenjs)
07-06 14:01:28.930+0900 I/Tizen::App( 1034): (512) > Not registered pid(15905)
07-06 14:01:28.930+0900 I/Tizen::App( 1034): (782) > Finished invoking application event listener for org.tizen.nornenjs, 15905.
07-06 14:01:28.940+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 14:01:28.940+0900 I/Tizen::App( 1047): (782) > Finished invoking application event listener for org.tizen.nornenjs, 15905.
07-06 14:01:29.080+0900 E/EFL     (16019): evas_main<16019> evas_font_dir.c:70 _evas_font_init_instance() ENTER:: evas_font_init
07-06 14:01:29.090+0900 I/TIZEN_N_CAMERA(16019): camera_product.c: camera_preload_framework(814) > done load plugin
07-06 14:01:29.090+0900 W/CAM_APP (16019): cam.c: __cam_preloading_thread_run(842) > [33mEND[0m
07-06 14:01:29.100+0900 E/EFL     (16019): evas_main<16019> evas_font_dir.c:90 evas_font_init() DONE:: evas_font_init
07-06 14:01:29.120+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 14:01:29.120+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 14:01:29.120+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 14:01:29.220+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 15
07-06 14:01:29.240+0900 I/indicator(  492): indicator_ui.c: _property_changed_cb(1238) > "app pkgname = com.samsung.homescreen, pid = 606"
07-06 14:01:29.260+0900 W/CAM_APP (16019): cam_app.c: cam_app_create_main_view(1303) > [33mEND[0m
07-06 14:01:29.260+0900 W/CAM_APP (16019): cam.c: cam_service(594) > [33mapp state is CAM_APP_PRELAUNCH_STATE. so do not show window[0m
07-06 14:01:29.280+0900 W/CAM_APP (16019): cam.c: cam_service(730) > [33m############## cam_service END ##############[0m
07-06 14:01:29.290+0900 I/APP_CORE(16019): appcore-efl.c: __do_app(509) > Legacy lifecycle: 0
07-06 14:01:29.290+0900 I/APP_CORE(16019): appcore-efl.c: __do_app(511) > [APP 16019] Initial Launching, call the resume_cb
07-06 14:01:29.290+0900 I/CAPI_APPFW_APPLICATION(16019): app_main.c: _ui_app_appcore_resume(624) > app_appcore_resume
07-06 14:01:29.290+0900 W/CAM_APP (16019): cam.c: cam_resume(345) > [33m############## cam_resume START ##############[0m
07-06 14:01:29.290+0900 W/CAM_APP (16019): cam.c: cam_resume(356) > [33mapp state is CAM_APP_PRELAUNCH_STATE[0m
07-06 14:01:29.300+0900 W/CAM_APP (16019): cam.c: __app_init_idler(926) > [33mSTART[0m
07-06 14:01:29.300+0900 W/CAM_APP (16019): cam_shot.c: cam_shot_create(254) > [33mstart[0m
07-06 14:01:29.300+0900 W/CAM_APP (16019): cam_shot.c: cam_shot_create(270) > [33mend[0m
07-06 14:01:29.300+0900 W/AUL_AMD (  452): amd_status.c: __app_terminate_timer_cb(136) > send SIGKILL: No such process
07-06 14:01:29.310+0900 W/CAM_APP (16019): cam_lbs.c: cam_lbs_init(190) > [33mstart[0m
07-06 14:01:29.320+0900 W/LOCATION(16019): module-internal.c: module_is_supported(282) > module name(gps) is found
07-06 14:01:29.340+0900 I/PRIVACY-MANAGER-CLIENT(16019): PrivacyChecker.cpp: initialize(174) > Got lock. Starting initialize
07-06 14:01:29.340+0900 I/PRIVACY-MANAGER-CLIENT(16019): PrivacyChecker.cpp: runSignalListenerThread(204) > Running g main loop for signal
07-06 14:01:29.340+0900 I/PRIVACY-MANAGER-CLIENT(16019): PrivacyChecker.cpp: initializeDbus(230) > Starting initialize
07-06 14:01:29.360+0900 W/AUL_AMD (  452): amd_status.c: __app_terminate_timer_cb(136) > send SIGKILL: No such process
07-06 14:01:29.370+0900 I/PRIVACY-MANAGER-CLIENT(16019): PrivacyChecker.cpp: initializeDbus(245) > Initialized
07-06 14:01:29.370+0900 I/PRIVACY-MANAGER-CLIENT(16019): PrivacyChecker.cpp: initialize(192) > Initialized
07-06 14:01:29.380+0900 W/LOCATION(16019): module-internal.c: module_is_supported(282) > module name(gps) is found
07-06 14:01:29.400+0900 W/LOCATION(16019): module-internal.c: module_new(253) > module (gps) open success
07-06 14:01:29.420+0900 W/AUL_AMD (  452): amd_status.c: __app_terminate_timer_cb(136) > send SIGKILL: No such process
07-06 14:01:29.430+0900 W/LOCATION(16019): module-internal.c: module_is_supported(282) > module name(wps) is found
07-06 14:01:29.440+0900 W/LOCATION(16019): module-internal.c: module_new(253) > module (wps) open success
07-06 14:01:29.460+0900 W/CAM_APP (16019): cam_lbs.c: cam_lbs_init(228) > [33mend[0m
07-06 14:01:29.460+0900 W/CAM_APP (16019): cam_dream_shot_template_info.c: cam_dream_shot_template_info_init(32) > [33mSTART[0m
07-06 14:01:29.460+0900 W/CAM_APP (16019): cam_dream_shot_template_info.c: cam_dream_shot_template_info_init(49) > [33mg_key_file_load_from_file failed: No such file or directory[0m
07-06 14:01:29.460+0900 W/CAM_APP (16019): cam_app.c: cam_app_camera_control_thread_run(8494) > [33mstart[0m
07-06 14:01:29.460+0900 W/CAM_APP (16019): cam_dream_shot_template_info.c: cam_dream_shot_template_info_init(55) > [33mload default data for dreamshot[0m
07-06 14:01:29.460+0900 W/CAM_APP (16019): cam_dream_shot_template_info.c: cam_dream_shot_template_info_init(67) > [33mEND[0m
07-06 14:01:29.460+0900 W/CAM_APP (16019): cam.c: __app_init_idler(961) > [33mEND[0m
07-06 14:01:29.470+0900 W/AUL_AMD (  452): amd_status.c: __app_terminate_timer_cb(136) > send SIGKILL: No such process
07-06 14:01:29.490+0900 W/CAM_APP (16019): cam_app.c: cam_app_join_thread_async_cb(5880) > [33mjoin thread [0][0m
07-06 14:01:29.490+0900 W/CAM_APP (16019): cam_app.c: cam_app_join_thread(8195) > [33mpthread_join 0[0m
07-06 14:01:29.490+0900 W/CAM_APP (16019): cam_app.c: cam_app_join_thread(8198) > [33mpthread_join end 0[0m
07-06 14:01:29.680+0900 I/UXT     (16059): uxt_object_manager.cpp: on_initialized(287) > Initialized.
07-06 14:01:30.080+0900 W/cluster-view(  606): mainmenu-apps-view-impl.cpp: _OnScrollComplete(2041) >  booster timer is still running on apps-view, Stop boost timer!!!
07-06 14:01:30.450+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 1
07-06 14:01:30.520+0900 I/CAPI_APPFW_APPLICATION(16066): app_main.c: ui_app_main(699) > app_efl_main
07-06 14:01:30.550+0900 I/UXT     (16066): uxt_object_manager.cpp: on_initialized(287) > Initialized.
07-06 14:01:30.580+0900 E/RESOURCED(  768): proc-main.c: resourced_proc_status_change(614) > [resourced_proc_status_change,614] available memory = 458
07-06 14:01:30.590+0900 I/CAPI_APPFW_APPLICATION(16066): app_main.c: _ui_app_appcore_create(560) > app_appcore_create
07-06 14:01:30.590+0900 I/Tizen::App( 1034): (499) > LaunchedApp(org.tizen.other)
07-06 14:01:30.590+0900 I/Tizen::App( 1034): (733) > Finished invoking application event listener for org.tizen.other, 16066.
07-06 14:01:30.600+0900 I/Tizen::App( 1047): (733) > Finished invoking application event listener for org.tizen.other, 16066.
07-06 14:01:30.860+0900 W/PROCESSMGR(  377): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=16066
07-06 14:01:30.880+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 15
07-06 14:01:30.880+0900 I/indicator(  492): indicator_ui.c: _property_changed_cb(1238) > "app pkgname = org.tizen.other, pid = 16066"
07-06 14:01:30.880+0900 I/Tizen::System( 1047): (259) > Active app [org.tizen.], current [com.samsun] 
07-06 14:01:30.880+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 14:01:30.890+0900 E/EFL     (16066): evas_main<16066> evas_font_dir.c:70 _evas_font_init_instance() ENTER:: evas_font_init
07-06 14:01:30.900+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 14:01:30.910+0900 E/EFL     (16066): evas_main<16066> evas_font_dir.c:90 evas_font_init() DONE:: evas_font_init
07-06 14:01:30.920+0900 F/socket.io(16066): thread_start
07-06 14:01:30.920+0900 F/socket.io(16066): finish 0
07-06 14:01:30.920+0900 I/CAPI_APPFW_APPLICATION(16066): app_main.c: _ui_app_appcore_reset(642) > app_appcore_reset
07-06 14:01:30.940+0900 I/APP_CORE(16066): appcore-efl.c: __do_app(509) > Legacy lifecycle: 0
07-06 14:01:30.940+0900 I/APP_CORE(16066): appcore-efl.c: __do_app(511) > [APP 16066] Initial Launching, call the resume_cb
07-06 14:01:30.940+0900 I/CAPI_APPFW_APPLICATION(16066): app_main.c: _ui_app_appcore_resume(624) > app_appcore_resume
07-06 14:01:30.940+0900 W/APP_CORE(16066): appcore-efl.c: __show_cb(822) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:4600003
07-06 14:01:30.960+0900 E/socket.io(16066): 566: Connected.
07-06 14:01:30.960+0900 E/socket.io(16066): 554: On handshake, sid
07-06 14:01:30.960+0900 E/socket.io(16066): 651: Received Message type(connect)
07-06 14:01:30.960+0900 E/socket.io(16066): 489: On Connected
07-06 14:01:30.960+0900 F/sio_packet(16066): accept()
07-06 14:01:30.960+0900 E/socket.io(16066): 743: encoded paylod length: 18
07-06 14:01:30.970+0900 E/socket.io(16066): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 14:01:30.990+0900 I/CAPI_APPFW_APPLICATION(  606): app_main.c: app_appcore_pause(202) > app_appcore_pause
07-06 14:01:30.990+0900 E/cluster-home(  606): homescreen-main.cpp: app_pause(355) >  app pause
07-06 14:01:31.060+0900 E/socket.io(16066): 669: Received Message type(Event)
07-06 14:01:31.060+0900 F/sio_packet(16066): accept()
07-06 14:01:31.060+0900 E/socket.io(16066): 743: encoded paylod length: 21
07-06 14:01:31.060+0900 E/socket.io(16066): 800: ping exit, con is expired? 0, ec: Operation canceled
07-06 14:01:31.110+0900 E/socket.io(16066): 669: Received Message type(Event)
07-06 14:01:31.110+0900 F/get_binary(16066): in get binary_message()...
07-06 14:01:33.740+0900 I/MALI    (16066): egl_platform_x11_tizen.c: tizen_update_native_surface_private(172) > [EGL-X11] surface->[0xb7d188b0] swap changed from async to sync
07-06 14:01:36.400+0900 I/CAPI_APPFW_APPLICATION(16019): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 14:01:36.400+0900 W/CAM_APP (16019): cam_sensor_control.c: cam_sensor_rotation_change(166) > [33mignore rotation callback[0m
07-06 14:01:36.410+0900 I/CAPI_APPFW_APPLICATION(16066): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 14:01:37.120+0900 I/CAPI_APPFW_APPLICATION(16019): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 14:01:37.120+0900 W/CAM_APP (16019): cam_sensor_control.c: cam_sensor_rotation_change(166) > [33mignore rotation callback[0m
07-06 14:01:37.120+0900 I/CAPI_APPFW_APPLICATION(16066): app_main.c: _ui_app_appcore_rotation_event(490) > _ui_app_appcore_rotation_event
07-06 14:01:37.300+0900 W/STARTER (  525): hw_key.c: _key_press_cb(673) > [_key_press_cb:673] Home Key is pressed
07-06 14:01:37.300+0900 W/STARTER (  525): hw_key.c: _key_press_cb(691) > [_key_press_cb:691] homekey count : 1
07-06 14:01:37.440+0900 W/STARTER (  525): hw_key.c: _key_release_cb(503) > [_key_release_cb:503] Home Key is released
07-06 14:01:37.470+0900 I/SYSPOPUP(  595): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
07-06 14:01:37.470+0900 I/SYSPOPUP(  595): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
07-06 14:01:37.470+0900 E/VOLUME  (  595): volume_x_event.c: volume_x_input_event_unregister(351) > [volume_x_input_event_unregister:351] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
07-06 14:01:37.470+0900 E/VOLUME  (  595): volume_control.c: volume_control_close(708) > [volume_control_close:708] Failed to unregister x input event handler
07-06 14:01:37.470+0900 E/VOLUME  (  595): volume_view.c: volume_view_setting_icon_callback_del(533) > [volume_view_setting_icon_callback_del:533] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
07-06 14:01:37.690+0900 W/STARTER (  525): hw_key.c: _homekey_timer_cb(404) > [_homekey_timer_cb:404] _homekey_timer_cb, homekey count[1], lock state[0]
07-06 14:01:37.700+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 0
07-06 14:01:37.710+0900 I/AUL     (  452): menu_db_util.h: _get_app_info_from_db_by_apppath(240) > path : /usr/bin/starter, ret : 0
07-06 14:01:37.710+0900 E/AUL_AMD (  452): amd_appinfo.c: appinfo_get_value(791) > appinfo get value: Invalid argument, 24
07-06 14:01:37.720+0900 W/AUL_AMD (  452): amd_launch.c: __nofork_processing(1083) > __nofork_processing, cmd: 0, pid: 606
07-06 14:01:37.730+0900 I/CAPI_APPFW_APPLICATION(  606): app_main.c: app_appcore_reset(245) > app_appcore_reset
07-06 14:01:37.730+0900 W/AUL_AMD (  452): amd_request.c: __send_home_launch_signal(441) > send a home launch signal
07-06 14:01:37.730+0900 W/AUL_AMD (  452): amd_launch.c: __reply_handler(851) > listen fd(32) , send fd(31), pid(606), cmd(0)
07-06 14:01:37.740+0900 E/STARTER (  525): dbus-util.c: starter_dbus_home_raise_signal_send(168) > [starter_dbus_home_raise_signal_send:168] Sending HOME RAISE signal, result:0
07-06 14:01:37.740+0900 I/APP_CORE(  606): appcore-efl.c: __do_app(516) > Legacy lifecycle: 1
07-06 14:01:37.750+0900 W/AUL_AMD (  452): amd_key.c: _key_ungrab(250) > fail(-1) to ungrab key(XF86Stop)
07-06 14:01:37.750+0900 W/AUL_AMD (  452): amd_launch.c: __e17_status_handler(2132) > back key ungrab error
07-06 14:01:37.760+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 14:01:37.760+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 14:01:37.760+0900 W/test-log(  606): mainmenu-page-impl.cpp: SetPageEditMode(554) >  editState:[1]
07-06 14:01:37.830+0900 W/PROCESSMGR(  377): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=606
07-06 14:01:37.840+0900 I/MALI    (16066): egl_platform_x11_tizen.c: tizen_update_native_surface_private(181) > [EGL-X11] surface->[0xb7d188b0] swap changed from sync to async
07-06 14:01:37.850+0900 W/STARTER (  525): hw_key.c: _key_press_cb(673) > [_key_press_cb:673] Home Key is pressed
07-06 14:01:37.850+0900 W/STARTER (  525): hw_key.c: _key_press_cb(691) > [_key_press_cb:691] homekey count : 1
07-06 14:01:37.900+0900 I/Tizen::System( 1047): (259) > Active app [com.samsun], current [org.tizen.] 
07-06 14:01:37.900+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 14:01:37.930+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 14:01:37.940+0900 I/CAPI_APPFW_APPLICATION(16066): app_main.c: _ui_app_appcore_pause(607) > app_appcore_pause
07-06 14:01:37.940+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 15
07-06 14:01:37.950+0900 I/indicator(  492): indicator_ui.c: _property_changed_cb(1238) > "app pkgname = com.samsung.homescreen, pid = 606"
07-06 14:01:38.170+0900 I/SYSPOPUP(  595): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
07-06 14:01:38.180+0900 I/SYSPOPUP(  595): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
07-06 14:01:38.180+0900 E/VOLUME  (  595): volume_x_event.c: volume_x_input_event_unregister(351) > [volume_x_input_event_unregister:351] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
07-06 14:01:38.180+0900 E/VOLUME  (  595): volume_control.c: volume_control_close(708) > [volume_control_close:708] Failed to unregister x input event handler
07-06 14:01:38.180+0900 E/VOLUME  (  595): volume_view.c: volume_view_setting_icon_callback_del(533) > [volume_view_setting_icon_callback_del:533] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
07-06 14:01:38.280+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 0
07-06 14:01:38.290+0900 I/AUL     (  452): menu_db_util.h: _get_app_info_from_db_by_apppath(240) > path : /usr/bin/starter, ret : 0
07-06 14:01:38.290+0900 E/AUL_AMD (  452): amd_appinfo.c: appinfo_get_value(791) > appinfo get value: Invalid argument, 24
07-06 14:01:38.380+0900 I/UXT     (16091): uxt_object_manager.cpp: on_initialized(287) > Initialized.
07-06 14:01:38.410+0900 E/RESOURCED(  768): proc-main.c: resourced_proc_status_change(614) > [resourced_proc_status_change,614] available memory = 453
07-06 14:01:38.440+0900 I/Tizen::App( 1047): (733) > Finished invoking application event listener for com.samsung.task-mgr, 16091.
07-06 14:01:38.450+0900 I/Tizen::App( 1034): (499) > LaunchedApp(com.samsung.task-mgr)
07-06 14:01:38.450+0900 I/Tizen::App( 1034): (733) > Finished invoking application event listener for com.samsung.task-mgr, 16091.
07-06 14:01:38.630+0900 I/EFL-ASSIST(16091): efl_assist_theme.c: _theme_changeable_ui_data_set(222) > changeable state [1] is set to ecore_evas [b7cd04b0]
07-06 14:01:38.640+0900 I/EFL-ASSIST(16091): efl_assist_theme_color.cpp: ea_theme_color_table_new(763) > color table (b7dff0b0) from (/usr/share/themes/ChangeableColorTable2.xml) is created
07-06 14:01:38.650+0900 I/EFL-ASSIST(16091): efl_assist_theme_color.cpp: ea_theme_color_table_free(781) > color table (b7dff0b0) is freed
07-06 14:01:38.740+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 12
07-06 14:01:38.740+0900 E/RUA     (16091): rua.c: rua_history_load_db(278) > rua_history_load_db ok. nrows : 2, ncols : 5
07-06 14:01:38.790+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 12
07-06 14:01:38.810+0900 W/PROCESSMGR(  377): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=16091
07-06 14:01:38.830+0900 I/APP_CORE(16091): appcore-efl.c: __do_app(509) > Legacy lifecycle: 0
07-06 14:01:38.830+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 15
07-06 14:01:38.840+0900 I/Tizen::System( 1047): (259) > Active app [com.samsun], current [com.samsun] 
07-06 14:01:38.840+0900 I/Tizen::System( 1047): (273) > Current App[com.samsun] is already actived.
07-06 14:01:38.840+0900 I/APP_CORE(16091): appcore-efl.c: __do_app(511) > [APP 16091] Initial Launching, call the resume_cb
07-06 14:01:38.850+0900 W/APP_CORE(16091): appcore-efl.c: __show_cb(822) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:4c00008
07-06 14:01:38.850+0900 E/EFL     (16091): evas_main<16091> evas_font_dir.c:70 _evas_font_init_instance() ENTER:: evas_font_init
07-06 14:01:38.860+0900 E/EFL     (16091): evas_main<16091> evas_font_dir.c:90 evas_font_init() DONE:: evas_font_init
07-06 14:01:38.860+0900 I/indicator(  492): indicator_ui.c: _property_changed_cb(1238) > "app pkgname = com.samsung.task-mgr, pid = 16091"
07-06 14:01:38.900+0900 W/AUL_AMD (  452): amd_key.c: _key_ungrab(250) > fail(-1) to ungrab key(XF86Stop)
07-06 14:01:38.900+0900 W/AUL_AMD (  452): amd_launch.c: __e17_status_handler(2132) > back key ungrab error
07-06 14:01:38.940+0900 E/TASK_MGR_LITE(16091): genlist_item.c: del_cb(758) > Deleted
07-06 14:01:39.270+0900 W/STARTER (  525): hw_key.c: _key_release_cb(503) > [_key_release_cb:503] Home Key is released
07-06 14:01:39.290+0900 I/SYSPOPUP(  595): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
07-06 14:01:39.290+0900 I/SYSPOPUP(  595): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
07-06 14:01:39.290+0900 E/VOLUME  (  595): volume_x_event.c: volume_x_input_event_unregister(351) > [volume_x_input_event_unregister:351] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
07-06 14:01:39.290+0900 E/VOLUME  (  595): volume_control.c: volume_control_close(708) > [volume_control_close:708] Failed to unregister x input event handler
07-06 14:01:39.300+0900 E/VOLUME  (  595): volume_view.c: volume_view_setting_icon_callback_del(533) > [volume_view_setting_icon_callback_del:533] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_widget.c:5012 _elm_widget_item_data_get() Elm_Widget_Item item is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_genlist.c:6112 elm_genlist_item_next_get() Elm_Widget_Item ((Elm_Widget_Item *)item) is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_widget.c:5012 _elm_widget_item_data_get() Elm_Widget_Item item is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_genlist.c:6112 elm_genlist_item_next_get() Elm_Widget_Item ((Elm_Widget_Item *)item) is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_widget.c:5012 _elm_widget_item_data_get() Elm_Widget_Item item is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_genlist.c:6112 elm_genlist_item_next_get() Elm_Widget_Item ((Elm_Widget_Item *)item) is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_widget.c:5012 _elm_widget_item_data_get() Elm_Widget_Item item is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_genlist.c:6112 elm_genlist_item_next_get() Elm_Widget_Item ((Elm_Widget_Item *)item) is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_widget.c:5012 _elm_widget_item_data_get() Elm_Widget_Item item is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_genlist.c:6112 elm_genlist_item_next_get() Elm_Widget_Item ((Elm_Widget_Item *)item) is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_widget.c:5012 _elm_widget_item_data_get() Elm_Widget_Item item is NULL
07-06 14:01:39.730+0900 E/EFL     (16091): elementary<16091> elm_genlist.c:6112 elm_genlist_item_next_get() Elm_Widget_Item ((Elm_Widget_Item *)item) is NULL
07-06 14:01:39.970+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 12
07-06 14:01:40.010+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 4
07-06 14:01:40.030+0900 W/AUL_AMD (  452): amd_launch.c: __reply_handler(851) > listen fd(32) , send fd(31), pid(16019), cmd(4)
07-06 14:01:40.030+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 22
07-06 14:01:40.030+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(803) > app status : 5
07-06 14:01:40.030+0900 I/APP_CORE(16019): appcore-efl.c: __after_loop(1057) > Legacy lifecycle: 0
07-06 14:01:40.030+0900 I/CAPI_APPFW_APPLICATION(16019): app_main.c: _ui_app_appcore_terminate(581) > app_appcore_terminate
07-06 14:01:40.030+0900 W/CAM_APP (16019): cam.c: cam_terminate(276) > [33m############## cam_terminate START ##############[0m
07-06 14:01:40.040+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 4
07-06 14:01:40.040+0900 W/CAM_APP (16019): cam_app.c: cam_app_destroy_main_view(1309) > [33mSTART:[0][0m
07-06 14:01:40.040+0900 W/CAM_APP (16019): cam_ui_switch_effect_utils.c: cam_ui_switch_effect_reset(39) > [33mstart[0m
07-06 14:01:40.040+0900 W/CAM_APP (16019): cam_app.c: cam_app_request_display_freeze(8949) > [33mstart - [0][0m
07-06 14:01:40.040+0900 W/CAM_APP (16019): cam_app.c: cam_app_request_display_freeze(8961) > [33mskip, display_freeze is FALSE[0m
07-06 14:01:40.040+0900 W/CAM_APP (16019): cam_ui_switch_effect_utils.c: cam_ui_switch_effect_reset(49) > [33mend[0m
07-06 14:01:40.040+0900 W/CAM_APP (16019): cam_auto_trigger_layout.c: cam_auto_trigger_layout_destroy(823) > [33mauto_trigger_camera_layout is not exist[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_destroy_main_view(1334) > [33mEND[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_shot.c: cam_shot_destroy(225) > [33mSTART:[0][0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_shot.c: cam_shot_destroy(241) > [33mEND[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_camera_control_thread_run(8504) > [33msignal received[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_camera_control_thread_run(8508) > [33mcmd is CAM_CTRL_THREAD_EXIT[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_camera_control_thread_run(8542) > [33mthread exit[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_auto_trigger.c: cam_auto_trigger_face_detect_thread_exit(228) > [33mstart[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_auto_trigger.c: cam_auto_trigger_face_detect_thread_exit(231) > [33mCAM_THREAD_AUTO_TRIGGER_FACE_DETECT is NULL[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_panorama_burst_shot.c: cam_panorama_burst_shot_exit_thread(229) > [33mstart[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_panorama_burst_shot.c: cam_panorama_burst_shot_exit_thread(267) > [33mend[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_join_thread(8195) > [33mpthread_join 1[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_join_thread(8198) > [33mpthread_join end 1[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_join_thread(8195) > [33mpthread_join 2[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_join_thread(8198) > [33mpthread_join end 2[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_stop(997) > [33m############# cam_app_stop - START #############[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_request_display_freeze(8949) > [33mstart - [0][0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_request_display_freeze(8961) > [33mskip, display_freeze is FALSE[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_app.c: cam_app_preview_stop(1149) > [33mcam_app_preview_stop - START[0m
07-06 14:01:40.050+0900 E/CAM_APP (16019): cam_app.c: cam_app_preview_stop(1162) > [31mcam_mm_preview_stop failed[0m
07-06 14:01:40.050+0900 E/CAM_APP (16019): cam_app.c: cam_app_stop(1030) > [31mcam_app_preview_stop failed[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_mm.c: cam_mm_destory(1705) > [33mSTART[0m
07-06 14:01:40.050+0900 W/CAM_APP (16019): cam_sound_session_manager.c: cam_sound_session_destroy(68) > [33mdestroy sound session[0m
07-06 14:01:40.060+0900 I/TIZEN_N_SOUND_MANAGER(16019): sound_manager_product.c: sound_manager_multi_session_destroy(850) > >> enter : session=0xb7d54700
07-06 14:01:40.070+0900 I/TIZEN_N_SOUND_MANAGER(16019): sound_manager_product.c: sound_manager_multi_session_destroy(911) > << leave : already set same option(0), skip it
07-06 14:01:40.070+0900 I/TIZEN_N_SOUND_MANAGER(16019): sound_manager_product.c: sound_manager_multi_session_destroy(920) > << leave : session=(nil)
07-06 14:01:40.070+0900 E/TIZEN_N_RECORDER(16019): recorder.c: __convert_recorder_error_code(192) > [recorder_destroy] ERROR_NONE(0x00000000) : core frameworks error code(0x00000000)
07-06 14:01:40.070+0900 W/TIZEN_N_CAMERA(16019): camera.c: camera_destroy(844) > camera handle 0xb7e76600
07-06 14:01:40.070+0900 I/TIZEN_N_CAMERA(16019): camera.c: _camera_remove_cb_message(92) > start
07-06 14:01:40.070+0900 W/TIZEN_N_CAMERA(16019): camera.c: _camera_remove_cb_message(118) > There is no remained callback
07-06 14:01:40.070+0900 I/TIZEN_N_CAMERA(16019): camera.c: _camera_remove_cb_message(123) > done
07-06 14:01:40.070+0900 W/CAM_APP (16019): cam_mm.c: cam_mm_destory(1735) > [33mEND[0m
07-06 14:01:40.090+0900 W/AUL_AMD (  452): amd_launch.c: __reply_handler(851) > listen fd(32) , send fd(31), pid(16066), cmd(4)
07-06 14:01:40.090+0900 W/CAM_APP (16019): cam_lbs.c: cam_lbs_finialize(246) > [33mstart[0m
07-06 14:01:40.090+0900 I/PRIVACY-MANAGER-CLIENT(16019): PrivacyChecker.cpp: finalize(487) > Quit signal to dbus thread
07-06 14:01:40.090+0900 W/CAM_APP (16019): cam_lbs.c: cam_lbs_finialize(262) > [33mend[0m
07-06 14:01:40.090+0900 I/PRIVACY-MANAGER-CLIENT(16019): PrivacyChecker.cpp: runSignalListenerThread(218) > Thread Exit
07-06 14:01:40.110+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 22
07-06 14:01:40.110+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(803) > app status : 5
07-06 14:01:40.110+0900 I/CAPI_CONTENT_MEDIA_CONTENT(16019): media_content.c: media_content_disconnect(942) > [32m[16019]ref count changed to: 0
07-06 14:01:40.110+0900 E/TASK_MGR_LITE(16091): genlist_item.c: del_cb(758) > Deleted
07-06 14:01:40.110+0900 W/CAM_APP (16019): cam_dream_shot_template_info.c: cam_dream_shot_template_info_deinit(73) > [33mSTART[0m
07-06 14:01:40.110+0900 W/CAM_APP (16019): cam_dream_shot_template_info.c: cam_dream_shot_template_info_deinit(80) > [33mEND[0m
07-06 14:01:40.150+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 22
07-06 14:01:40.150+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(803) > app status : 5
07-06 14:01:40.150+0900 I/APP_CORE(16091): appcore-efl.c: __after_loop(1057) > Legacy lifecycle: 0
07-06 14:01:40.150+0900 I/APP_CORE(16091): appcore-efl.c: __after_loop(1059) > [APP 16091] PAUSE before termination
07-06 14:01:40.150+0900 W/CAM_APP (16019): cam_app.c: cam_app_stop(1079) > [33m############# cam_app_stop - END #############[0m
07-06 14:01:40.160+0900 E/APP_CORE(16091): appcore.c: appcore_flush_memory(631) > Appcore not initialized
07-06 14:01:40.180+0900 W/PROCESSMGR(  377): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=606
07-06 14:01:40.190+0900 I/EFL-ASSIST(16019): efl_assist_theme_color.cpp: ea_theme_color_table_free(781) > color table (b21066a0) is freed
07-06 14:01:40.190+0900 I/EFL-ASSIST(16019): efl_assist_theme_font.c: ea_theme_font_table_free(407) > color table (b210c350) is freed
07-06 14:01:40.190+0900 W/CAM_APP (16019): cam.c: cam_terminate(300) > [33m############## cam_terminate END ##############[0m
07-06 14:01:40.200+0900 I/MALI    (16091): egl_platform_x11.c: __egl_platform_terminate(306) > [EGL-X11] ################################################
07-06 14:01:40.200+0900 I/MALI    (16091): egl_platform_x11.c: __egl_platform_terminate(307) > [EGL-X11] PID=16091   close drm_fd=21 
07-06 14:01:40.200+0900 I/MALI    (16091): egl_platform_x11.c: __egl_platform_terminate(308) > [EGL-X11] ################################################
07-06 14:01:40.210+0900 I/APP_CORE(16066): appcore-efl.c: __after_loop(1057) > Legacy lifecycle: 0
07-06 14:01:40.210+0900 I/CAPI_APPFW_APPLICATION(16066): app_main.c: _ui_app_appcore_terminate(581) > app_appcore_terminate
07-06 14:01:40.220+0900 I/MALI    (16019): egl_platform_x11.c: __egl_platform_terminate(306) > [EGL-X11] ################################################
07-06 14:01:40.220+0900 I/MALI    (16019): egl_platform_x11.c: __egl_platform_terminate(307) > [EGL-X11] PID=16019   close drm_fd=21 
07-06 14:01:40.220+0900 I/MALI    (16019): egl_platform_x11.c: __egl_platform_terminate(308) > [EGL-X11] ################################################
07-06 14:01:40.230+0900 I/Tizen::System( 1047): (259) > Active app [com.samsun], current [com.samsun] 
07-06 14:01:40.230+0900 I/Tizen::System( 1047): (273) > Current App[com.samsun] is already actived.
07-06 14:01:40.340+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 15
07-06 14:01:40.360+0900 I/UXT     (16091): uxt_object_manager.cpp: on_terminating(301) > Terminating.
07-06 14:01:40.360+0900 I/indicator(  492): indicator_ui.c: _property_changed_cb(1238) > "app pkgname = com.samsung.homescreen, pid = 606"
07-06 14:01:40.370+0900 I/UXT     (16019): uxt_object_manager.cpp: on_terminating(301) > Terminating.
07-06 14:01:40.490+0900 I/AUL_PAD (  491): sigchild.h: __launchpad_sig_child(142) > dead_pid = 16091 pgid = 16091
07-06 14:01:40.490+0900 I/AUL_PAD (  491): sigchild.h: __sigchild_action(123) > dead_pid(16091)
07-06 14:01:40.490+0900 I/AUL_PAD (  491): sigchild.h: __sigchild_action(129) > __send_app_dead_signal(0)
07-06 14:01:40.490+0900 I/AUL_PAD (  491): sigchild.h: __launchpad_sig_child(150) > after __sigchild_action
07-06 14:01:40.500+0900 I/Tizen::System( 1047): (246) > Terminated app [com.samsung.task-mgr]
07-06 14:01:40.500+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 14:01:40.500+0900 I/Tizen::App( 1034): (243) > App[com.samsung.task-mgr] pid[16091] terminate event is forwarded
07-06 14:01:40.500+0900 I/Tizen::System( 1034): (256) > osp.accessorymanager.service provider is found.
07-06 14:01:40.500+0900 I/Tizen::System( 1034): (196) > Accessory Owner is removed [com.samsung.task-mgr, 16091, ]
07-06 14:01:40.500+0900 I/Tizen::System( 1034): (256) > osp.system.service provider is found.
07-06 14:01:40.500+0900 I/Tizen::App( 1034): (506) > TerminatedApp(com.samsung.task-mgr)
07-06 14:01:40.500+0900 I/Tizen::App( 1034): (512) > Not registered pid(16091)
07-06 14:01:40.500+0900 I/Tizen::App( 1034): (782) > Finished invoking application event listener for com.samsung.task-mgr, 16091.
07-06 14:01:40.500+0900 I/AUL_AMD (  452): amd_main.c: __app_dead_handler(256) > __app_dead_handler, pid: 16091
07-06 14:01:40.510+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 14:01:40.510+0900 I/Tizen::App( 1047): (782) > Finished invoking application event listener for com.samsung.task-mgr, 16091.
07-06 14:01:40.550+0900 I/AUL_PAD ( 1181): sigchild.h: __launchpad_sig_child(142) > dead_pid = 16019 pgid = 16019
07-06 14:01:40.550+0900 I/AUL_PAD ( 1181): sigchild.h: __sigchild_action(123) > dead_pid(16019)
07-06 14:01:40.550+0900 I/AUL_PAD ( 1181): sigchild.h: __sigchild_action(129) > __send_app_dead_signal(0)
07-06 14:01:40.550+0900 I/AUL_PAD ( 1181): sigchild.h: __launchpad_sig_child(150) > after __sigchild_action
07-06 14:01:40.550+0900 I/Tizen::System( 1047): (246) > Terminated app [com.samsung.camera-app-lite]
07-06 14:01:40.550+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 14:01:40.550+0900 I/Tizen::App( 1034): (243) > App[com.samsung.camera-app-lite] pid[16019] terminate event is forwarded
07-06 14:01:40.550+0900 I/Tizen::System( 1034): (256) > osp.accessorymanager.service provider is found.
07-06 14:01:40.550+0900 I/Tizen::System( 1034): (196) > Accessory Owner is removed [com.samsung.camera-app-lite, 16019, ]
07-06 14:01:40.550+0900 I/Tizen::System( 1034): (256) > osp.system.service provider is found.
07-06 14:01:40.550+0900 I/Tizen::App( 1034): (506) > TerminatedApp(com.samsung.camera-app-lite)
07-06 14:01:40.550+0900 I/Tizen::App( 1034): (512) > Not registered pid(16019)
07-06 14:01:40.550+0900 I/Tizen::App( 1034): (782) > Finished invoking application event listener for com.samsung.camera-app-lite, 16019.
07-06 14:01:40.550+0900 I/AUL_AMD (  452): amd_main.c: __app_dead_handler(256) > __app_dead_handler, pid: 16019
07-06 14:01:40.550+0900 W/CAM_SERVICE( 1174): cam_service.c: __app_context_status_cb(77) > [33mSTART[0m
07-06 14:01:40.550+0900 W/CAM_SERVICE( 1174): cam_service.c: __app_context_status_cb(82) > [33m0, com.samsung.camera-app-lite[0m
07-06 14:01:40.550+0900 W/CAM_SERVICE( 1174): cam_service.c: __camera_app_launch(29) > [33mSTART[0m
07-06 14:01:40.560+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 14
07-06 14:01:40.560+0900 W/AUL_AMD (  452): amd_request.c: __send_result_to_client(79) > __send_result_to_client, pid: -1
07-06 14:01:40.560+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 14:01:40.560+0900 I/Tizen::App( 1047): (782) > Finished invoking application event listener for com.samsung.camera-app-lite, 16019.
07-06 14:01:40.570+0900 W/AUL_AMD (  452): amd_request.c: __request_handler(601) > __request_handler: 0
07-06 14:01:40.660+0900 W/CAM_APP (16059): cam.c: main(973) > [33mmain START[0m
07-06 14:01:40.660+0900 I/CAPI_APPFW_APPLICATION(16059): app_main.c: ui_app_main(699) > app_efl_main
07-06 14:01:40.660+0900 I/CAPI_APPFW_APPLICATION(16059): app_main.c: _ui_app_appcore_create(560) > app_appcore_create
07-06 14:01:40.680+0900 E/RESOURCED(  768): proc-main.c: resourced_proc_status_change(614) > [resourced_proc_status_change,614] available memory = 457
07-06 14:01:40.690+0900 W/CAM_SERVICE( 1174): cam_service.c: __camera_app_launch(72) > [33mEND[0m
07-06 14:01:40.690+0900 W/CAM_SERVICE( 1174): cam_service.c: __app_context_status_cb(89) > [33mEND[0m
07-06 14:01:40.700+0900 W/CAM_APP (16059): cam.c: cam_create(118) > [33m############## cam_create START ##############[0m
07-06 14:01:40.700+0900 W/CAM_APP (16059): cam.c: __cam_preloading_thread_run(827) > [33mSTART[0m
07-06 14:01:40.700+0900 I/TIZEN_N_CAMERA(16059): camera_product.c: camera_preinit_framework(748) > initializing gstreamer with following parameter
07-06 14:01:40.700+0900 I/TIZEN_N_CAMERA(16059): camera_product.c: camera_preinit_framework(749) > argc : 2
07-06 14:01:40.700+0900 I/TIZEN_N_CAMERA(16059): camera_product.c: camera_preinit_framework(754) > argv[0] : camera
07-06 14:01:40.700+0900 I/TIZEN_N_CAMERA(16059): camera_product.c: camera_preinit_framework(754) > argv[1] : --gst-disable-registry-fork
07-06 14:01:40.730+0900 I/EFL-ASSIST(16059): efl_assist_theme.c: _theme_changeable_ui_data_set(222) > changeable state [1] is set to ecore_evas [b7c70328]
07-06 14:01:40.740+0900 I/EFL-ASSIST(16059): efl_assist_theme_color.cpp: ea_theme_color_table_new(763) > color table (b2108688) from (/usr/share/themes/ChangeableColorTable2.xml) is created
07-06 14:01:40.750+0900 I/AUL_PAD (  491): sigchild.h: __launchpad_sig_child(142) > dead_pid = 16066 pgid = 16066
07-06 14:01:40.750+0900 I/AUL_PAD (  491): sigchild.h: __sigchild_action(123) > dead_pid(16066)
07-06 14:01:40.750+0900 I/AUL_PAD (  491): sigchild.h: __sigchild_action(129) > __send_app_dead_signal(0)
07-06 14:01:40.750+0900 I/AUL_PAD (  491): sigchild.h: __launchpad_sig_child(150) > after __sigchild_action
07-06 14:01:40.780+0900 I/TIZEN_N_CAMERA(16059): camera_product.c: camera_preinit_framework(772) > release - argv[0] : camera
07-06 14:01:40.780+0900 I/TIZEN_N_CAMERA(16059): camera_product.c: camera_preinit_framework(772) > release - argv[1] : --gst-disable-registry-fork
07-06 14:01:40.780+0900 I/TIZEN_N_CAMERA(16059): camera_product.c: camera_preload_framework(804) > start load plugin
07-06 14:01:40.790+0900 I/Tizen::System( 1047): (246) > Terminated app [org.tizen.other]
07-06 14:01:40.790+0900 I/Tizen::Io( 1047): (729) > Entry not found
07-06 14:01:40.790+0900 I/Tizen::App( 1034): (243) > App[org.tizen.other] pid[16066] terminate event is forwarded
07-06 14:01:40.790+0900 I/Tizen::System( 1034): (256) > osp.accessorymanager.service provider is found.
07-06 14:01:40.790+0900 I/Tizen::System( 1034): (196) > Accessory Owner is removed [org.tizen.other, 16066, ]
07-06 14:01:40.790+0900 I/Tizen::System( 1034): (256) > osp.system.service provider is found.
07-06 14:01:40.790+0900 I/Tizen::App( 1034): (506) > TerminatedApp(org.tizen.other)
07-06 14:01:40.790+0900 I/Tizen::App( 1034): (512) > Not registered pid(16066)
07-06 14:01:40.790+0900 I/Tizen::App( 1034): (782) > Finished invoking application event listener for org.tizen.other, 16066.
07-06 14:01:40.790+0900 I/AUL_AMD (  452): amd_main.c: __app_dead_handler(256) > __app_dead_handler, pid: 16066
07-06 14:01:40.800+0900 I/Tizen::System( 1047): (157) > change brightness system value.
07-06 14:01:40.800+0900 I/Tizen::App( 1047): (782) > Finished invoking application event listener for org.tizen.other, 16066.
07-06 14:01:40.820+0900 I/EFL-ASSIST(16059): efl_assist_theme_color.cpp: ea_theme_color_table_free(781) > color table (b2108688) is freed
07-06 14:01:40.820+0900 I/EFL-ASSIST(16059): efl_assist_theme_color.cpp: ea_theme_color_table_new(763) > color table (b2108670) from (/usr/apps/com.samsung.camera-app-lite/shared/res/tables/com.samsung.camera-app-lite_ChangeableColorInfo.xml) is created
07-06 14:01:40.830+0900 I/EFL-ASSIST(16059): efl_assist_theme_font.c: ea_theme_font_table_new(393) > font table (b210def0) from (/usr/apps/com.samsung.camera-app-lite/shared/res/tables/com.samsung.camera-app-lite_ChangeableFontInfo.xml) is created
07-06 14:01:40.850+0900 W/CRASH_MANAGER(16106): worker.c: worker_job(1236) > 11160666f7468143615890
