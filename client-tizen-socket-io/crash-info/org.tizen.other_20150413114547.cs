S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 21823
Date: 2015-04-13 11:45:47+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 2
      invalid permissions for mapped object
      si_addr = 0xb3eb13d4

Register Information
r0   = 0xaf384100, r1   = 0xb3eaebc4
r2   = 0x00000400, r3   = 0xb3eb13d4
r4   = 0x00000010, r5   = 0xb3eaebc4
r6   = 0xaf384000, r7   = 0x00000160
r8   = 0x00000400, r9   = 0xb3eaebc4
r10  = 0x00000200, fp   = 0x000000b0
ip   = 0x00004000, sp   = 0xbec65810
lr   = 0xb3b41650, pc   = 0xb3b41664
cpsr = 0x20000010

Memory Information
MemTotal:   797840 KB
MemFree:    350352 KB
Buffers:     74460 KB
Cached:     148628 KB
VmPeak:     137340 KB
VmSize:     118820 KB
VmLck:           0 KB
VmHWM:       15124 KB
VmRSS:       15124 KB
VmData:      62924 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24992 KB
VmPTE:          76 KB
VmSwap:          0 KB

Threads Information
Threads: 8
PID = 21823 TID = 21823
21823 21827 21828 21829 21830 21831 21832 21833 

Maps Information
b17a8000 b17a9000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b17e6000 b17e7000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b17ef000 b17f6000 r-xp /usr/lib/libfeedback.so.0.1.4
b1809000 b180a000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b1812000 b1829000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b19af000 b19b4000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b31fd000 b3248000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b3251000 b325b000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b3264000 b32c0000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3acd000 b3ad3000 r-xp /usr/lib/libUMP.so
b3adb000 b3ae2000 r-xp /usr/lib/libtbm.so.1.0.0
b3aea000 b3aef000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3af7000 b3afe000 r-xp /usr/lib/libdrm.so.2.4.0
b3b07000 b3b09000 r-xp /usr/lib/libdri2.so.0.0.0
b3b11000 b3b24000 r-xp /usr/lib/libEGL_platform.so
b3b2d000 b3c04000 r-xp /usr/lib/libMali.so
b3c0f000 b3c16000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3c1e000 b3c23000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3c2b000 b3c42000 r-xp /usr/lib/libEGL.so.1.4
b3c4b000 b3c50000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3c59000 b3c5a000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3c63000 b3c7b000 r-xp /usr/lib/libpng12.so.0.50.0
b3c83000 b3cc1000 r-xp /usr/lib/libGLESv2.so.2.0
b3cc9000 b3ccd000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3cd6000 b3cd9000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3ce2000 b3d98000 r-xp /usr/lib/libcairo.so.2.11200.14
b3da3000 b3db9000 r-xp /usr/lib/libtts.so
b3dc2000 b3dd3000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3ddb000 b3ddd000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3de5000 b3ded000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3df5000 b3eb1000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3ebb000 b3ec5000 r-xp /lib/libnss_files-2.13.so
b3ed3000 b3ed5000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b40de000 b40ff000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b4108000 b4125000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b412e000 b41fc000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b4213000 b4239000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b4243000 b4245000 r-xp /usr/lib/libiniparser.so.0
b424f000 b4255000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b425e000 b4264000 r-xp /usr/lib/libappsvc.so.0.1.0
b426d000 b426f000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b4278000 b427c000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b4284000 b4288000 r-xp /usr/lib/libogg.so.0.7.1
b4290000 b42b2000 r-xp /usr/lib/libvorbis.so.0.4.3
b42ba000 b439e000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b43b2000 b43e3000 r-xp /usr/lib/libFLAC.so.8.2.0
b43ec000 b43ee000 r-xp /usr/lib/libXau.so.6.0.0
b43f6000 b4442000 r-xp /usr/lib/libssl.so.1.0.0
b444f000 b447d000 r-xp /usr/lib/libidn.so.11.5.44
b4485000 b448f000 r-xp /usr/lib/libcares.so.2.1.0
b4497000 b44dc000 r-xp /usr/lib/libsndfile.so.1.0.25
b44ea000 b44f1000 r-xp /usr/lib/libsensord-share.so
b44f9000 b450f000 r-xp /lib/libexpat.so.1.5.2
b451d000 b4520000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b4528000 b455c000 r-xp /usr/lib/libicule.so.51.1
b4565000 b4578000 r-xp /usr/lib/libxcb.so.1.1.0
b4580000 b45bb000 r-xp /usr/lib/libcurl.so.4.3.0
b45c4000 b45cd000 r-xp /usr/lib/libethumb.so.1.7.99
b5b3b000 b5bcf000 r-xp /usr/lib/libstdc++.so.6.0.16
b5be2000 b5be4000 r-xp /usr/lib/libctxdata.so.0.0.0
b5bec000 b5bf9000 r-xp /usr/lib/libremix.so.0.0.0
b5c01000 b5c02000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5c0a000 b5c21000 r-xp /usr/lib/liblua-5.1.so
b5c2a000 b5c31000 r-xp /usr/lib/libembryo.so.1.7.99
b5c39000 b5c5c000 r-xp /usr/lib/libjpeg.so.8.0.2
b5c74000 b5c8a000 r-xp /usr/lib/libsensor.so.1.1.0
b5c93000 b5ce9000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5cf6000 b5d19000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5d22000 b5d68000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5d71000 b5d84000 r-xp /usr/lib/libfribidi.so.0.3.1
b5d8c000 b5ddc000 r-xp /usr/lib/libfreetype.so.6.8.1
b5de7000 b5dea000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5df2000 b5df6000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5dfe000 b5e03000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5e0c000 b5e16000 r-xp /usr/lib/libXext.so.6.4.0
b5e1e000 b5eff000 r-xp /usr/lib/libX11.so.6.3.0
b5f0a000 b5f0d000 r-xp /usr/lib/libXtst.so.6.1.0
b5f15000 b5f1b000 r-xp /usr/lib/libXrender.so.1.3.0
b5f23000 b5f28000 r-xp /usr/lib/libXrandr.so.2.2.0
b5f30000 b5f31000 r-xp /usr/lib/libXinerama.so.1.0.0
b5f3a000 b5f42000 r-xp /usr/lib/libXi.so.6.1.0
b5f43000 b5f46000 r-xp /usr/lib/libXfixes.so.3.1.0
b5f4e000 b5f50000 r-xp /usr/lib/libXgesture.so.7.0.0
b5f58000 b5f5a000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5f62000 b5f63000 r-xp /usr/lib/libXdamage.so.1.1.0
b5f6c000 b5f72000 r-xp /usr/lib/libXcursor.so.1.0.2
b5f7b000 b5f94000 r-xp /usr/lib/libecore_con.so.1.7.99
b5f9e000 b5fa4000 r-xp /usr/lib/libecore_imf.so.1.7.99
b5fac000 b5fb4000 r-xp /usr/lib/libethumb_client.so.1.7.99
b5fbc000 b5fc0000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b5fc9000 b5fdf000 r-xp /usr/lib/libefreet.so.1.7.99
b5fe8000 b5ff1000 r-xp /usr/lib/libedbus.so.1.7.99
b5ff9000 b60de000 r-xp /usr/lib/libicuuc.so.51.1
b60f3000 b6232000 r-xp /usr/lib/libicui18n.so.51.1
b6242000 b629e000 r-xp /usr/lib/libedje.so.1.7.99
b62a8000 b62b9000 r-xp /usr/lib/libecore_input.so.1.7.99
b62c1000 b62c6000 r-xp /usr/lib/libecore_file.so.1.7.99
b62ce000 b62e7000 r-xp /usr/lib/libeet.so.1.7.99
b62f8000 b62fc000 r-xp /usr/lib/libappcore-common.so.1.1
b6305000 b63d1000 r-xp /usr/lib/libevas.so.1.7.99
b63f6000 b6417000 r-xp /usr/lib/libecore_evas.so.1.7.99
b6420000 b644f000 r-xp /usr/lib/libecore_x.so.1.7.99
b6459000 b658d000 r-xp /usr/lib/libelementary.so.1.7.99
b65a5000 b65a6000 r-xp /usr/lib/libjournal.so.0.1.0
b65af000 b667a000 r-xp /usr/lib/libxml2.so.2.7.8
b6688000 b6698000 r-xp /lib/libresolv-2.13.so
b669c000 b66b2000 r-xp /lib/libz.so.1.2.5
b66ba000 b66bc000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b66c4000 b66c9000 r-xp /usr/lib/libffi.so.5.0.10
b66d2000 b66d3000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b66db000 b66de000 r-xp /lib/libattr.so.1.1.0
b66e6000 b688e000 r-xp /usr/lib/libcrypto.so.1.0.0
b68ae000 b68c8000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b68d1000 b693a000 r-xp /lib/libm-2.13.so
b6943000 b6983000 r-xp /usr/lib/libeina.so.1.7.99
b698c000 b6994000 r-xp /usr/lib/libvconf.so.0.2.45
b699c000 b699f000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b69a7000 b69db000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b69e4000 b6ab8000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6ac4000 b6aca000 r-xp /lib/librt-2.13.so
b6ad3000 b6ad8000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6ae1000 b6ae8000 r-xp /lib/libcrypt-2.13.so
b6b18000 b6b1b000 r-xp /lib/libcap.so.2.21
b6b23000 b6b25000 r-xp /usr/lib/libiri.so
b6b2d000 b6b4c000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6b54000 b6b6a000 r-xp /usr/lib/libecore.so.1.7.99
b6b80000 b6b85000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6b8e000 b6c5e000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6c5f000 b6c6d000 r-xp /usr/lib/libail.so.0.1.0
b6c75000 b6c8c000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6c95000 b6c9f000 r-xp /lib/libunwind.so.8.0.1
b6ccd000 b6de8000 r-xp /lib/libc-2.13.so
b6df6000 b6dfe000 r-xp /lib/libgcc_s-4.6.4.so.1
b6e06000 b6e30000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6e39000 b6e3c000 r-xp /usr/lib/libbundle.so.0.1.22
b6e44000 b6e46000 r-xp /lib/libdl-2.13.so
b6e4f000 b6e52000 r-xp /usr/lib/libsmack.so.1.0.0
b6e5a000 b6ebc000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6ec6000 b6ed8000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6ee1000 b6ef5000 r-xp /lib/libpthread-2.13.so
b6f02000 b6f06000 r-xp /usr/lib/libappcore-efl.so.1.1
b6f10000 b6f12000 r-xp /usr/lib/libdlog.so.0.0.0
b6f1a000 b6f25000 r-xp /usr/lib/libaul.so.0.1.0
b6f2f000 b6f33000 r-xp /usr/lib/libsys-assert.so
b6f3c000 b6f59000 r-xp /lib/ld-2.13.so
b6f62000 b6f68000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6f70000 b6f9c000 rw-p [heap]
b6f9c000 b7196000 rw-p [heap]
bec46000 bec67000 rwxp [stack]
End of Maps Information

Callstack Information (PID:21823)
Call Stack Count: 1
 0: (0xb3b41664) [/usr/lib/libMali.so] + 0x14664
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
on_check_dead_signal(621) > [menu_daemon_check_dead_signal:621] Process 21794 is termianted
04-13 11:45:44.640+0900 D/STARTER ( 2225): menu_daemon.c: menu_daemon_check_dead_signal(646) > [menu_daemon_check_dead_signal:646] Unknown process, ignore it (dead pid 21794, home pid 2245, taskmgr pid -1)
04-13 11:45:44.640+0900 I/AUL_AMD ( 2170): amd_main.c: __app_dead_handler(257) > __app_dead_handler, pid: 21794
04-13 11:45:44.640+0900 D/AUL_AMD ( 2170): amd_key.c: _unregister_key_event(156) > ===key stack===
04-13 11:45:44.640+0900 D/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-13 11:45:44.640+0900 E/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(264) > access
04-13 11:45:44.840+0900 D/STARTER ( 2225): hw_key.c: _destroy_syspopup_cb(630) > [_destroy_syspopup_cb:630] timer for cancel key operation
04-13 11:45:44.885+0900 I/SYSPOPUP( 2244): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
04-13 11:45:44.890+0900 I/SYSPOPUP( 2244): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
04-13 11:45:44.890+0900 D/VOLUME  ( 2244): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
04-13 11:45:44.890+0900 E/VOLUME  ( 2244): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
04-13 11:45:44.890+0900 E/VOLUME  ( 2244): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
04-13 11:45:44.890+0900 D/VOLUME  ( 2244): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
04-13 11:45:44.890+0900 D/VOLUME  ( 2244): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
04-13 11:45:44.890+0900 D/VOLUME  ( 2244): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
04-13 11:45:44.890+0900 E/VOLUME  ( 2244): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
04-13 11:45:44.890+0900 D/VOLUME  ( 2244): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
04-13 11:45:44.985+0900 D/STARTER ( 2225): hw_key.c: _launch_taskmgr_cb(132) > [_launch_taskmgr_cb:132] Launch TASKMGR
04-13 11:45:44.985+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
04-13 11:45:44.990+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
04-13 11:45:45.025+0900 D/AUL     ( 2225): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : org.tizen.task-mgr
04-13 11:45:45.025+0900 D/AUL     ( 2225): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(0)
04-13 11:45:45.025+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 0
04-13 11:45:45.030+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.task-mgr
04-13 11:45:45.065+0900 D/AUL     ( 2170): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 2221, pid = 2225
04-13 11:45:45.065+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
04-13 11:45:45.090+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1770) > process_pool: false
04-13 11:45:45.090+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1773) > h/w acceleration: USE
04-13 11:45:45.090+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.task-mgr
04-13 11:45:45.090+0900 D/AUL_AMD ( 2170): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
04-13 11:45:45.090+0900 D/AUL     ( 2170): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
04-13 11:45:45.090+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.task-mgr
04-13 11:45:45.090+0900 D/AUL_PAD ( 2204): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
04-13 11:45:45.100+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 21818 /usr/apps/org.tizen.task-mgr/bin/task-mgr
04-13 11:45:45.100+0900 D/AUL_PAD (21818): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
04-13 11:45:45.100+0900 D/AUL_PAD ( 2204): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-13 11:45:45.105+0900 D/AUL_PAD (21818): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
04-13 11:45:45.105+0900 D/AUL_PAD (21818): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.task-mgr / pkg_type : rpm / app_path : /usr/apps/org.tizen.task-mgr/bin/task-mgr 
04-13 11:45:45.170+0900 D/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_anr_ping_begin_handler(406) > [PROCESSMGR] ecore_x_netwm_ping_send to the client_win=0x1c00003
04-13 11:45:45.190+0900 D/AUL_PAD (21818): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
04-13 11:45:45.190+0900 D/AUL_PAD (21818): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /usr/apps/org.tizen.task-mgr/bin/task-mgr##
04-13 11:45:45.190+0900 D/AUL_PAD (21818): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : HIDE_LAUNCH##
04-13 11:45:45.190+0900 D/AUL_PAD (21818): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_STARTTIME__##
04-13 11:45:45.190+0900 D/AUL_PAD (21818): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 6 : __AUL_CALLER_PID__##
04-13 11:45:45.190+0900 D/LAUNCH  (21818): launchpad.c: __real_launch(229) > [SECURE_LOG] [/usr/apps/org.tizen.task-mgr/bin/task-mgr:Platform:launchpad:done]
04-13 11:45:45.190+0900 E/AUL_PAD (21818): preload.h: __preload_exec(124) > dlopen("/usr/apps/org.tizen.task-mgr/bin/task-mgr") failed
04-13 11:45:45.190+0900 E/AUL_PAD (21818): preload.h: __preload_exec(126) > dlopen error: /usr/apps/org.tizen.task-mgr/bin/task-mgr: cannot dynamically load executable
04-13 11:45:45.190+0900 D/AUL_PAD (21818): launchpad.c: __normal_fork_exec(190) > start real fork and exec
04-13 11:45:45.200+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
04-13 11:45:45.200+0900 D/AUL_PAD ( 2204): sigchild.h: __send_app_launch_signal(112) > send launch signal done
04-13 11:45:45.200+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
04-13 11:45:45.200+0900 D/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-13 11:45:45.200+0900 E/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(264) > access
04-13 11:45:45.200+0900 D/RESOURCED( 2372): proc-noti.c: recv_str(87) > [recv_str,87] str is null
04-13 11:45:45.200+0900 D/RESOURCED( 2372): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2170
04-13 11:45:45.200+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 21818
04-13 11:45:45.200+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.task-mgr, pid 21818, type 4 
04-13 11:45:45.200+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.task-mgr
04-13 11:45:45.200+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.task-mgr, 21818
04-13 11:45:45.200+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.task-mgr with pkgname
04-13 11:45:45.200+0900 D/RESOURCED( 2372): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 21818
04-13 11:45:45.200+0900 D/RESOURCED( 2372): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 21818
04-13 11:45:45.200+0900 D/RESOURCED( 2372): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-13 11:45:45.200+0900 D/RESOURCED( 2372): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-13 11:45:45.200+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-13 11:45:45.200+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-13 11:45:45.200+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-13 11:45:45.200+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-13 11:45:45.200+0900 E/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 509
04-13 11:45:45.200+0900 D/RESOURCED( 2372): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-13 11:45:45.210+0900 D/AUL     ( 2225): launch.c: app_request_to_launchpad(295) > launch request result : 21818
04-13 11:45:45.215+0900 D/STARTER ( 2225): dbus-util.c: starter_dbus_set_oomadj(223) > [starter_dbus_set_oomadj:223] org.tizen.system.deviced.Process-oomadj_set : 0
04-13 11:45:45.295+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: main(461) > Application Main Function 
04-13 11:45:45.295+0900 D/LAUNCH  (21818): appcore-efl.c: appcore_efl_main(1569) > [task-mgr:Application:main:done]
04-13 11:45:45.305+0900 D/AUL     (21818): pkginfo.c: aul_app_get_appid_bypid(196) > [SECURE_LOG] appid for 21818 is org.tizen.task-mgr
04-13 11:45:45.445+0900 D/APP_CORE(21818): appcore-efl.c: __before_loop(1012) > elm_config_preferred_engine_set : opengl_x11
04-13 11:45:45.455+0900 D/AUL     (21818): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 21818 is org.tizen.task-mgr
04-13 11:45:45.455+0900 D/APP_CORE(21818): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.task-mgr/res/locale
04-13 11:45:45.455+0900 D/APP_CORE(21818): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-13 11:45:45.455+0900 D/AUL     (21818): app_sock.c: __create_server_sock(135) > pg path - already exists
04-13 11:45:45.455+0900 D/LAUNCH  (21818): appcore-efl.c: __before_loop(1035) > [task-mgr:Platform:appcore_init:done]
04-13 11:45:45.455+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: _create_app(331) > Application Create Callback 
04-13 11:45:45.620+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: create_layout(197) > create_layout
04-13 11:45:45.620+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: create_layout(216) > Resolution: HD
04-13 11:45:45.625+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: load_data(282) > load_data
04-13 11:45:45.625+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_app_list_create(625) > recent_app_list_create
04-13 11:45:45.625+0900 D/PKGMGR_INFO(21818): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_filter_foreach_appinfo(3109) > [SECURE_LOG] where = package_app_info.app_taskmanage IN ('true','True') and package_app_info.app_disable IN ('false','False')
04-13 11:45:45.625+0900 D/PKGMGR_INFO(21818): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_filter_foreach_appinfo(3115) > [SECURE_LOG] query = select DISTINCT package_app_info.*, package_app_localized_info.app_locale, package_app_localized_info.app_label, package_app_localized_info.app_icon from package_app_info LEFT OUTER JOIN package_app_localized_info ON package_app_info.app_id=package_app_localized_info.app_id and package_app_localized_info.app_locale IN ('No Locale', 'en-us') LEFT OUTER JOIN package_app_app_svc ON package_app_info.app_id=package_app_app_svc.app_id LEFT OUTER JOIN package_app_app_category ON package_app_info.app_id=package_app_app_category.app_id where package_app_info.app_taskmanage IN ('true','True') and package_app_info.app_disable IN ('false','False')
04-13 11:45:45.635+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 12
04-13 11:45:45.635+0900 D/AUL     (21818): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 283
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_app_list_create(653) > USP mode is disabled.
04-13 11:45:45.635+0900 E/RUA     (21818): rua.c: rua_history_load_db(278) > rua_history_load_db ok. nrows : 2, ncols : 5
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_app_list_create(680) > Apps in history: 2
04-13 11:45:45.635+0900 E/TASK_MGR_LITE(21818): recent_apps.c: list_retrieve_item(348) > Fail to get ai table !!
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): recent_apps.c: list_retrieve_item(444) > [org.tizen.other]
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): [/opt/usr/apps/org.tizen.other/shared/res/other.png]
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): [other]
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_app_list_create(746) > App added into the history_list : : pkgid:org.tizen.other - appid:org.tizen.other
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_app_list_create(773) > HISTORY LIST: 0x1741f8 1
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_app_list_create(774) > RUNNING LIST: (nil) 0
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: load_data(292) > LISTs : 0x1bf8d8, 0x1ba058
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: load_data(304) > App list should display !!
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): genlist.c: genlist_init(258) > in gen list: (nil) 0x1741f8
04-13 11:45:45.635+0900 D/TASK_MGR_LITE(21818): genlist.c: recent_app_panel_create(98) > Creating task_mgr_genlist widget, noti count is : [-1]
04-13 11:45:45.655+0900 D/TASK_MGR_LITE(21818): genlist_item.c: genlist_item_class_create(308) > Genlist item class create
04-13 11:45:45.655+0900 D/TASK_MGR_LITE(21818): genlist_item.c: genlist_clear_all_class_create(323) > Genlist clear all class create
04-13 11:45:45.665+0900 D/TASK_MGR_LITE(21818): genlist.c: genlist_update(432) > genlist_update
04-13 11:45:45.665+0900 D/TASK_MGR_LITE(21818): genlist.c: genlist_clear_list(297) > genlist_clear_list
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): genlist.c: add_clear_btn_to_genlist(417) > add_clear_btn_to_genlist
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): genlist.c: genlist_update(449) > Are there recent apps: 0x1741f8
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): genlist.c: genlist_update(450) > Adding recent apps... 1
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): genlist.c: genlist_add_item(165) > genlist_add_item
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): genlist.c: genlist_add_item(171) > Adding item: 0x1bd5c0
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_apps_find_by_appid(210) > recent_apps_find_by_appid
04-13 11:45:45.670+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 12
04-13 11:45:45.670+0900 D/AUL     (21818): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 283
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.lockscreen (2228)
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.volume (2244)
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.menu-screen (2245)
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.task-mgr (21818)
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): genlist.c: genlist_item_show(192) > genlist_item_show
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): genlist.c: genlist_item_show(199) > ELM COUNT = 2
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: load_data(323) > load_data END. 
04-13 11:45:45.670+0900 D/LAUNCH  (21818): appcore-efl.c: __before_loop(1045) > [task-mgr:Application:create:done]
04-13 11:45:45.670+0900 D/APP_CORE(21818): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-13 11:45:45.670+0900 D/APP_CORE(21818): appcore.c: __aul_handler(423) > [APP 21818]     AUL event: AUL_START
04-13 11:45:45.670+0900 D/APP_CORE(21818): appcore-efl.c: __do_app(470) > [APP 21818] Event: RESET State: CREATED
04-13 11:45:45.670+0900 D/APP_CORE(21818): appcore-efl.c: __do_app(496) > [APP 21818] RESET
04-13 11:45:45.670+0900 D/LAUNCH  (21818): appcore-efl.c: __do_app(498) > [task-mgr:Application:reset:start]
04-13 11:45:45.670+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: _reset_app(446) > _reset_app
04-13 11:45:45.670+0900 D/LAUNCH  (21818): appcore-efl.c: __do_app(501) > [task-mgr:Application:reset:done]
04-13 11:45:45.670+0900 E/PKGMGR_INFO(21818): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_get_appinfo(1308) > (appid == NULL) appid is NULL
04-13 11:45:45.670+0900 I/APP_CORE(21818): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-13 11:45:45.670+0900 I/APP_CORE(21818): appcore-efl.c: __do_app(509) > [APP 21818] Initial Launching, call the resume_cb
04-13 11:45:45.670+0900 D/APP_CORE(21818): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
04-13 11:45:45.675+0900 D/APP_CORE(21818): appcore.c: __prt_ltime(183) > [APP 21818] first idle after reset: 650 msec
04-13 11:45:45.680+0900 D/APP_CORE(21818): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2600008
04-13 11:45:45.680+0900 D/APP_CORE(21818): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2600008
04-13 11:45:45.710+0900 W/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=21818
04-13 11:45:45.715+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1c00003"
04-13 11:45:45.715+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2600008"
04-13 11:45:45.720+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-13 11:45:45.720+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2600008"
04-13 11:45:45.720+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-13 11:45:45.760+0900 D/RESOURCED( 2372): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 21818, type = 0
04-13 11:45:45.760+0900 D/AUL_AMD ( 2170): amd_launch.c: __e17_status_handler(1888) > pid(21818) status(3)
04-13 11:45:45.760+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 21818
04-13 11:45:45.760+0900 D/RESOURCED( 2372): cpu.c: cpu_foreground_state(92) > [cpu_foreground_state,92] cpu_foreground_state : pid = 21818, appname = (null)
04-13 11:45:45.760+0900 D/RESOURCED( 2372): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/cpu/cgroup.procs, value 21818
04-13 11:45:45.780+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _clear_all_btn_show_cb(243) > _clear_all_btn_show_cb
04-13 11:45:45.780+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
04-13 11:45:45.780+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.other/shared/res/other.png; other 0x1f37b0
04-13 11:45:45.780+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_set(169) > It is already set.
04-13 11:45:45.780+0900 D/TASK_MGR_LITE(21818): genlist_item.c: recent_app_item_create(543) > 
04-13 11:45:45.780+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_main_create(765) > 
04-13 11:45:45.790+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
04-13 11:45:45.790+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
04-13 11:45:45.790+0900 E/TASK_MGR_LITE(21818): genlist_item.c: del_cb(758) > Deleted
04-13 11:45:45.800+0900 D/APP_CORE(21818): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2600008 fully_obscured 0
04-13 11:45:45.800+0900 D/APP_CORE(21818): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
04-13 11:45:45.800+0900 D/APP_CORE(21818): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
04-13 11:45:45.800+0900 D/APP_CORE(21818): appcore-efl.c: __do_app(470) > [APP 21818] Event: RESUME State: RUNNING
04-13 11:45:45.800+0900 D/LAUNCH  (21818): appcore-efl.c: __do_app(557) > [task-mgr:Application:resume:start]
04-13 11:45:45.800+0900 D/LAUNCH  (21818): appcore-efl.c: __do_app(567) > [task-mgr:Application:resume:done]
04-13 11:45:45.800+0900 D/LAUNCH  (21818): appcore-efl.c: __do_app(569) > [task-mgr:Application:Launching:done]
04-13 11:45:45.800+0900 D/APP_CORE(21818): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
04-13 11:45:45.800+0900 E/APP_CORE(21818): appcore-efl.c: __trm_app_info_send_socket(233) > access
04-13 11:45:45.805+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _clear_all_btn_show_cb(243) > _clear_all_btn_show_cb
04-13 11:45:45.805+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
04-13 11:45:45.805+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.other/shared/res/other.png; other 0x1f37b0
04-13 11:45:45.805+0900 D/TASK_MGR_LITE(21818): genlist_item.c: recent_app_item_create(543) > 
04-13 11:45:45.805+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_main_create(765) > 
04-13 11:45:45.805+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
04-13 11:45:45.805+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
04-13 11:45:45.955+0900 D/STARTER ( 2225): hw_key.c: _key_release_cb(470) > [_key_release_cb:470] _key_release_cb : XF86Phone Released
04-13 11:45:45.955+0900 D/MENU_SCREEN( 2245): key.c: _key_release_cb(68) > Key(XF86Phone) released 1
04-13 11:45:45.955+0900 W/STARTER ( 2225): hw_key.c: _key_release_cb(498) > [_key_release_cb:498] Home Key is released
04-13 11:45:45.955+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
04-13 11:45:45.960+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
04-13 11:45:45.965+0900 D/AUL_AMD ( 2170): amd_status.c: __app_terminate_timer_cb(108) > pid(21794)
04-13 11:45:45.965+0900 D/AUL_AMD ( 2170): amd_status.c: __app_terminate_timer_cb(112) > send SIGKILL: No such process
04-13 11:45:45.985+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(989) > LIST APPEND : org.tizen.hyok
04-13 11:45:45.990+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(989) > LIST APPEND : org.tizen.other
04-13 11:45:45.990+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(989) > LIST APPEND : org.tizen.setting
04-13 11:45:45.990+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(989) > LIST APPEND : org.tizen.texture
04-13 11:45:45.995+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(989) > LIST APPEND : org.tizen.thread
04-13 11:45:45.995+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(998) > PAGE GET : 0xb71b32f8
04-13 11:45:45.995+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(1010) > LIST GET : [0] org.tizen.hyok
04-13 11:45:46.000+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(1010) > LIST GET : [1] org.tizen.other
04-13 11:45:46.000+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(1010) > LIST GET : [2] org.tizen.setting
04-13 11:45:46.000+0900 D/STARTER ( 2225): hw_key.c: _key_release_cb(536) > [_key_release_cb:536] delete cancelkey timer
04-13 11:45:46.005+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(1010) > LIST GET : [3] org.tizen.texture
04-13 11:45:46.005+0900 I/SYSPOPUP( 2244): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
04-13 11:45:46.010+0900 I/SYSPOPUP( 2244): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
04-13 11:45:46.010+0900 D/VOLUME  ( 2244): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
04-13 11:45:46.010+0900 E/VOLUME  ( 2244): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
04-13 11:45:46.010+0900 E/VOLUME  ( 2244): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
04-13 11:45:46.010+0900 D/MENU_SCREEN( 2245): page_scroller.c: page_scroller_trim_items(1010) > LIST GET : [4] org.tizen.thread
04-13 11:45:46.015+0900 D/VOLUME  ( 2244): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
04-13 11:45:46.015+0900 D/VOLUME  ( 2244): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
04-13 11:45:46.015+0900 D/VOLUME  ( 2244): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
04-13 11:45:46.020+0900 D/BADGE   ( 2245): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.hyok'], count[0]
04-13 11:45:46.020+0900 E/VOLUME  ( 2244): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
04-13 11:45:46.020+0900 D/VOLUME  ( 2244): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
04-13 11:45:46.025+0900 D/BADGE   ( 2245): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.other'], count[0]
04-13 11:45:46.030+0900 D/BADGE   ( 2245): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.setting'], count[0]
04-13 11:45:46.035+0900 D/BADGE   ( 2245): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.texture'], count[0]
04-13 11:45:46.035+0900 D/BADGE   ( 2245): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.thread'], count[0]
04-13 11:45:46.210+0900 D/AUL_AMD ( 2170): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.task-mgr /usr/apps/org.tizen.task-mgr/bin/task-mgr
04-13 11:45:46.210+0900 D/RUA     ( 2170): rua.c: rua_add_history(179) > rua_add_history start
04-13 11:45:46.230+0900 D/RUA     ( 2170): rua.c: rua_add_history(247) > rua_add_history ok
04-13 11:45:46.385+0900 D/EFL     (21818): ecore_x<21818> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=289615201 button=1
04-13 11:45:46.480+0900 D/EFL     (21818): ecore_x<21818> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=289615292 button=1
04-13 11:45:46.480+0900 D/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600044  register trigger_timer!  pointed_win=0x657df9 
04-13 11:45:46.480+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
04-13 11:45:46.480+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected list 0.000000 no_shutdown: 0
04-13 11:45:46.480+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
04-13 11:45:46.480+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected layout 0.000000 no_shutdown: 0
04-13 11:45:46.480+0900 D/TASK_MGR_LITE(21818): genlist_item.c: clear_all_btn_clicked_cb(218) > Removing all items...
04-13 11:45:46.480+0900 D/TASK_MGR_LITE(21818): genlist_item.c: clear_all_btn_clicked_cb(230) > On REDWOOD target (HD) : No Animation.
04-13 11:45:46.480+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_apps_kill_all(164) > recent_apps_kill_all
04-13 11:45:46.480+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 12
04-13 11:45:46.480+0900 D/AUL     (21818): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 283
04-13 11:45:46.495+0900 D/TASK_MGR_LITE(21818): recent_apps.c: get_app_taskmanage(121) > app org.tizen.lockscreen is taskmanage: 0
04-13 11:45:46.500+0900 D/TASK_MGR_LITE(21818): recent_apps.c: get_app_taskmanage(121) > app org.tizen.volume is taskmanage: 0
04-13 11:45:46.510+0900 D/TASK_MGR_LITE(21818): recent_apps.c: get_app_taskmanage(121) > app org.tizen.menu-screen is taskmanage: 0
04-13 11:45:46.520+0900 D/TASK_MGR_LITE(21818): recent_apps.c: get_app_taskmanage(121) > app org.tizen.task-mgr is taskmanage: 0
04-13 11:45:46.520+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_del(189) > _recent_app_item_content_del
04-13 11:45:46.520+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_del(194) > Data exist: 0x2abab0
04-13 11:45:46.520+0900 E/TASK_MGR_LITE(21818): genlist_item.c: del_cb(758) > Deleted
04-13 11:45:46.520+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _recent_app_item_content_del(198) > Item deleted
04-13 11:45:46.520+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _clear_all_button_del(206) > _clear_all_button_del
04-13 11:45:46.525+0900 D/TASK_MGR_LITE(21818): genlist_item.c: _clear_all_button_del(213) > _clear_all_button_del END
04-13 11:45:46.540+0900 D/AUL     (21818): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
04-13 11:45:46.540+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 22
04-13 11:45:46.540+0900 D/APP_CORE(21818): appcore-efl.c: __after_loop(1060) > [APP 21818] PAUSE before termination
04-13 11:45:46.540+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: _pause_app(453) > _pause_app
04-13 11:45:46.540+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: _terminate_app(393) > _terminate_app
04-13 11:45:46.540+0900 D/TASK_MGR_LITE(21818): genlist.c: genlist_clear_list(297) > genlist_clear_list
04-13 11:45:46.545+0900 D/TASK_MGR_LITE(21818): genlist.c: recent_app_panel_del_cb(84) > recent_app_panel_del_cb
04-13 11:45:46.545+0900 D/TASK_MGR_LITE(21818): genlist_item.c: genlist_item_class_destroy(340) > Genlist class free
04-13 11:45:46.545+0900 D/TASK_MGR_LITE(21818): task-mgr-lite.c: delete_layout(272) > delete_layout
04-13 11:45:46.545+0900 D/TASK_MGR_LITE(21818): recent_apps.c: recent_app_list_destroy(791) > recent_app_list_destroy
04-13 11:45:46.545+0900 D/TASK_MGR_LITE(21818): recent_apps.c: list_destroy(475) > START list_destroy
04-13 11:45:46.545+0900 D/TASK_MGR_LITE(21818): recent_apps.c: list_destroy(487) > FREE ALL list_destroy
04-13 11:45:46.545+0900 D/TASK_MGR_LITE(21818): recent_apps.c: list_destroy(493) > FREING: list_destroy org.tizen.other
04-13 11:45:46.545+0900 D/TASK_MGR_LITE(21818): recent_apps.c: list_unretrieve_item(295) > FREING 0x1bd5c0 org.tizen.other 's Item(list_type_default_s) in list_unretrieve_item
04-13 11:45:46.545+0900 D/TASK_MGR_LITE(21818): recent_apps.c: list_unretrieve_item(333) > list_unretrieve_item END 
04-13 11:45:46.545+0900 D/TASK_MGR_LITE(21818): recent_apps.c: list_destroy(500) > END list_destroy
04-13 11:45:46.565+0900 E/APP_CORE(21818): appcore.c: appcore_flush_memory(583) > Appcore not initialized
04-13 11:45:46.575+0900 W/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=2245
04-13 11:45:46.585+0900 D/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_wininfo_del(141) > [PROCESSMGR] delete anr_trigger_timer!
04-13 11:45:46.585+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 2600008"
04-13 11:45:46.585+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 1c00003"
04-13 11:45:46.585+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-13 11:45:46.585+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 1c00003"
04-13 11:45:46.585+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-13 11:45:46.950+0900 D/EFL     ( 2245): ecore_x<2245> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=289615766 button=1
04-13 11:45:46.950+0900 D/MENU_SCREEN( 2245): mouse.c: _down_cb(103) > Mouse down (364,1032)
04-13 11:45:47.015+0900 D/EFL     ( 2245): ecore_x<2245> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=289615834 button=1
04-13 11:45:47.015+0900 D/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600044  register trigger_timer!  pointed_win=0x60006c 
04-13 11:45:47.020+0900 D/MENU_SCREEN( 2245): mouse.c: _up_cb(120) > Mouse up (371,1018)
04-13 11:45:47.300+0900 I/AUL_PAD ( 2204): sigchild.h: __launchpad_sig_child(142) > dead_pid = 21818 pgid = 21818
04-13 11:45:47.300+0900 I/AUL_PAD ( 2204): sigchild.h: __sigchild_action(123) > dead_pid(21818)
04-13 11:45:47.300+0900 D/AUL_PAD ( 2204): sigchild.h: __send_app_dead_signal(81) > send dead signal done
04-13 11:45:47.300+0900 I/AUL_PAD ( 2204): sigchild.h: __sigchild_action(129) > __send_app_dead_signal(0)
04-13 11:45:47.300+0900 I/AUL_PAD ( 2204): sigchild.h: __launchpad_sig_child(150) > after __sigchild_action
04-13 11:45:47.320+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_app_dead_cb_lite(998) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:998:D] app dead cb call! (pid : 21818)
04-13 11:45:47.320+0900 D/STARTER ( 2225): menu_daemon.c: menu_daemon_check_dead_signal(621) > [menu_daemon_check_dead_signal:621] Process 21818 is termianted
04-13 11:45:47.320+0900 I/AUL_AMD ( 2170): amd_main.c: __app_dead_handler(257) > __app_dead_handler, pid: 21818
04-13 11:45:47.320+0900 D/AUL_AMD ( 2170): amd_key.c: _unregister_key_event(156) > ===key stack===
04-13 11:45:47.320+0900 D/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-13 11:45:47.320+0900 E/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(264) > access
04-13 11:45:47.320+0900 D/STARTER ( 2225): menu_daemon.c: menu_daemon_check_dead_signal(646) > [menu_daemon_check_dead_signal:646] Unknown process, ignore it (dead pid 21818, home pid 2245, taskmgr pid -1)
04-13 11:45:47.425+0900 D/EFL     ( 2245): ecore_x<2245> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=289616233 button=1
04-13 11:45:47.425+0900 D/MENU_SCREEN( 2245): mouse.c: _down_cb(103) > Mouse down (275,326)
04-13 11:45:47.425+0900 D/MENU_SCREEN( 2245): item_event.c: _item_down_cb(63) > ITEM: mouse down event callback is invoked for 0xb7207dc0
04-13 11:45:47.480+0900 D/EFL     ( 2245): ecore_x<2245> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=289616291 button=1
04-13 11:45:47.480+0900 D/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600044  register trigger_timer!  pointed_win=0x60006c 
04-13 11:45:47.480+0900 D/MENU_SCREEN( 2245): mouse.c: _up_cb(120) > Mouse up (274,324)
04-13 11:45:47.480+0900 D/MENU_SCREEN( 2245): item.c: _focus_clicked_cb(668) > ITEM: mouse up event callback is invoked for 0xb7207dc0
04-13 11:45:47.480+0900 D/MENU_SCREEN( 2245): layout.c: layout_enable_block(116) > Enable layout blocker
04-13 11:45:47.480+0900 D/AUL     ( 2245): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : org.tizen.other
04-13 11:45:47.480+0900 D/AUL     ( 2245): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(1)
04-13 11:45:47.480+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 1
04-13 11:45:47.480+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.other
04-13 11:45:47.480+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : org.tizen.menu-screen
04-13 11:45:47.485+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1591) > win(c00002) ecore_x_pointer_grab(1)
04-13 11:45:47.490+0900 E/AUL_AMD ( 2170): amd_launch.c: invoke_dbus_method_sync(1177) > dbus_connection_send error(org.freedesktop.DBus.Error.ServiceUnknown:The name org.tizen.system.coord was not provided by any .service files)
04-13 11:45:47.490+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1675) > org.tizen.system.coord.rotation-Degree : -74
04-13 11:45:47.490+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1770) > process_pool: false
04-13 11:45:47.490+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1773) > h/w acceleration: SYS
04-13 11:45:47.490+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.other
04-13 11:45:47.490+0900 D/AUL_AMD ( 2170): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
04-13 11:45:47.490+0900 D/AUL     ( 2170): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(1)
04-13 11:45:47.490+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.other
04-13 11:45:47.490+0900 D/AUL_PAD ( 2204): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
04-13 11:45:47.490+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 21823 /opt/usr/apps/org.tizen.other/bin/other
04-13 11:45:47.490+0900 D/AUL_PAD (21823): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
04-13 11:45:47.490+0900 D/AUL_PAD ( 2204): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
04-13 11:45:47.490+0900 D/AUL_PAD (21823): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
04-13 11:45:47.490+0900 D/AUL_PAD (21823): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.other / pkg_type : rpm / app_path : /opt/usr/apps/org.tizen.other/bin/other 
04-13 11:45:47.515+0900 D/AUL_PAD (21823): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
04-13 11:45:47.515+0900 D/AUL_PAD (21823): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /opt/usr/apps/org.tizen.other/bin/other##
04-13 11:45:47.515+0900 D/AUL_PAD (21823): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : __AUL_STARTTIME__##
04-13 11:45:47.515+0900 D/AUL_PAD (21823): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_CALLER_PID__##
04-13 11:45:47.515+0900 D/AUL_PAD (21823): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 6 : __AUL_CALLER_APPID__##
04-13 11:45:47.515+0900 D/LAUNCH  (21823): launchpad.c: __real_launch(229) > [SECURE_LOG] [/opt/usr/apps/org.tizen.other/bin/other:Platform:launchpad:done]
04-13 11:45:47.550+0900 I/CAPI_APPFW_APPLICATION(21823): app_main.c: ui_app_main(697) > app_efl_main
04-13 11:45:47.550+0900 D/LAUNCH  (21823): appcore-efl.c: appcore_efl_main(1569) > [other:Application:main:done]
04-13 11:45:47.590+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
04-13 11:45:47.590+0900 D/AUL_PAD ( 2204): sigchild.h: __send_app_launch_signal(112) > send launch signal done
04-13 11:45:47.590+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
04-13 11:45:47.590+0900 D/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
04-13 11:45:47.590+0900 E/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(264) > access
04-13 11:45:47.590+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 21823
04-13 11:45:47.590+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.other
04-13 11:45:47.590+0900 D/AUL     ( 2245): launch.c: app_request_to_launchpad(295) > launch request result : 21823
04-13 11:45:47.590+0900 D/MENU_SCREEN( 2245): item.c: item_launch(1003) > Launch app's ret : [21823]
04-13 11:45:47.590+0900 D/LAUNCH  ( 2245): item.c: item_launch(1005) > [org.tizen.other:Menuscreen:launch:done]
04-13 11:45:47.590+0900 D/MENU_SCREEN( 2245): item_event.c: _item_up_cb(85) > ITEM: mouse up event callback is invoked for 0xb7207dc0
04-13 11:45:47.595+0900 D/RESOURCED( 2372): proc-noti.c: recv_str(87) > [recv_str,87] str is null
04-13 11:45:47.595+0900 D/RESOURCED( 2372): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2170
04-13 11:45:47.595+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.other, pid 21823, type 4 
04-13 11:45:47.595+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.other, 21823
04-13 11:45:47.595+0900 D/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.other with pkgname
04-13 11:45:47.595+0900 D/RESOURCED( 2372): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 21823
04-13 11:45:47.595+0900 D/RESOURCED( 2372): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 21823
04-13 11:45:47.595+0900 D/RESOURCED( 2372): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
04-13 11:45:47.595+0900 D/RESOURCED( 2372): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
04-13 11:45:47.595+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-13 11:45:47.595+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-13 11:45:47.595+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
04-13 11:45:47.595+0900 D/RESOURCED( 2372): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
04-13 11:45:47.595+0900 E/RESOURCED( 2372): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 509
04-13 11:45:47.595+0900 D/RESOURCED( 2372): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
04-13 11:45:47.610+0900 D/APP_CORE(21823): appcore-efl.c: __before_loop(1017) > elm_config_preferred_engine_set is not called
04-13 11:45:47.620+0900 D/AUL     (21823): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 21823 is org.tizen.other
04-13 11:45:47.620+0900 D/APP_CORE(21823): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.other/res/locale
04-13 11:45:47.620+0900 D/APP_CORE(21823): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
04-13 11:45:47.620+0900 D/AUL     (21823): app_sock.c: __create_server_sock(135) > pg path - already exists
04-13 11:45:47.620+0900 D/LAUNCH  (21823): appcore-efl.c: __before_loop(1035) > [other:Platform:appcore_init:done]
04-13 11:45:47.620+0900 I/CAPI_APPFW_APPLICATION(21823): app_main.c: _ui_app_appcore_create(556) > app_appcore_create
04-13 11:45:47.785+0900 W/PROCESSMGR( 2100): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=21823
04-13 11:45:47.795+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1c00003"
04-13 11:45:47.795+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2600003"
04-13 11:45:47.795+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
04-13 11:45:47.795+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2600003"
04-13 11:45:47.795+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
04-13 11:45:47.805+0900 F/socket.io(21823): thread_start
04-13 11:45:47.805+0900 F/socket.io(21823): finish 0
04-13 11:45:47.805+0900 D/LAUNCH  (21823): appcore-efl.c: __before_loop(1045) > [other:Application:create:done]
04-13 11:45:47.810+0900 D/APP_CORE(21823): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
04-13 11:45:47.810+0900 D/APP_CORE(21823): appcore.c: __aul_handler(423) > [APP 21823]     AUL event: AUL_START
04-13 11:45:47.810+0900 D/APP_CORE(21823): appcore-efl.c: __do_app(470) > [APP 21823] Event: RESET State: CREATED
04-13 11:45:47.810+0900 D/APP_CORE(21823): appcore-efl.c: __do_app(496) > [APP 21823] RESET
04-13 11:45:47.810+0900 D/LAUNCH  (21823): appcore-efl.c: __do_app(498) > [other:Application:reset:start]
04-13 11:45:47.810+0900 I/CAPI_APPFW_APPLICATION(21823): app_main.c: _ui_app_appcore_reset(638) > app_appcore_reset
04-13 11:45:47.810+0900 D/APP_SVC (21823): appsvc.c: __set_bundle(161) > __set_bundle
04-13 11:45:47.810+0900 D/LAUNCH  (21823): appcore-efl.c: __do_app(501) > [other:Application:reset:done]
04-13 11:45:47.810+0900 F/socket.io(21823): Connect Start
04-13 11:45:47.810+0900 F/socket.io(21823): Set ConnectListener
04-13 11:45:47.810+0900 F/socket.io(21823): Set ClosetListener
04-13 11:45:47.810+0900 F/socket.io(21823): Set FaileListener
04-13 11:45:47.810+0900 F/socket.io(21823): Connect
04-13 11:45:47.810+0900 F/socket.io(21823): Lock
04-13 11:45:47.810+0900 F/socket.io(21823): !!!
04-13 11:45:47.815+0900 I/APP_CORE(21823): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
04-13 11:45:47.815+0900 I/APP_CORE(21823): appcore-efl.c: __do_app(509) > [APP 21823] Initial Launching, call the resume_cb
04-13 11:45:47.815+0900 I/CAPI_APPFW_APPLICATION(21823): app_main.c: _ui_app_appcore_resume(620) > app_appcore_resume
04-13 11:45:47.815+0900 D/APP_CORE(21823): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : org.tizen.menu-screen
04-13 11:45:47.820+0900 D/APP_CORE(21823): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2600003
04-13 11:45:47.820+0900 D/APP_CORE(21823): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2600003
04-13 11:45:47.910+0900 F/other   (21823): success!!!! -1223310336 -1223310336 
04-13 11:45:47.970+0900 W/CRASH_MANAGER(21773): worker.c: worker_job(1189) > 11218236f7468142889314
