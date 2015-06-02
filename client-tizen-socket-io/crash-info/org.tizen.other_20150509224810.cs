S/W Version Information
Model: Mobile-RD-PQ
Tizen-Version: 2.3.0
Build-Number: Tizen-2.3.0_Mobile-RD-PQ_20150311.1610
Build-Date: 2015.03.11 16:10:43

Crash Information
Process Name: other
PID: 2992
Date: 2015-05-09 22:48:10+0900
Executable File Path: /opt/usr/apps/org.tizen.other/bin/other
Signal: 11
      (SIGSEGV)
      si_code: 1
      address not mapped to object
      si_addr = 0xad4f9004

Register Information
r0   = 0xad4f9008, r1   = 0xb3dbabf3
r2   = 0x000000e4, r3   = 0x00000000
r4   = 0xb3e61f1c, r5   = 0xad4f9008
r6   = 0x00000241, r7   = 0xbee11ff0
r8   = 0xb6eb62e8, r9   = 0xb6eb62e4
r10  = 0x00000000, fp   = 0xb6f0df40
ip   = 0xb3e61f98, sp   = 0xbee11fd8
lr   = 0xb3dbabf3, pc   = 0xb6ce4150
cpsr = 0xa0000010

Memory Information
MemTotal:   797840 KB
MemFree:    438032 KB
Buffers:     17204 KB
Cached:     112668 KB
VmPeak:     175876 KB
VmSize:     147812 KB
VmLck:           0 KB
VmHWM:       44400 KB
VmRSS:       44400 KB
VmData:      91996 KB
VmStk:         136 KB
VmExe:          24 KB
VmLib:       24912 KB
VmPTE:         114 KB
VmSwap:          0 KB

Threads Information
Threads: 6
PID = 2992 TID = 2992
2992 2997 2998 2999 3001 3002 

Maps Information
b1759000 b175a000 r-xp /usr/lib/evas/modules/loaders/eet/linux-gnueabi-armv7l-1.7.99/module.so
b17a2000 b17a3000 r-xp /usr/lib/libmmfkeysound.so.0.0.0
b17ab000 b17b2000 r-xp /usr/lib/libfeedback.so.0.1.4
b17c5000 b17c6000 r-xp /usr/lib/edje/modules/feedback/linux-gnueabi-armv7l-1.0.0/module.so
b17ce000 b17e5000 r-xp /usr/lib/edje/modules/elm/linux-gnueabi-armv7l-1.0.0/module.so
b196b000 b1970000 r-xp /usr/lib/bufmgr/libtbm_exynos4412.so.0.0.0
b31b9000 b3204000 r-xp /usr/lib/libGLESv1_CM.so.1.1
b320d000 b3217000 r-xp /usr/lib/evas/modules/engines/software_generic/linux-gnueabi-armv7l-1.7.99/module.so
b3220000 b327c000 r-xp /usr/lib/evas/modules/engines/gl_x11/linux-gnueabi-armv7l-1.7.99/module.so
b3a89000 b3a8f000 r-xp /usr/lib/libUMP.so
b3a97000 b3aaa000 r-xp /usr/lib/libEGL_platform.so
b3ab3000 b3b8a000 r-xp /usr/lib/libMali.so
b3b95000 b3bac000 r-xp /usr/lib/libEGL.so.1.4
b3bb5000 b3bba000 r-xp /usr/lib/libxcb-render.so.0.0.0
b3bc3000 b3bc4000 r-xp /usr/lib/libxcb-shm.so.0.0.0
b3bcd000 b3be5000 r-xp /usr/lib/libpng12.so.0.50.0
b3bed000 b3c2b000 r-xp /usr/lib/libGLESv2.so.2.0
b3c33000 b3c37000 r-xp /usr/lib/libcapi-system-info.so.0.2.0
b3c40000 b3c42000 r-xp /usr/lib/libdri2.so.0.0.0
b3c4a000 b3c51000 r-xp /usr/lib/libdrm.so.2.4.0
b3c5a000 b3d10000 r-xp /usr/lib/libcairo.so.2.11200.14
b3d1b000 b3d31000 r-xp /usr/lib/libtts.so
b3d3a000 b3d41000 r-xp /usr/lib/libtbm.so.1.0.0
b3d49000 b3d4e000 r-xp /usr/lib/libcapi-media-tool.so.0.1.1
b3d56000 b3d67000 r-xp /usr/lib/libefl-assist.so.0.1.0
b3d6f000 b3d76000 r-xp /usr/lib/libmmutil_imgp.so.0.0.0
b3d7e000 b3d83000 r-xp /usr/lib/libmmutil_jpeg.so.0.0.0
b3d8b000 b3d8d000 r-xp /usr/lib/libefl-extension.so.0.1.0
b3d95000 b3d9d000 r-xp /usr/lib/libcapi-system-system-settings.so.0.0.2
b3da5000 b3da8000 r-xp /usr/lib/libcapi-media-image-util.so.0.1.2
b3db1000 b3e59000 r-xp /opt/usr/apps/org.tizen.other/bin/other
b3e63000 b3e6d000 r-xp /lib/libnss_files-2.13.so
b3e7b000 b3e7d000 r-xp /opt/usr/apps/org.tizen.other/lib/libboost_system.so.1.55.0
b4086000 b40a7000 r-xp /usr/lib/libsecurity-server-commons.so.1.0.0
b40b0000 b40cd000 r-xp /usr/lib/libsecurity-server-client.so.1.0.1
b40d6000 b41a4000 r-xp /usr/lib/libscim-1.0.so.8.2.3
b41bb000 b41e1000 r-xp /usr/lib/ecore/immodules/libisf-imf-module.so
b41eb000 b41ed000 r-xp /usr/lib/libiniparser.so.0
b41f7000 b41fd000 r-xp /usr/lib/libcapi-security-privilege-manager.so.0.0.3
b4206000 b420c000 r-xp /usr/lib/libappsvc.so.0.1.0
b4215000 b4217000 r-xp /usr/lib/libcapi-appfw-app-common.so.0.3.1.0
b4220000 b4224000 r-xp /usr/lib/libcapi-appfw-app-control.so.0.3.1.0
b422c000 b4230000 r-xp /usr/lib/libogg.so.0.7.1
b4238000 b425a000 r-xp /usr/lib/libvorbis.so.0.4.3
b4262000 b4346000 r-xp /usr/lib/libvorbisenc.so.2.0.6
b435a000 b438b000 r-xp /usr/lib/libFLAC.so.8.2.0
b4394000 b4396000 r-xp /usr/lib/libXau.so.6.0.0
b439e000 b43ea000 r-xp /usr/lib/libssl.so.1.0.0
b43f7000 b4425000 r-xp /usr/lib/libidn.so.11.5.44
b442d000 b4437000 r-xp /usr/lib/libcares.so.2.1.0
b443f000 b4484000 r-xp /usr/lib/libsndfile.so.1.0.25
b4492000 b4499000 r-xp /usr/lib/libsensord-share.so
b44a1000 b44b7000 r-xp /lib/libexpat.so.1.5.2
b44c5000 b44c8000 r-xp /usr/lib/libcapi-appfw-application.so.0.3.1.0
b44d0000 b4504000 r-xp /usr/lib/libicule.so.51.1
b450d000 b4520000 r-xp /usr/lib/libxcb.so.1.1.0
b4528000 b4563000 r-xp /usr/lib/libcurl.so.4.3.0
b456c000 b4575000 r-xp /usr/lib/libethumb.so.1.7.99
b5ae3000 b5b77000 r-xp /usr/lib/libstdc++.so.6.0.16
b5b8a000 b5b8c000 r-xp /usr/lib/libctxdata.so.0.0.0
b5b94000 b5ba1000 r-xp /usr/lib/libremix.so.0.0.0
b5ba9000 b5baa000 r-xp /usr/lib/libecore_imf_evas.so.1.7.99
b5bb2000 b5bc9000 r-xp /usr/lib/liblua-5.1.so
b5bd2000 b5bd9000 r-xp /usr/lib/libembryo.so.1.7.99
b5be1000 b5c04000 r-xp /usr/lib/libjpeg.so.8.0.2
b5c1c000 b5c32000 r-xp /usr/lib/libsensor.so.1.1.0
b5c3b000 b5c91000 r-xp /usr/lib/libpixman-1.so.0.28.2
b5c9e000 b5cc1000 r-xp /usr/lib/libfontconfig.so.1.5.0
b5cca000 b5d10000 r-xp /usr/lib/libharfbuzz.so.0.907.0
b5d19000 b5d2c000 r-xp /usr/lib/libfribidi.so.0.3.1
b5d34000 b5d84000 r-xp /usr/lib/libfreetype.so.6.8.1
b5d8f000 b5d92000 r-xp /usr/lib/libecore_input_evas.so.1.7.99
b5d9a000 b5d9e000 r-xp /usr/lib/libecore_ipc.so.1.7.99
b5da6000 b5dab000 r-xp /usr/lib/libecore_fb.so.1.7.99
b5db4000 b5dbe000 r-xp /usr/lib/libXext.so.6.4.0
b5dc6000 b5ea7000 r-xp /usr/lib/libX11.so.6.3.0
b5eb2000 b5eb5000 r-xp /usr/lib/libXtst.so.6.1.0
b5ebd000 b5ec3000 r-xp /usr/lib/libXrender.so.1.3.0
b5ecb000 b5ed0000 r-xp /usr/lib/libXrandr.so.2.2.0
b5ed8000 b5ed9000 r-xp /usr/lib/libXinerama.so.1.0.0
b5ee2000 b5eea000 r-xp /usr/lib/libXi.so.6.1.0
b5eeb000 b5eee000 r-xp /usr/lib/libXfixes.so.3.1.0
b5ef6000 b5ef8000 r-xp /usr/lib/libXgesture.so.7.0.0
b5f00000 b5f02000 r-xp /usr/lib/libXcomposite.so.1.0.0
b5f0a000 b5f0b000 r-xp /usr/lib/libXdamage.so.1.1.0
b5f14000 b5f1a000 r-xp /usr/lib/libXcursor.so.1.0.2
b5f23000 b5f3c000 r-xp /usr/lib/libecore_con.so.1.7.99
b5f46000 b5f4c000 r-xp /usr/lib/libecore_imf.so.1.7.99
b5f54000 b5f5c000 r-xp /usr/lib/libethumb_client.so.1.7.99
b5f64000 b5f68000 r-xp /usr/lib/libefreet_mime.so.1.7.99
b5f71000 b5f87000 r-xp /usr/lib/libefreet.so.1.7.99
b5f90000 b5f99000 r-xp /usr/lib/libedbus.so.1.7.99
b5fa1000 b6086000 r-xp /usr/lib/libicuuc.so.51.1
b609b000 b61da000 r-xp /usr/lib/libicui18n.so.51.1
b61ea000 b6246000 r-xp /usr/lib/libedje.so.1.7.99
b6250000 b6261000 r-xp /usr/lib/libecore_input.so.1.7.99
b6269000 b626e000 r-xp /usr/lib/libecore_file.so.1.7.99
b6276000 b628f000 r-xp /usr/lib/libeet.so.1.7.99
b62a0000 b62a4000 r-xp /usr/lib/libappcore-common.so.1.1
b62ad000 b6379000 r-xp /usr/lib/libevas.so.1.7.99
b639e000 b63bf000 r-xp /usr/lib/libecore_evas.so.1.7.99
b63c8000 b63f7000 r-xp /usr/lib/libecore_x.so.1.7.99
b6401000 b6535000 r-xp /usr/lib/libelementary.so.1.7.99
b654d000 b654e000 r-xp /usr/lib/libjournal.so.0.1.0
b6557000 b6622000 r-xp /usr/lib/libxml2.so.2.7.8
b6630000 b6640000 r-xp /lib/libresolv-2.13.so
b6644000 b665a000 r-xp /lib/libz.so.1.2.5
b6662000 b6664000 r-xp /usr/lib/libgmodule-2.0.so.0.3200.3
b666c000 b6671000 r-xp /usr/lib/libffi.so.5.0.10
b667a000 b667b000 r-xp /usr/lib/libgthread-2.0.so.0.3200.3
b6683000 b6686000 r-xp /lib/libattr.so.1.1.0
b668e000 b6836000 r-xp /usr/lib/libcrypto.so.1.0.0
b6856000 b6870000 r-xp /usr/lib/libpkgmgr_parser.so.0.1.0
b6879000 b68e2000 r-xp /lib/libm-2.13.so
b68eb000 b692b000 r-xp /usr/lib/libeina.so.1.7.99
b6934000 b693c000 r-xp /usr/lib/libvconf.so.0.2.45
b6944000 b6947000 r-xp /usr/lib/libSLP-db-util.so.0.1.0
b694f000 b6983000 r-xp /usr/lib/libgobject-2.0.so.0.3200.3
b698c000 b6a60000 r-xp /usr/lib/libgio-2.0.so.0.3200.3
b6a6c000 b6a72000 r-xp /lib/librt-2.13.so
b6a7b000 b6a80000 r-xp /usr/lib/libcapi-base-common.so.0.1.7
b6a89000 b6a90000 r-xp /lib/libcrypt-2.13.so
b6ac0000 b6ac3000 r-xp /lib/libcap.so.2.21
b6acb000 b6acd000 r-xp /usr/lib/libiri.so
b6ad5000 b6af4000 r-xp /usr/lib/libpkgmgr-info.so.0.0.17
b6afc000 b6b12000 r-xp /usr/lib/libecore.so.1.7.99
b6b28000 b6b2d000 r-xp /usr/lib/libxdgmime.so.1.1.0
b6b36000 b6c06000 r-xp /usr/lib/libglib-2.0.so.0.3200.3
b6c07000 b6c15000 r-xp /usr/lib/libail.so.0.1.0
b6c1d000 b6c34000 r-xp /usr/lib/libdbus-glib-1.so.2.2.2
b6c3d000 b6c47000 r-xp /lib/libunwind.so.8.0.1
b6c75000 b6d90000 r-xp /lib/libc-2.13.so
b6d9e000 b6da6000 r-xp /lib/libgcc_s-4.6.4.so.1
b6dae000 b6dd8000 r-xp /usr/lib/libdbus-1.so.3.7.2
b6de1000 b6de4000 r-xp /usr/lib/libbundle.so.0.1.22
b6dec000 b6dee000 r-xp /lib/libdl-2.13.so
b6df7000 b6dfa000 r-xp /usr/lib/libsmack.so.1.0.0
b6e02000 b6e64000 r-xp /usr/lib/libsqlite3.so.0.8.6
b6e6e000 b6e80000 r-xp /usr/lib/libprivilege-control.so.0.0.2
b6e89000 b6e9d000 r-xp /lib/libpthread-2.13.so
b6eaa000 b6eae000 r-xp /usr/lib/libappcore-efl.so.1.1
b6eb8000 b6eba000 r-xp /usr/lib/libdlog.so.0.0.0
b6ec2000 b6ecd000 r-xp /usr/lib/libaul.so.0.1.0
b6ed7000 b6edb000 r-xp /usr/lib/libsys-assert.so
b6ee4000 b6f01000 r-xp /lib/ld-2.13.so
b6f0a000 b6f10000 r-xp /usr/bin/launchpad_preloading_preinitializing_daemon
b6f18000 b6f63000 rw-p [heap]
b6f63000 b8491000 rw-p [heap]
bedf2000 bee13000 rwxp [stack]
End of Maps Information

Callstack Information (PID:2992)
Call Stack Count: 9
 0: cfree + 0x30 (0xb6ce4150) [/lib/libc.so.6] + 0x6f150
 1: app_terminate + 0x2a (0xb3dbabf3) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9bf3
 2: (0xb44c61e9) [/usr/lib/libcapi-appfw-application.so.0] + 0x11e9
 3: appcore_efl_main + 0x2c6 (0xb6eac717) [/usr/lib/libappcore-efl.so.1] + 0x2717
 4: ui_app_main + 0xb0 (0xb44c6c79) [/usr/lib/libcapi-appfw-application.so.0] + 0x1c79
 5: main + 0x10c (0xb3dbad95) [/opt/usr/apps/org.tizen.other/bin/other] + 0x9d95
 6: (0xb6f0cc6f) [/opt/usr/apps/org.tizen.other/bin/other] + 0x2c6f
 7: __libc_start_main + 0x114 (0xb6c8c82c) [/lib/libc.so.6] + 0x1782c
 8: (0xb6f0d35c) [/opt/usr/apps/org.tizen.other/bin/other] + 0x335c
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
0 D/STARTER ( 2225): hw_key.c: _key_release_cb(470) > [_key_release_cb:470] _key_release_cb : XF86Phone Released
05-09 22:48:06.645+0900 W/STARTER ( 2225): hw_key.c: _key_release_cb(498) > [_key_release_cb:498] Home Key is released
05-09 22:48:06.645+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
05-09 22:48:06.645+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
05-09 22:48:06.655+0900 D/STARTER ( 2225): hw_key.c: _key_release_cb(536) > [_key_release_cb:536] delete cancelkey timer
05-09 22:48:06.655+0900 D/STARTER ( 2225): hw_key.c: _key_release_cb(559) > [_key_release_cb:559] delete long press timer
05-09 22:48:06.655+0900 I/SYSPOPUP( 2238): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
05-09 22:48:06.660+0900 I/SYSPOPUP( 2238): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
05-09 22:48:06.660+0900 D/VOLUME  ( 2238): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
05-09 22:48:06.660+0900 E/VOLUME  ( 2238): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
05-09 22:48:06.660+0900 E/VOLUME  ( 2238): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
05-09 22:48:06.660+0900 D/VOLUME  ( 2238): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
05-09 22:48:06.660+0900 D/VOLUME  ( 2238): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
05-09 22:48:06.660+0900 D/VOLUME  ( 2238): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
05-09 22:48:06.660+0900 E/VOLUME  ( 2238): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
05-09 22:48:06.660+0900 D/VOLUME  ( 2238): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
05-09 22:48:06.790+0900 D/indicator( 2218): wifi.c: indicator_wifi_change_cb(231) > indicator_wifi_change_cb[231]	 "CONNECTION WiFi Status: 2"
05-09 22:48:06.790+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit(96) > indicator_signal_emit[96]	 "SIGNAL EMIT indicator.wifi.updown.none"
05-09 22:48:06.855+0900 W/STARTER ( 2225): hw_key.c: _homekey_timer_cb(399) > [_homekey_timer_cb:399] _homekey_timer_cb, homekey count[1], lock state[0]
05-09 22:48:06.855+0900 D/STARTER ( 2225): hw_key.c: _launch_by_home_key(175) > [_launch_by_home_key:175] lock_state : 0 
05-09 22:48:06.855+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
05-09 22:48:06.860+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
05-09 22:48:06.860+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_lock_state(366) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:366:D] Get lock state : 0
05-09 22:48:06.860+0900 D/STARTER ( 2225): menu_daemon.c: menu_daemon_open_app(136) > [menu_daemon_open_app:136] pkgname: org.tizen.menu-screen
05-09 22:48:06.860+0900 D/AUL     ( 2225): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : org.tizen.menu-screen
05-09 22:48:06.860+0900 D/AUL     ( 2225): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(1)
05-09 22:48:06.860+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 1
05-09 22:48:06.860+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.menu-screen
05-09 22:48:06.865+0900 D/AUL     ( 2170): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 2220, pid = 2225
05-09 22:48:06.865+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
05-09 22:48:06.870+0900 D/RESOURCED( 2369): proc-noti.c: recv_str(87) > [recv_str,87] str is null
05-09 22:48:06.870+0900 D/RESOURCED( 2369): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2170
05-09 22:48:06.870+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.menu-screen, pid 2240, type 5 
05-09 22:48:06.870+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_status_change(592) > [SECURE_LOG] [resourced_proc_status_change,592] resume request 2240
05-09 22:48:06.870+0900 D/AUL_AMD ( 2170): amd_launch.c: __nofork_processing(995) > __nofork_processing, cmd: 1, pid: 2240
05-09 22:48:06.870+0900 D/AUL_AMD ( 2170): amd_launch.c: __nofork_processing(999) > resume app's pid : 2240
05-09 22:48:06.870+0900 D/AUL     ( 2170): app_sock.c: __app_send_raw_with_delay_reply(434) > pid(2240) : cmd(3)
05-09 22:48:06.870+0900 D/AUL_AMD ( 2170): amd_launch.c: _resume_app(661) > resume done
05-09 22:48:06.870+0900 D/AUL_AMD ( 2170): amd_launch.c: __set_reply_handler(860) > listen fd : 28, send fd : 17
05-09 22:48:06.870+0900 D/AUL_AMD ( 2170): amd_launch.c: __nofork_processing(1002) > resume app done
05-09 22:48:06.870+0900 D/AUL_AMD ( 2170): amd_request.c: __send_home_launch_signal(364) > send dead signal done
05-09 22:48:06.870+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 2240
05-09 22:48:06.870+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.menu-screen
05-09 22:48:06.895+0900 D/AUL_AMD ( 2170): amd_launch.c: __reply_handler(784) > listen fd(28) , send fd(17), pid(2240), cmd(3)
05-09 22:48:06.895+0900 D/AUL     ( 2225): launch.c: app_request_to_launchpad(295) > launch request result : 2240
05-09 22:48:06.895+0900 D/APP_CORE( 2240): appcore.c: __aul_handler(433) > [APP 2240]     AUL event: AUL_RESUME
05-09 22:48:06.895+0900 D/STARTER ( 2225): dbus-util.c: starter_dbus_set_oomadj(223) > [starter_dbus_set_oomadj:223] org.tizen.system.deviced.Process-oomadj_set : 0
05-09 22:48:06.895+0900 D/STARTER ( 2225): dbus-util.c: _dbus_message_send(172) > [_dbus_message_send:172] dbus_connection_send, ret=1
05-09 22:48:06.895+0900 E/STARTER ( 2225): dbus-util.c: starter_dbus_home_raise_signal_send(184) > [starter_dbus_home_raise_signal_send:184] Sending HOME RAISE signal, result:0
05-09 22:48:06.900+0900 D/RESOURCED( 2369): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2240, type = 0
05-09 22:48:06.900+0900 D/AUL_AMD ( 2170): amd_launch.c: __e17_status_handler(1888) > pid(2240) status(3)
05-09 22:48:06.900+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 2240
05-09 22:48:06.900+0900 D/RESOURCED( 2369): cpu.c: cpu_foreground_state(92) > [cpu_foreground_state,92] cpu_foreground_state : pid = 2240, appname = (null)
05-09 22:48:06.900+0900 D/RESOURCED( 2369): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/cpu/cgroup.procs, value 2240
05-09 22:48:07.050+0900 W/PROCESSMGR( 2140): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=2240
05-09 22:48:07.055+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 2600003"
05-09 22:48:07.055+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 1c00003"
05-09 22:48:07.055+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
05-09 22:48:07.055+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 1c00003"
05-09 22:48:07.055+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
05-09 22:48:07.135+0900 D/APP_CORE( 2240): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:1c00003 fully_obscured 0
05-09 22:48:07.135+0900 D/APP_CORE( 2240): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active 0
05-09 22:48:07.135+0900 D/APP_CORE( 2240): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
05-09 22:48:07.135+0900 D/APP_CORE( 2240): appcore-efl.c: __do_app(470) > [APP 2240] Event: RESUME State: PAUSED
05-09 22:48:07.135+0900 D/LAUNCH  ( 2240): appcore-efl.c: __do_app(557) > [menu-screen:Application:resume:start]
05-09 22:48:07.135+0900 D/APP_CORE( 2240): appcore-efl.c: __do_app(559) > [APP 2240] RESUME
05-09 22:48:07.135+0900 I/CAPI_APPFW_APPLICATION( 2240): app_main.c: app_appcore_resume(216) > app_appcore_resume
05-09 22:48:07.135+0900 D/MENU_SCREEN( 2240): menu_screen.c: _resume_cb(547) > START RESUME
05-09 22:48:07.135+0900 D/MENU_SCREEN( 2240): page_scroller.c: page_scroller_focus(1133) > Focus set scroller(0xb71396d0), page:0, item:Nornenjs
05-09 22:48:07.135+0900 D/LAUNCH  ( 2240): appcore-efl.c: __do_app(567) > [menu-screen:Application:resume:done]
05-09 22:48:07.135+0900 D/LAUNCH  ( 2240): appcore-efl.c: __do_app(569) > [menu-screen:Application:Launching:done]
05-09 22:48:07.135+0900 D/APP_CORE( 2240): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
05-09 22:48:07.135+0900 E/APP_CORE( 2240): appcore-efl.c: __trm_app_info_send_socket(233) > access
05-09 22:48:07.135+0900 D/RESOURCED( 2369): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 2992, type = 2
05-09 22:48:07.145+0900 D/RESOURCED( 2369): cpu.c: cpu_background_state(100) > [cpu_background_state,100] cpu_background_state : pid = 2992, appname = other
05-09 22:48:07.145+0900 D/RESOURCED( 2369): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/cpu/service/cgroup.procs, value 2992
05-09 22:48:07.150+0900 D/RESOURCED( 2369): proc-process.c: proc_backgrd_manage(145) > [proc_backgrd_manage,145] BACKGRD : process 2228 set score 390 (before 360)
05-09 22:48:07.150+0900 D/RESOURCED( 2369): proc-process.c: proc_backgrd_manage(152) > [proc_backgrd_manage,152] found candidate pid = -1091844616, count = 1
05-09 22:48:07.150+0900 I/RESOURCED( 2369): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/background/cgroup.procs, pid : 2992, oom : 300
05-09 22:48:07.150+0900 E/RESOURCED( 2369): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/background/cgroup.procs open failed
05-09 22:48:07.160+0900 D/APP_CORE( 2992): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2600003 fully_obscured 1
05-09 22:48:07.160+0900 D/APP_CORE( 2992): appcore-efl.c: __visibility_cb(884) > bvisibility 0, b_active 1
05-09 22:48:07.160+0900 D/APP_CORE( 2992): appcore-efl.c: __visibility_cb(898) >  Go to Pasue state 
05-09 22:48:07.160+0900 D/APP_CORE( 2992): appcore-efl.c: __do_app(470) > [APP 2992] Event: PAUSE State: RUNNING
05-09 22:48:07.160+0900 D/APP_CORE( 2992): appcore-efl.c: __do_app(538) > [APP 2992] PAUSE
05-09 22:48:07.160+0900 I/CAPI_APPFW_APPLICATION( 2992): app_main.c: _ui_app_appcore_pause(603) > app_appcore_pause
05-09 22:48:07.160+0900 D/APP_CORE( 2992): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
05-09 22:48:07.160+0900 E/APP_CORE( 2992): appcore-efl.c: __trm_app_info_send_socket(233) > access
05-09 22:48:07.760+0900 D/STARTER ( 2225): hw_key.c: _key_press_cb(656) > [_key_press_cb:656] _key_press_cb : XF86Phone Pressed
05-09 22:48:07.760+0900 W/STARTER ( 2225): hw_key.c: _key_press_cb(668) > [_key_press_cb:668] Home Key is pressed
05-09 22:48:07.760+0900 W/STARTER ( 2225): hw_key.c: _key_press_cb(686) > [_key_press_cb:686] homekey count : 1
05-09 22:48:07.760+0900 D/STARTER ( 2225): hw_key.c: _key_press_cb(695) > [_key_press_cb:695] create long press timer
05-09 22:48:07.760+0900 D/MENU_SCREEN( 2240): key.c: _key_press_cb(147) > Key pressed 1
05-09 22:48:07.875+0900 D/AUL_AMD ( 2170): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.menu-screen /usr/apps/org.tizen.menu-screen/bin/menu-screen
05-09 22:48:07.875+0900 D/RUA     ( 2170): rua.c: rua_add_history(179) > rua_add_history start
05-09 22:48:07.890+0900 D/RUA     ( 2170): rua.c: rua_add_history(247) > rua_add_history ok
05-09 22:48:08.055+0900 D/STARTER ( 2225): hw_key.c: _destroy_syspopup_cb(630) > [_destroy_syspopup_cb:630] timer for cancel key operation
05-09 22:48:08.065+0900 I/SYSPOPUP( 2238): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
05-09 22:48:08.065+0900 I/SYSPOPUP( 2238): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
05-09 22:48:08.065+0900 D/VOLUME  ( 2238): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
05-09 22:48:08.065+0900 E/VOLUME  ( 2238): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
05-09 22:48:08.065+0900 E/VOLUME  ( 2238): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
05-09 22:48:08.070+0900 D/VOLUME  ( 2238): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
05-09 22:48:08.070+0900 D/VOLUME  ( 2238): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
05-09 22:48:08.070+0900 D/VOLUME  ( 2238): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
05-09 22:48:08.070+0900 E/VOLUME  ( 2238): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
05-09 22:48:08.070+0900 D/VOLUME  ( 2238): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
05-09 22:48:08.165+0900 D/STARTER ( 2225): hw_key.c: _launch_taskmgr_cb(132) > [_launch_taskmgr_cb:132] Launch TASKMGR
05-09 22:48:08.165+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
05-09 22:48:08.165+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
05-09 22:48:08.170+0900 D/AUL     ( 2225): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : org.tizen.task-mgr
05-09 22:48:08.170+0900 D/AUL     ( 2225): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(0)
05-09 22:48:08.175+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 0
05-09 22:48:08.175+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(536) > launch a single-instance appid: org.tizen.task-mgr
05-09 22:48:08.180+0900 D/AUL     ( 2170): pkginfo.c: aul_app_get_appid_bypid(205) > second change pgid = 2220, pid = 2225
05-09 22:48:08.180+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1466) > [SECURE_LOG] caller : (null)
05-09 22:48:08.185+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1770) > process_pool: false
05-09 22:48:08.185+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1773) > h/w acceleration: USE
05-09 22:48:08.185+0900 D/AUL_AMD ( 2170): amd_launch.c: _start_app(1775) > [SECURE_LOG] appid: org.tizen.task-mgr
05-09 22:48:08.185+0900 D/AUL_AMD ( 2170): amd_launch.c: __set_appinfo_for_launchpad(1927) > Add hwacc, taskmanage, app_path and pkg_type into bundle for sending those to launchpad.
05-09 22:48:08.185+0900 D/AUL     ( 2170): app_sock.c: __app_send_raw(264) > pid(-1) : cmd(0)
05-09 22:48:08.185+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(641) > [SECURE_LOG] pkg name : org.tizen.task-mgr
05-09 22:48:08.185+0900 D/AUL_PAD ( 2204): launchpad.c: __modify_bundle(380) > parsing app_path: No arguments
05-09 22:48:08.185+0900 D/AUL_PAD ( 2204): launchpad.c: __launchpad_main_loop(699) > [SECURE_LOG] ==> real launch pid : 3011 /usr/apps/org.tizen.task-mgr/bin/task-mgr
05-09 22:48:08.185+0900 D/AUL_PAD ( 2204): launchpad.c: __send_result_to_caller(555) > -- now wait to change cmdline --
05-09 22:48:08.185+0900 D/AUL_PAD ( 3011): launchpad.c: __launchpad_main_loop(668) > lock up test log(no error) : fork done
05-09 22:48:08.190+0900 D/AUL_PAD ( 3011): launchpad.c: __launchpad_main_loop(679) > lock up test log(no error) : prepare exec - first done
05-09 22:48:08.190+0900 D/AUL_PAD ( 3011): launchpad.c: __prepare_exec(136) > [SECURE_LOG] pkg_name : org.tizen.task-mgr / pkg_type : rpm / app_path : /usr/apps/org.tizen.task-mgr/bin/task-mgr 
05-09 22:48:08.205+0900 D/AUL_PAD ( 3011): launchpad.c: __launchpad_main_loop(693) > lock up test log(no error) : prepare exec - second done
05-09 22:48:08.205+0900 D/AUL_PAD ( 3011): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 0 : /usr/apps/org.tizen.task-mgr/bin/task-mgr##
05-09 22:48:08.205+0900 D/AUL_PAD ( 3011): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 2 : HIDE_LAUNCH##
05-09 22:48:08.205+0900 D/AUL_PAD ( 3011): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 4 : __AUL_STARTTIME__##
05-09 22:48:08.205+0900 D/AUL_PAD ( 3011): launchpad.c: __real_launch(223) > [SECURE_LOG] input argument 6 : __AUL_CALLER_PID__##
05-09 22:48:08.205+0900 D/LAUNCH  ( 3011): launchpad.c: __real_launch(229) > [SECURE_LOG] [/usr/apps/org.tizen.task-mgr/bin/task-mgr:Platform:launchpad:done]
05-09 22:48:08.205+0900 E/AUL_PAD ( 3011): preload.h: __preload_exec(124) > dlopen("/usr/apps/org.tizen.task-mgr/bin/task-mgr") failed
05-09 22:48:08.205+0900 E/AUL_PAD ( 3011): preload.h: __preload_exec(126) > dlopen error: /usr/apps/org.tizen.task-mgr/bin/task-mgr: cannot dynamically load executable
05-09 22:48:08.205+0900 D/AUL_PAD ( 3011): launchpad.c: __normal_fork_exec(190) > start real fork and exec
05-09 22:48:08.285+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_block_sigchld(230) > SIGCHLD blocked
05-09 22:48:08.285+0900 D/AUL_PAD ( 2204): sigchild.h: __send_app_launch_signal(112) > send launch signal done
05-09 22:48:08.285+0900 D/AUL_PAD ( 2204): sigchild.h: __signal_unblock_sigchld(242) > SIGCHLD unblocked
05-09 22:48:08.285+0900 D/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(261) > __trm_app_info_send_socket
05-09 22:48:08.285+0900 E/AUL     ( 2170): simple_util.c: __trm_app_info_send_socket(264) > access
05-09 22:48:08.285+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(170) > [SECURE_LOG] __add_item_running_list pid: 3011
05-09 22:48:08.290+0900 D/AUL_AMD ( 2170): amd_main.c: __add_item_running_list(183) > [SECURE_LOG] __add_item_running_list appid: org.tizen.task-mgr
05-09 22:48:08.290+0900 D/AUL     ( 2225): launch.c: app_request_to_launchpad(295) > launch request result : 3011
05-09 22:48:08.290+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: main(461) > Application Main Function 
05-09 22:48:08.290+0900 D/LAUNCH  ( 3011): appcore-efl.c: appcore_efl_main(1569) > [task-mgr:Application:main:done]
05-09 22:48:08.290+0900 D/STARTER ( 2225): dbus-util.c: starter_dbus_set_oomadj(223) > [starter_dbus_set_oomadj:223] org.tizen.system.deviced.Process-oomadj_set : 0
05-09 22:48:08.295+0900 D/RESOURCED( 2369): proc-noti.c: recv_str(87) > [recv_str,87] str is null
05-09 22:48:08.295+0900 D/RESOURCED( 2369): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2170
05-09 22:48:08.295+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid org.tizen.task-mgr, pid 3011, type 4 
05-09 22:48:08.295+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_status_change(570) > [SECURE_LOG] [resourced_proc_status_change,570] launch request org.tizen.task-mgr, 3011
05-09 22:48:08.295+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_status_change(572) > [SECURE_LOG] [resourced_proc_status_change,572] launch request org.tizen.task-mgr with pkgname
05-09 22:48:08.295+0900 D/RESOURCED( 2369): net-cls-cgroup.c: make_net_cls_cgroup_with_pid(311) > [SECURE_LOG] [make_net_cls_cgroup_with_pid,311] pkg: org; pid: 3011
05-09 22:48:08.295+0900 D/RESOURCED( 2369): cgroup.c: cgroup_write_node(89) > [SECURE_LOG] [cgroup_write_node,89] cgroup_buf /sys/fs/cgroup/net_cls/org/cgroup.procs, value 3011
05-09 22:48:08.295+0900 D/RESOURCED( 2369): net-cls-cgroup.c: update_classids(249) > [update_classids,249] class id table updated
05-09 22:48:08.295+0900 D/RESOURCED( 2369): cgroup.c: cgroup_read_node(107) > [SECURE_LOG] [cgroup_read_node,107] cgroup_buf /sys/fs/cgroup/net_cls/org/net_cls.classid, value 0
05-09 22:48:08.295+0900 D/RESOURCED( 2369): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
05-09 22:48:08.295+0900 D/RESOURCED( 2369): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
05-09 22:48:08.295+0900 D/RESOURCED( 2369): datausage-common.c: create_each_iptable_rule(689) > [create_each_iptable_rule,689] Unsupported network interface type 0
05-09 22:48:08.295+0900 D/RESOURCED( 2369): datausage-common.c: create_each_iptable_rule(697) > [create_each_iptable_rule,697] Counter already exists!
05-09 22:48:08.295+0900 E/RESOURCED( 2369): proc-main.c: resourced_proc_status_change(577) > [resourced_proc_status_change,577] available memory = 541
05-09 22:48:08.295+0900 D/RESOURCED( 2369): proc-noti.c: safe_write_int(178) > [safe_write_int,178] Response is not needed
05-09 22:48:08.305+0900 D/AUL     ( 3011): pkginfo.c: aul_app_get_appid_bypid(196) > [SECURE_LOG] appid for 3011 is org.tizen.task-mgr
05-09 22:48:08.445+0900 D/APP_CORE( 3011): appcore-efl.c: __before_loop(1012) > elm_config_preferred_engine_set : opengl_x11
05-09 22:48:08.455+0900 D/AUL     ( 3011): pkginfo.c: aul_app_get_pkgid_bypid(257) > [SECURE_LOG] appid for 3011 is org.tizen.task-mgr
05-09 22:48:08.455+0900 D/APP_CORE( 3011): appcore.c: appcore_init(532) > [SECURE_LOG] dir : /usr/apps/org.tizen.task-mgr/res/locale
05-09 22:48:08.455+0900 D/APP_CORE( 3011): appcore-i18n.c: update_region(71) > *****appcore setlocale=en_US.UTF-8
05-09 22:48:08.460+0900 D/AUL     ( 3011): app_sock.c: __create_server_sock(135) > pg path - already exists
05-09 22:48:08.460+0900 D/LAUNCH  ( 3011): appcore-efl.c: __before_loop(1035) > [task-mgr:Platform:appcore_init:done]
05-09 22:48:08.460+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: _create_app(331) > Application Create Callback 
05-09 22:48:08.615+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: create_layout(197) > create_layout
05-09 22:48:08.615+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: create_layout(216) > Resolution: HD
05-09 22:48:08.620+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: load_data(282) > load_data
05-09 22:48:08.620+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_app_list_create(625) > recent_app_list_create
05-09 22:48:08.620+0900 D/PKGMGR_INFO( 3011): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_filter_foreach_appinfo(3109) > [SECURE_LOG] where = package_app_info.app_taskmanage IN ('true','True') and package_app_info.app_disable IN ('false','False')
05-09 22:48:08.620+0900 D/PKGMGR_INFO( 3011): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_filter_foreach_appinfo(3115) > [SECURE_LOG] query = select DISTINCT package_app_info.*, package_app_localized_info.app_locale, package_app_localized_info.app_label, package_app_localized_info.app_icon from package_app_info LEFT OUTER JOIN package_app_localized_info ON package_app_info.app_id=package_app_localized_info.app_id and package_app_localized_info.app_locale IN ('No Locale', 'en-us') LEFT OUTER JOIN package_app_app_svc ON package_app_info.app_id=package_app_app_svc.app_id LEFT OUTER JOIN package_app_app_category ON package_app_info.app_id=package_app_app_category.app_id where package_app_info.app_taskmanage IN ('true','True') and package_app_info.app_disable IN ('false','False')
05-09 22:48:08.630+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 12
05-09 22:48:08.630+0900 D/AUL     ( 3011): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 343
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_app_list_create(653) > USP mode is disabled.
05-09 22:48:08.630+0900 E/RUA     ( 3011): rua.c: rua_history_load_db(278) > rua_history_load_db ok. nrows : 2, ncols : 5
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_app_list_create(680) > Apps in history: 2
05-09 22:48:08.630+0900 E/TASK_MGR_LITE( 3011): recent_apps.c: list_retrieve_item(348) > Fail to get ai table !!
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: list_retrieve_item(444) > [org.tizen.other]
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): [/opt/usr/apps/org.tizen.other/shared/res/favicon.png]
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): [Nornenjs]
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_app_list_create(741) > App added into the running_list : pkgid:org.tizen.other - appid:org.tizen.other
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_app_list_create(773) > HISTORY LIST: (nil) 0
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_app_list_create(774) > RUNNING LIST: 0x173e68 1
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: load_data(292) > LISTs : 0x1bd6c8, 0x1bde28
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: load_data(304) > App list should display !!
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): genlist.c: genlist_init(258) > in gen list: 0x173e68 (nil)
05-09 22:48:08.630+0900 D/TASK_MGR_LITE( 3011): genlist.c: recent_app_panel_create(98) > Creating task_mgr_genlist widget, noti count is : [-1]
05-09 22:48:08.650+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: genlist_item_class_create(308) > Genlist item class create
05-09 22:48:08.650+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: genlist_clear_all_class_create(323) > Genlist clear all class create
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): genlist.c: genlist_update(432) > genlist_update
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): genlist.c: genlist_clear_list(297) > genlist_clear_list
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): genlist.c: add_clear_btn_to_genlist(417) > add_clear_btn_to_genlist
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): genlist.c: genlist_add_item(165) > genlist_add_item
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): genlist.c: genlist_add_item(171) > Adding item: 0x1ba800
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_apps_find_by_appid(210) > recent_apps_find_by_appid
05-09 22:48:08.665+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 12
05-09 22:48:08.665+0900 D/AUL     ( 3011): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 343
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.lockscreen (2228)
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.volume (2238)
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.menu-screen (2240)
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.other (2992)
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_apps_find_by_appid_cb(201) > FOUND
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_apps_find_by_appid_cb(197) > Comparing: org.tizen.other <> org.tizen.task-mgr (3011)
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): genlist.c: genlist_update(449) > Are there recent apps: (nil)
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): genlist.c: genlist_update(450) > Adding recent apps... 0
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): genlist.c: genlist_item_show(192) > genlist_item_show
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): genlist.c: genlist_item_show(199) > ELM COUNT = 2
05-09 22:48:08.665+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: load_data(323) > load_data END. 
05-09 22:48:08.665+0900 D/LAUNCH  ( 3011): appcore-efl.c: __before_loop(1045) > [task-mgr:Application:create:done]
05-09 22:48:08.665+0900 D/APP_CORE( 3011): appcore-efl.c: __check_wm_rotation_support(752) > Disable window manager rotation
05-09 22:48:08.670+0900 D/APP_CORE( 3011): appcore.c: __aul_handler(423) > [APP 3011]     AUL event: AUL_START
05-09 22:48:08.670+0900 D/APP_CORE( 3011): appcore-efl.c: __do_app(470) > [APP 3011] Event: RESET State: CREATED
05-09 22:48:08.670+0900 D/APP_CORE( 3011): appcore-efl.c: __do_app(496) > [APP 3011] RESET
05-09 22:48:08.670+0900 D/LAUNCH  ( 3011): appcore-efl.c: __do_app(498) > [task-mgr:Application:reset:start]
05-09 22:48:08.670+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: _reset_app(446) > _reset_app
05-09 22:48:08.670+0900 D/LAUNCH  ( 3011): appcore-efl.c: __do_app(501) > [task-mgr:Application:reset:done]
05-09 22:48:08.670+0900 E/PKGMGR_INFO( 3011): pkgmgrinfo_appinfo.c: pkgmgrinfo_appinfo_get_appinfo(1308) > (appid == NULL) appid is NULL
05-09 22:48:08.670+0900 I/APP_CORE( 3011): appcore-efl.c: __do_app(507) > Legacy lifecycle: 0
05-09 22:48:08.670+0900 I/APP_CORE( 3011): appcore-efl.c: __do_app(509) > [APP 3011] Initial Launching, call the resume_cb
05-09 22:48:08.670+0900 D/APP_CORE( 3011): appcore.c: __aul_handler(426) > [SECURE_LOG] caller_appid : (null)
05-09 22:48:08.670+0900 D/APP_CORE( 3011): appcore.c: __prt_ltime(183) > [APP 3011] first idle after reset: 499 msec
05-09 22:48:08.675+0900 D/APP_CORE( 3011): appcore-efl.c: __show_cb(826) > [EVENT_TEST][EVENT] GET SHOW EVENT!!!. WIN:2a00008
05-09 22:48:08.675+0900 D/APP_CORE( 3011): appcore-efl.c: __add_win(672) > [EVENT_TEST][EVENT] __add_win WIN:2a00008
05-09 22:48:08.695+0900 W/PROCESSMGR( 2140): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=3011
05-09 22:48:08.700+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 1c00003"
05-09 22:48:08.700+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 2a00008"
05-09 22:48:08.700+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
05-09 22:48:08.705+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 2a00008"
05-09 22:48:08.705+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
05-09 22:48:08.745+0900 D/RESOURCED( 2369): proc-monitor.c: proc_dbus_proc_signal_handler(198) > [proc_dbus_proc_signal_handler,198] call proc_dbus_proc_signal_handler : pid = 3011, type = 0
05-09 22:48:08.745+0900 D/AUL_AMD ( 2170): amd_launch.c: __e17_status_handler(1888) > pid(3011) status(3)
05-09 22:48:08.745+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_status_change(555) > [SECURE_LOG] [resourced_proc_status_change,555] set foreground : 3011
05-09 22:48:08.745+0900 I/RESOURCED( 2369): lowmem-handler.c: lowmem_move_memcgroup(1185) > [lowmem_move_memcgroup,1185] buf : /sys/fs/cgroup/memory/foreground/cgroup.procs, pid : 3011, oom : 200
05-09 22:48:08.745+0900 E/RESOURCED( 2369): lowmem-handler.c: lowmem_move_memcgroup(1188) > [lowmem_move_memcgroup,1188] /sys/fs/cgroup/memory/foreground/cgroup.procs open failed
05-09 22:48:08.760+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _clear_all_btn_show_cb(243) > _clear_all_btn_show_cb
05-09 22:48:08.765+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
05-09 22:48:08.765+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.other/shared/res/favicon.png; Nornenjs 0x21e8f0
05-09 22:48:08.765+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_set(169) > It is already set.
05-09 22:48:08.765+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: recent_app_item_create(543) > 
05-09 22:48:08.765+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_main_create(765) > 
05-09 22:48:08.775+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
05-09 22:48:08.775+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
05-09 22:48:08.775+0900 E/TASK_MGR_LITE( 3011): genlist_item.c: del_cb(758) > Deleted
05-09 22:48:08.790+0900 D/APP_CORE( 3011): appcore-efl.c: __update_win(718) > [EVENT_TEST][EVENT] __update_win WIN:2a00008 fully_obscured 0
05-09 22:48:08.790+0900 D/APP_CORE( 3011): appcore-efl.c: __visibility_cb(884) > bvisibility 1, b_active -1
05-09 22:48:08.790+0900 D/APP_CORE( 3011): appcore-efl.c: __visibility_cb(887) >  Go to Resume state
05-09 22:48:08.790+0900 D/APP_CORE( 3011): appcore-efl.c: __do_app(470) > [APP 3011] Event: RESUME State: RUNNING
05-09 22:48:08.790+0900 D/LAUNCH  ( 3011): appcore-efl.c: __do_app(557) > [task-mgr:Application:resume:start]
05-09 22:48:08.790+0900 D/LAUNCH  ( 3011): appcore-efl.c: __do_app(567) > [task-mgr:Application:resume:done]
05-09 22:48:08.790+0900 D/LAUNCH  ( 3011): appcore-efl.c: __do_app(569) > [task-mgr:Application:Launching:done]
05-09 22:48:08.790+0900 D/APP_CORE( 3011): appcore-efl.c: __trm_app_info_send_socket(230) > __trm_app_info_send_socket
05-09 22:48:08.790+0900 E/APP_CORE( 3011): appcore-efl.c: __trm_app_info_send_socket(233) > access
05-09 22:48:08.795+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _clear_all_btn_show_cb(243) > _clear_all_btn_show_cb
05-09 22:48:08.795+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_set(152) > _recent_app_item_content_set
05-09 22:48:08.795+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_set(160) > Setting content: /opt/usr/apps/org.tizen.other/shared/res/favicon.png; Nornenjs 0x21e8f0
05-09 22:48:08.795+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: recent_app_item_create(543) > 
05-09 22:48:08.795+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_main_create(765) > 
05-09 22:48:08.800+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_set(179) > Parent is: elm_genlist
05-09 22:48:08.800+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_set(183) > _recent_app_item_content_set END
05-09 22:48:09.105+0900 D/STARTER ( 2225): hw_key.c: _key_release_cb(470) > [_key_release_cb:470] _key_release_cb : XF86Phone Released
05-09 22:48:09.105+0900 D/MENU_SCREEN( 2240): key.c: _key_release_cb(68) > Key(XF86Phone) released 1
05-09 22:48:09.105+0900 W/STARTER ( 2225): hw_key.c: _key_release_cb(498) > [_key_release_cb:498] Home Key is released
05-09 22:48:09.105+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(337) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:337:D] [ == lockd_get_hall_status == ]
05-09 22:48:09.105+0900 D/STARTER ( 2225): lock-daemon-lite.c: lockd_get_hall_status(354) > [STARTER/home/abuild/rpmbuild/BUILD/starter-0.5.52/src/lock-daemon-lite.c:354:D] org.tizen.system.deviced.hall-getstatus : 1
05-09 22:48:09.115+0900 D/MENU_SCREEN( 2240): page_scroller.c: page_scroller_trim_items(989) > LIST APPEND : org.tizen.other
05-09 22:48:09.115+0900 D/MENU_SCREEN( 2240): page_scroller.c: page_scroller_trim_items(989) > LIST APPEND : org.tizen.setting
05-09 22:48:09.115+0900 D/MENU_SCREEN( 2240): page_scroller.c: page_scroller_trim_items(998) > PAGE GET : 0xb715b158
05-09 22:48:09.115+0900 D/MENU_SCREEN( 2240): page_scroller.c: page_scroller_trim_items(1010) > LIST GET : [0] org.tizen.other
05-09 22:48:09.115+0900 D/MENU_SCREEN( 2240): page_scroller.c: page_scroller_trim_items(1010) > LIST GET : [1] org.tizen.setting
05-09 22:48:09.115+0900 D/BADGE   ( 2240): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.other'], count[0]
05-09 22:48:09.115+0900 D/STARTER ( 2225): hw_key.c: _key_release_cb(536) > [_key_release_cb:536] delete cancelkey timer
05-09 22:48:09.120+0900 D/BADGE   ( 2240): badge_internal.c: _badge_check_data_inserted(154) > [SECURE_LOG] [_badge_check_data_inserted : 154] [SELECT count(*) FROM badge_data WHERE pkgname = 'org.tizen.setting'], count[0]
05-09 22:48:09.120+0900 I/SYSPOPUP( 2238): syspopup.c: __X_syspopup_term_handler(98) > enter syspopup term handler
05-09 22:48:09.120+0900 I/SYSPOPUP( 2238): syspopup.c: __X_syspopup_term_handler(108) > term action 1 - volume
05-09 22:48:09.120+0900 D/VOLUME  ( 2238): volume_control.c: volume_control_close(667) > [volume_control_close:667] Start closing volume
05-09 22:48:09.120+0900 E/VOLUME  ( 2238): volume_x_event.c: volume_x_input_event_unregister(349) > [volume_x_input_event_unregister:349] (s_info.event_outer_touch_handler == NULL) -> volume_x_input_event_unregister() return
05-09 22:48:09.120+0900 E/VOLUME  ( 2238): volume_control.c: volume_control_close(680) > [volume_control_close:680] Failed to unregister x input event handler
05-09 22:48:09.120+0900 D/VOLUME  ( 2238): volume_key_event.c: volume_key_event_key_ungrab(166) > [volume_key_event_key_ungrab:166] key ungrabed
05-09 22:48:09.120+0900 D/VOLUME  ( 2238): volume_control.c: volume_control_close(692) > [volume_control_close:692] ungrab key : 1/1
05-09 22:48:09.120+0900 D/VOLUME  ( 2238): volume_key_event.c: volume_key_event_key_grab(115) > [volume_key_event_key_grab:115] count_grabed : 1
05-09 22:48:09.125+0900 E/VOLUME  ( 2238): volume_view.c: volume_view_setting_icon_callback_del(519) > [volume_view_setting_icon_callback_del:519] (!s_info.is_registered_callback) -> volume_view_setting_icon_callback_del() return
05-09 22:48:09.125+0900 D/VOLUME  ( 2238): volume_control.c: volume_control_close(714) > [volume_control_close:714] End closing volume
05-09 22:48:09.295+0900 D/AUL_AMD ( 2170): amd_request.c: __add_history_handler(243) > [SECURE_LOG] add rua history org.tizen.task-mgr /usr/apps/org.tizen.task-mgr/bin/task-mgr
05-09 22:48:09.295+0900 D/RUA     ( 2170): rua.c: rua_add_history(179) > rua_add_history start
05-09 22:48:09.305+0900 D/RUA     ( 2170): rua.c: rua_add_history(247) > rua_add_history ok
05-09 22:48:09.670+0900 D/EFL     ( 3011): ecore_x<3011> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=216892 button=1
05-09 22:48:09.685+0900 D/PROCESSMGR( 2140): e_mod_processmgr.c: _e_mod_processmgr_anr_ping_confirm_handler(336) > [PROCESSMGR] last_pointed_win=0x6004e0 bd->visible=0
05-09 22:48:09.810+0900 D/EFL     ( 3011): ecore_x<3011> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=217029 button=1
05-09 22:48:09.810+0900 D/PROCESSMGR( 2140): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600045  register trigger_timer!  pointed_win=0x6005d5 
05-09 22:48:09.810+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
05-09 22:48:09.810+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected list 0.000000 no_shutdown: 0
05-09 22:48:09.810+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
05-09 22:48:09.810+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected layout 0.000000 no_shutdown: 0
05-09 22:48:10.540+0900 D/EFL     ( 3011): ecore_x<3011> ecore_x_events.c:531 _ecore_x_event_handle_button_press() ButtonEvent:press time=217760 button=1
05-09 22:48:10.615+0900 D/EFL     ( 3011): ecore_x<3011> ecore_x_events.c:683 _ecore_x_event_handle_button_release() ButtonEvent:release time=217839 button=1
05-09 22:48:10.615+0900 D/PROCESSMGR( 2140): e_mod_processmgr.c: _e_mod_processmgr_anr_ping(466) > [PROCESSMGR] ev_win=0x600045  register trigger_timer!  pointed_win=0x6005d5 
05-09 22:48:10.620+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
05-09 22:48:10.620+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected list 0.000000 no_shutdown: 0
05-09 22:48:10.620+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: on_swipe_gesture(149) > on_swipe_gesture
05-09 22:48:10.620+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: on_swipe_gesture(153) > FLICK detected layout 0.000000 no_shutdown: 0
05-09 22:48:10.620+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: clear_all_btn_clicked_cb(218) > Removing all items...
05-09 22:48:10.620+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: clear_all_btn_clicked_cb(230) > On REDWOOD target (HD) : No Animation.
05-09 22:48:10.620+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_apps_kill_all(164) > recent_apps_kill_all
05-09 22:48:10.620+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 12
05-09 22:48:10.620+0900 D/AUL     ( 3011): app_sock.c: __app_send_cmd_with_result(608) > recv result  = 343
05-09 22:48:10.620+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: get_app_taskmanage(121) > app org.tizen.lockscreen is taskmanage: 0
05-09 22:48:10.640+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: get_app_taskmanage(121) > app org.tizen.volume is taskmanage: 0
05-09 22:48:10.640+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: get_app_taskmanage(121) > app org.tizen.menu-screen is taskmanage: 0
05-09 22:48:10.650+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: get_app_taskmanage(121) > app org.tizen.other is taskmanage: 1
05-09 22:48:10.650+0900 D/AUL     ( 3011): launch.c: app_request_to_launchpad(281) > [SECURE_LOG] launch request : 2992
05-09 22:48:10.650+0900 D/AUL     ( 3011): app_sock.c: __app_send_raw(264) > pid(-2) : cmd(4)
05-09 22:48:10.655+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 4
05-09 22:48:10.680+0900 D/RESOURCED( 2369): proc-noti.c: recv_str(87) > [recv_str,87] str is null
05-09 22:48:10.680+0900 D/RESOURCED( 2369): proc-noti.c: process_message(169) > [process_message,169] process message caller pid 2170
05-09 22:48:10.680+0900 D/RESOURCED( 2369): proc-main.c: resourced_proc_action(669) > [SECURE_LOG] [resourced_proc_action,669] appid (null), pid 2992, type 6 
05-09 22:48:10.680+0900 E/RESOURCED( 2369): datausage-common.c: app_terminate_cb(254) > [app_terminate_cb,254] No classid to terminate!
05-09 22:48:10.680+0900 D/RESOURCED( 2369): proc-main.c: proc_remove_process_list(363) > [proc_remove_process_list,363] found_pid 2992
05-09 22:48:10.680+0900 D/AUL_AMD ( 2170): amd_request.c: __app_process_by_pid(175) > __app_process_by_pid, cmd: 4, pid: 2992, 
05-09 22:48:10.680+0900 D/AUL     ( 2170): app_sock.c: __app_send_raw_with_delay_reply(434) > pid(2992) : cmd(4)
05-09 22:48:10.680+0900 D/AUL_AMD ( 2170): amd_launch.c: _term_app(878) > term done
05-09 22:48:10.680+0900 D/AUL_AMD ( 2170): amd_launch.c: __set_reply_handler(860) > listen fd : 28, send fd : 17
05-09 22:48:10.700+0900 D/AUL_AMD ( 2170): amd_launch.c: __reply_handler(784) > listen fd(28) , send fd(17), pid(2992), cmd(4)
05-09 22:48:10.700+0900 D/AUL     ( 3011): launch.c: app_request_to_launchpad(295) > launch request result : 0
05-09 22:48:10.700+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: kill_pid(141) > terminate pid = 2992
05-09 22:48:10.700+0900 D/APP_CORE( 2992): appcore.c: __aul_handler(443) > [APP 2992]     AUL event: AUL_TERMINATE
05-09 22:48:10.700+0900 D/APP_CORE( 2992): appcore-efl.c: __do_app(470) > [APP 2992] Event: TERMINATE State: PAUSED
05-09 22:48:10.700+0900 D/APP_CORE( 2992): appcore-efl.c: __do_app(486) > [APP 2992] TERMINATE
05-09 22:48:10.700+0900 D/AUL     ( 2992): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
05-09 22:48:10.700+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 22
05-09 22:48:10.705+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: get_app_taskmanage(121) > app org.tizen.task-mgr is taskmanage: 0
05-09 22:48:10.705+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_del(189) > _recent_app_item_content_del
05-09 22:48:10.705+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_del(194) > Data exist: 0x2b77a8
05-09 22:48:10.705+0900 E/TASK_MGR_LITE( 3011): genlist_item.c: del_cb(758) > Deleted
05-09 22:48:10.710+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _recent_app_item_content_del(198) > Item deleted
05-09 22:48:10.710+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _clear_all_button_del(206) > _clear_all_button_del
05-09 22:48:10.710+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: _clear_all_button_del(213) > _clear_all_button_del END
05-09 22:48:10.725+0900 D/AUL     ( 3011): app_sock.c: __app_send_raw_with_noreply(363) > pid(-2) : cmd(22)
05-09 22:48:10.725+0900 D/AUL_AMD ( 2170): amd_request.c: __request_handler(491) > __request_handler: 22
05-09 22:48:10.725+0900 D/APP_CORE( 3011): appcore-efl.c: __after_loop(1060) > [APP 3011] PAUSE before termination
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: _pause_app(453) > _pause_app
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: _terminate_app(393) > _terminate_app
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): genlist.c: genlist_clear_list(297) > genlist_clear_list
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): genlist.c: recent_app_panel_del_cb(84) > recent_app_panel_del_cb
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): genlist_item.c: genlist_item_class_destroy(340) > Genlist class free
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): task-mgr-lite.c: delete_layout(272) > delete_layout
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: recent_app_list_destroy(791) > recent_app_list_destroy
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: list_destroy(475) > START list_destroy
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: list_destroy(487) > FREE ALL list_destroy
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: list_destroy(493) > FREING: list_destroy org.tizen.other
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: list_unretrieve_item(295) > FREING 0x1ba800 org.tizen.other 's Item(list_type_default_s) in list_unretrieve_item
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: list_unretrieve_item(333) > list_unretrieve_item END 
05-09 22:48:10.725+0900 D/TASK_MGR_LITE( 3011): recent_apps.c: list_destroy(500) > END list_destroy
05-09 22:48:10.735+0900 E/APP_CORE( 3011): appcore.c: appcore_flush_memory(583) > Appcore not initialized
05-09 22:48:10.740+0900 W/PROCESSMGR( 2140): e_mod_processmgr.c: _e_mod_processmgr_send_mDNIe_action(577) > [PROCESSMGR] =====================> Broadcast mDNIeStatus : PID=2240
05-09 22:48:10.745+0900 D/PROCESSMGR( 2140): e_mod_processmgr.c: _e_mod_processmgr_wininfo_del(141) > [PROCESSMGR] delete anr_trigger_timer!
05-09 22:48:10.755+0900 D/indicator( 2218): indicator_ui.c: _property_changed_cb(1198) > _property_changed_cb[1198]	 "UNSNIFF API 2a00008"
05-09 22:48:10.755+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1107) > [SECURE_LOG] _active_indicator_handle[1107]	 "Type : 1, opacity 0, active_win 1c00003"
05-09 22:48:10.755+0900 D/indicator( 2218): indicator_util.c: indicator_signal_emit_by_win(142) > [SECURE_LOG] indicator_signal_emit_by_win[142]	 "emission 0 bg.opaque"
05-09 22:48:10.755+0900 D/indicator( 2218): indicator_ui.c: _active_indicator_handle(1113) > [SECURE_LOG] _active_indicator_handle[1113]	 "Type : 2, angle 0, active_win 1c00003"
05-09 22:48:10.755+0900 D/indicator( 2218): indicator_ui.c: _rotate_window(644) > _rotate_window[644]	 "_rotate_window = 0"
05-09 22:48:10.875+0900 I/CAPI_APPFW_APPLICATION( 2992): app_main.c: _ui_app_appcore_terminate(577) > app_appcore_terminate
05-09 22:48:11.285+0900 W/CRASH_MANAGER( 3016): worker.c: worker_job(1189) > 11029926f7468143117929
